import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import heapq

class RobotSlotPlanner:
    def __init__(self, sim_config_path="configs/sim_config.yaml"):
        # -------------------------------
        # Load simulation parameters
        # -------------------------------
        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
            self.L = (self.sim_config["target"]["size"][0]) / 2   # target half-length
            self.r = self.sim_config["robot"]["radius"]           # robot radius

        # -------------------------------
        # Planner parameters
        # -------------------------------
        horizon = 30
        grid_size = (10,10)
        cell_size = 2*self.r
        self.grid_size = grid_size    # grid 범위 (셀 개수, 예: 100x100)
        self.horizon = horizon        # planning horizon
        self.cell_size = cell_size    # grid cell 크기 (>= 2r 권장)

    # -------------------------------
    # Step 1: Slot Assignment
    # -------------------------------
    def assign_slots(self, robots, obj_state, delta):
        candidate_slots = [i for i, v in enumerate(delta) if v == 1]

        # object transform
        obj_pos = np.array(obj_state["position"][:2])
        obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)
        obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))
        R = np.array([[np.cos(obj_ori), -np.sin(obj_ori)],
                      [np.sin(obj_ori),  np.cos(obj_ori)]])

        # 슬롯 world 좌표
        slot_positions = []
        for idx in candidate_slots:
            local_pos, _ = self._slot_local(idx)
            world_pos = obj_pos + R @ local_pos
            slot_positions.append((idx, world_pos))

        # 비용 행렬
        cost = np.zeros((len(robots), len(slot_positions)))
        for i, robot in enumerate(robots):
            rpos = np.array(robot["position"][:2])
            for j, (slot_idx, spos) in enumerate(slot_positions):
                cost[i, j] = np.linalg.norm(rpos - spos)

        # Hungarian 최적화
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
        new_delta_indicator = [slot_positions[col][0] for _, col in pairs]

        return new_delta_indicator

    # -------------------------------
    # Step 2: Path Planning (Body Frame HCA*)
    # -------------------------------
    def plan_paths(self, robots, obj_state, new_delta_indicator, previous_delta_indicator):
        reservation = {}
        trajectories = defaultdict(list)

        # object transform
        obj_pos = np.array(obj_state["position"][:2])
        obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)
        obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))
        R = np.array([[np.cos(obj_ori), -np.sin(obj_ori)],
                    [np.sin(obj_ori),  np.cos(obj_ori)]])   # world <- body
        R_T = R.T                                           # body <- world

        for i, r in enumerate(robots):
            slot_idx = new_delta_indicator[i]
            goal_local, _ = self._slot_local(slot_idx)  # body frame slot 좌표

            # world → body frame 변환
            start_body = R_T @ (np.array(r["position"][:2]) - obj_pos)
            goal_body = goal_local
            goal_world = obj_pos + R @ goal_local

            if (previous_delta_indicator[i] == new_delta_indicator[i]):
                cont_path = [r["position"][:2]]  # 현재 위치
                cont_path.append(goal_world)     # goal (slot)
                if len(cont_path) < self.horizon:
                    cont_path += [goal_world] * (self.horizon - len(cont_path))
                trajectories[i] = cont_path
                continue

            # body → grid (round 사용)
            start = self._to_grid(start_body)
            goal = self._to_grid(goal_body)

            # A* with reservation (body frame)
            path, true_goal = self._a_star_with_reservation(start, goal, reservation)

            # reservation 기록
            for t, p in enumerate(path):
                reservation[(p[0], p[1], t)] = i

            # body → world 변환
            cont_path = []
            for idx, p in enumerate(path):
                if idx == 0:
                    world_xy = obj_pos + R @ start_body
                    cont_path.append(world_xy)
                if idx > 0:
                    body_xy = self._to_continuous(p)
                    world_xy = obj_pos + R @ body_xy
                    cont_path.append(world_xy)

            # true goal append
            true_goal_world = obj_pos + R @ goal_local
            cont_path.append(true_goal_world)

            # horizon 맞추기
            if len(cont_path) < self.horizon:
                cont_path += [true_goal_world] * (self.horizon - len(cont_path))

            trajectories[i] = cont_path

        return trajectories

    # -------------------------------
    # Step 3: Trajectory Smoothing (start/goal 보존)
    # -------------------------------
    def smooth_paths(self, trajectories):
        smoothed = {}
        for rid, path in trajectories.items():
            if len(path) < 3:
                smoothed[rid] = path
                continue

            new_path = [path[0]]  # 시작점 보존
            for i in range(1, len(path)-1):
                prev, curr, nxt = path[i-1], path[i], path[i+1]
                smoothed_pt = (prev + curr + nxt) / 3.0
                new_path.append(smoothed_pt)
            new_path.append(path[-1])  # true goal 보존

            smoothed[rid] = new_path
        return smoothed
    
    def _remove_duplicates(self, path):
        unique_path = [path[0]]
        for p in path[1:]:
            if not np.allclose(p, unique_path[-1]):  # 연속 중복 제거
                unique_path.append(p)
        return np.array(unique_path)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _slot_local(self, idx):
        if idx == 0:
            return np.array([-self.L, -2*self.L]) + np.array([0, -self.r]), np.pi/2
        elif idx == 1:
            return np.array([self.L, -2*self.L]) + np.array([0, -self.r]), np.pi/2
        elif idx == 2:
            return np.array([2*self.L, -self.L]) + np.array([self.r, 0]), np.pi
        elif idx == 3:
            return np.array([2*self.L, self.L]) + np.array([self.r, 0]), np.pi
        elif idx == 4:
            return np.array([self.L, 2*self.L]) + np.array([0, self.r]), 3*np.pi/2
        elif idx == 5:
            return np.array([-self.L, 2*self.L]) + np.array([0, self.r]), 3*np.pi/2
        elif idx == 6:
            return np.array([-2*self.L, self.L]) + np.array([-self.r, 0]), 0
        elif idx == 7:
            return np.array([-2*self.L, -self.L]) + np.array([-self.r, 0]), 0
        else:
            raise ValueError(f"Invalid slot idx: {idx}")

    def _heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _neighbors(self, node):
        x, y = node
        moves = [(1,0), (-1,0), (0,1), (0,-1)]
        return [(x+dx, y+dy) for dx, dy in moves
                if abs(x+dx) < self.grid_size[0]//2 and abs(y+dy) < self.grid_size[1]//2]

    def _to_grid(self, pos):
        return (int(round(pos[0]/self.cell_size)), int(round(pos[1]/self.cell_size)))

    def _to_continuous(self, cell):
        return np.array([cell[0]*self.cell_size, cell[1]*self.cell_size])

    def _a_star_with_reservation(self, start, goal, reservation):
        # --- target object blocked region (inflate with r) ---
        blocked = set()
        half = int((2*self.L + self.r) / self.cell_size)
        for gx in range(-half, half+1):
            for gy in range(-half, half+1):
                blocked.add((gx, gy))

        # goal이 blocked에 들어가면 → 가장 가까운 free cell 찾기
        true_goal = goal
        if goal in blocked:
            min_dist, nearest_free = 1e9, None
            for dx in range(-half-2, half+3):
                for dy in range(-half-2, half+3):
                    if (dx,dy) not in blocked:
                        dist = self._heuristic(goal, (dx,dy))
                        if dist < min_dist:
                            min_dist, nearest_free = dist, (dx,dy)
            goal = nearest_free

        # --- A* 탐색 ---
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, [start]))
        visited = set()

        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            if g > self.horizon: break
            if current == goal:
                return path, true_goal
            if (current, g) in visited: continue
            visited.add((current,g))

            for nx,ny in self._neighbors(current):
                if (nx,ny) in blocked: continue
                if (nx,ny,g+1) in reservation: continue
                new_path = path+[(nx,ny)]
                cost = g+1+self._heuristic((nx,ny),goal)
                heapq.heappush(open_set,(cost,g+1,(nx,ny),new_path))

        return path, true_goal

    # -------------------------------
    # Main API
    # -------------------------------
    def compute(self, robots, obj_state, delta, previous_delta_indicator):
        new_delta_indicator = self.assign_slots(robots, obj_state, delta)
        raw_paths = self.plan_paths(robots, obj_state, new_delta_indicator, previous_delta_indicator)
        smooth_paths = self.smooth_paths(raw_paths)
        return new_delta_indicator, smooth_paths
