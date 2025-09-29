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
        grid_size = (30,30)
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
        orientations = defaultdict(list)

        # object transform
        obj_pos = np.array(obj_state["position"][:2])
        obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)
        obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))
        R = np.array([[np.cos(obj_ori), -np.sin(obj_ori)],
                    [np.sin(obj_ori),  np.cos(obj_ori)]])
        R_T = R.T

        # -------------------------------
        # 로봇별 목표까지 거리 계산
        # -------------------------------
        robot_infos = []
        for i, r in enumerate(robots):
            slot_idx = new_delta_indicator[i]
            goal_local, local_ori = self._slot_local(slot_idx)

            start_body = R_T @ (np.array(r["position"][:2]) - obj_pos)
            goal_body = goal_local
            goal_world = obj_pos + R @ goal_local
            goal_ori = (obj_ori + local_ori) % (2*np.pi)

            dist = np.linalg.norm(start_body - goal_body)

            robot_infos.append({
                "id": i,
                "robot": r,
                "start_body": start_body,
                "goal_body": goal_body,
                "goal_world": goal_world,
                "goal_ori": goal_ori,
                "slot_idx": slot_idx,
                "dist": dist
            })

        # -------------------------------
        # 거리 기준으로 우선순위 정렬
        # -------------------------------
        robot_infos.sort(key=lambda x: x["dist"])  # 가까운 로봇부터

        # -------------------------------
        # 경로 계획
        # -------------------------------
        for info in robot_infos:
            i = info["id"]
            r = info["robot"]
            start_body = info["start_body"]
            goal_body = info["goal_body"]
            goal_world = info["goal_world"]
            goal_ori = info["goal_ori"]

            # 같은 슬롯이면 직선 + 회전 보간만
            if previous_delta_indicator[i] == new_delta_indicator[i]:
                cont_path = [np.array(r["position"][:2]), goal_world]
                start_ori = self._yaw_from_matrix(r["rotation_matrix"])
                ori_path = self._interp_yaw(start_ori, goal_ori, len(cont_path))
                trajectories[i] = cont_path
                orientations[i] = ori_path
                continue

            # A* path
            start = self._to_grid(start_body)
            goal = self._to_grid(goal_body)
            path, _ = self._a_star_with_reservation(start, goal, reservation)

            safety_horizon = int(self.horizon/2)  # 지나간 뒤에도 몇 step 동안 유지

            for t, p in enumerate(path):
                # 상하좌우 포함한 vertex 점유
                # for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                for dx, dy in [(0,0)]:
                    for k in range(safety_horizon):
                        reservation[(p[0]+dx, p[1]+dy, t+k)] = i

                if t > 0:
                    prev = path[t-1]
                    # edge 점유도 상하좌우 확장
                    # for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                    for dx, dy in [(0,0)]:
                        for k in range(safety_horizon):
                            reservation[(prev[0]+dx, prev[1]+dy,
                                        p[0]+dx, p[1]+dy, t+k)] = i

            # body → world 변환
            cont_path = []
            for idx, p in enumerate(path):
                if idx == 0:
                    world_xy = obj_pos + R @ start_body
                else:
                    body_xy = self._to_continuous(p)
                    world_xy = obj_pos + R @ body_xy
                cont_path.append(world_xy)
            cont_path.append(goal_world)

            # orientation 보간
            start_ori = self._yaw_from_matrix(r["rotation_matrix"])
            ori_path = self._interp_yaw(start_ori, goal_ori, len(cont_path))

            trajectories[i] = cont_path
            orientations[i] = ori_path

        return trajectories, orientations


    def _yaw_from_matrix(self, rot_mat_flat):
        R = np.array(rot_mat_flat).reshape(3,3)
        return float(np.arctan2(R[1,0],R[0,0]))
    
    def _interp_yaw(self, start, goal, steps):
        diff = (goal - start + np.pi) % (2*np.pi) - np.pi
        return [(start + diff*(k/(steps-1))) % (2*np.pi) for k in range(steps)]


    # -------------------------------
    # Step 3: Trajectory Smoothing + Resampling (start/goal 보존)
    # -------------------------------
    def smooth_paths(self, trajectories, orientations):
        smoothed_pos = {}
        smoothed_ori = {}
        resample_factor = 30

        # 먼저 각 로봇 경로 smoothing+resample
        temp_pos, temp_ori = {}, {}

        for rid, path in trajectories.items():
            path = np.array(path)
            ori_path = np.array(orientations[rid])

            # --- 간단 smoothing ---
            if len(path) > 2:
                new_path = [path[0]]
                new_ori = [ori_path[0]]
                for i in range(1, len(path)-1):
                    smoothed_pt = (path[i-1] + path[i] + path[i+1]) / 3.0
                    new_path.append(smoothed_pt)
                    smoothed_ori_val = (ori_path[i-1] + ori_path[i] + ori_path[i+1]) / 3.0
                    new_ori.append(smoothed_ori_val % (2*np.pi))
                new_path.append(path[-1])
                new_ori.append(ori_path[-1])
            else:
                new_path, new_ori = path, ori_path

            new_path = np.array(new_path)
            new_ori = np.array(new_ori)

            # --- Resampling (pos + ori 같은 길이) ---
            num_points = len(new_path)
            new_num_points = (num_points - 1) * resample_factor + 1
            t_old = np.linspace(0, 1, num_points)
            t_new = np.linspace(0, 1, new_num_points)

            # position resample
            resampled_pos = np.zeros((new_num_points, 2))
            resampled_pos[:, 0] = np.interp(t_new, t_old, new_path[:, 0])
            resampled_pos[:, 1] = np.interp(t_new, t_old, new_path[:, 1])

            # orientation resample
            resampled_ori = []
            for j in range(len(new_ori) - 1):
                start = new_ori[j]
                goal = new_ori[j + 1]
                diff = (goal - start + np.pi) % (2*np.pi) - np.pi
                segment = [(start + diff * (k / resample_factor)) % (2*np.pi)
                           for k in range(resample_factor)]
                resampled_ori.extend(segment)
            resampled_ori.append(new_ori[-1])
            resampled_ori = np.array(resampled_ori)

            temp_pos[rid] = resampled_pos
            temp_ori[rid] = resampled_ori

        # --- 모든 로봇 경로 길이 통일 ---
        max_len = max(len(p) for p in temp_pos.values())
        for rid in temp_pos:
            pos = temp_pos[rid]
            ori = temp_ori[rid]
            if len(pos) < max_len:
                # 마지막 점을 복제해서 길이 맞추기
                pad_pos = np.vstack([pos, np.tile(pos[-1], (max_len - len(pos), 1))])
                pad_ori = np.concatenate([ori, np.tile(ori[-1], max_len - len(ori))])
            else:
                pad_pos, pad_ori = pos, ori

            smoothed_pos[rid] = pad_pos
            smoothed_ori[rid] = pad_ori

        return smoothed_pos, smoothed_ori

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

    def _a_star_with_reservation(self, start, goal, reservation, agent_id=None):
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
                if (nx,ny) in blocked: 
                    continue

                # --- vertex conflict ---
                if (nx,ny,g+1) in reservation:
                    continue

                # --- edge conflict (swap 방지) ---
                if (current[0],current[1],nx,ny,g+1) in reservation:
                    continue

                new_path = path+[(nx,ny)]
                cost = g+1+self._heuristic((nx,ny),goal)
                heapq.heappush(open_set,(cost,g+1,(nx,ny),new_path))

        return path, true_goal


    # -------------------------------
    # Main API
    # -------------------------------
    def compute(self, robots, obj_state, delta, previous_delta_indicator):
        new_delta_indicator = self.assign_slots(robots, obj_state, delta)
        raw_paths, raw_oris = self.plan_paths(robots, obj_state, new_delta_indicator, previous_delta_indicator)
        smooth_paths, smoothed_ori = self.smooth_paths(raw_paths, raw_oris)
        ext_trajs = {
            "positions": smooth_paths,
            "orientations": smoothed_ori
        }
        return new_delta_indicator, ext_trajs
