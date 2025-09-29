import numpy as np
import yaml
import math
from scipy.integrate import solve_ivp

class PushingEnvODE:
    def __init__(self, sim_config_path="configs/sim_config.yaml", dt=1/240):
        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)

        self.dt = dt
        self.time = 0

        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)

        # parameters
        self.num_robots = self.sim_config["robot"]["num"]
        self.robot_init_id = self.sim_config["robot"]["init_id"]
        self.gravity = self.sim_config["gravity"]
        self.mass_target = self.sim_config["mass"]["target"]
        self.mu = math.sqrt(self.sim_config["friction"]["plane"] * 
                            self.sim_config["friction"]["target"])
        
        self.L = self.sim_config["target"]["size"][0] / 2
        self.r = self.sim_config["robot"]["radius"]
        self.r_dyn = math.sqrt(5)*self.L
        self.c = 1
        self.c_f = self.mu*self.mass_target*self.gravity
        self.c_tau = self.c*self.r_dyn*self.c_f

        self.inertia_target = (1/6) * self.mass_target * (2*self.L)**2

        # target state: [x, y, theta]
        self.state_target = np.array([0, 0, 0], dtype=float)

        # robots: [x, y, rot_mat]
        self.robots = []
        self.modes = []       # "NAV" or "CON"
        self.slot_ids = []    # if "CON", which slot
        self.actions = []     # (vx, vy, omega) for NAV

        positions_cand = [
            [-2*self.L, -4*self.L], [2*self.L, -4*self.L],
            [4*self.L, -2*self.L], [4*self.L,  2*self.L],
            [2*self.L,  4*self.L], [-2*self.L, 4*self.L],
            [-4*self.L, 2*self.L], [-4*self.L,-2*self.L]
        ]
        orientations_cand = np.array([np.pi/2, np.pi/2, np.pi, np.pi,
                                      3*np.pi/2, 3*np.pi/2, 0, 0])

        self.udim = len(positions_cand)

        for i in range(self.num_robots):
            idx = self.robot_init_id[i]
            pos = positions_cand[idx]
            th = orientations_cand[idx]
            R = np.array([
                [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)]
            ])
            self.robots.append({"x": pos[0], "y": pos[1], "R": R})
            self.modes.append("NAV")
            self.slot_ids.append(None)
            self.actions.append((0.0, 0.0, 0.0))  # vx, vy, omega

    # -------------------
    # Target dynamics
    # -------------------
    def _dynamics_target(self, t, q, u, delta):
        """
        Target object dynamics
        q : [x, y, psi]
        u : (udim,) force inputs from robots
        delta : (udim,) actuator selection mask
        """
        psi = float(q[2])
        udim = len(delta)
        # qdim = 3

        # rotation
        R = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)]
        ])
        F = np.block([
            [R, np.zeros((2, 1))],
            [np.zeros((1, 2)), 1]
        ])

        M = np.block([
            [self.c_f * np.eye(2), np.zeros((2, 1))],
            [np.zeros((1, 2)), self.c_tau]
        ])

        # B matrix (slots -> generalized coords)
        B_cols = []
        for k in range(udim):
            if k == 0:  Bk = np.array([0, 1, -self.L])
            if k == 1:  Bk = np.array([0, 1,  self.L])
            if k == 2:  Bk = np.array([-1,0, -self.L])
            if k == 3:  Bk = np.array([-1,0,  self.L])
            if k == 4:  Bk = np.array([0,-1, -self.L])
            if k == 5:  Bk = np.array([0,-1,  self.L])
            if k == 6:  Bk = np.array([1, 0, -self.L])
            if k == 7:  Bk = np.array([1, 0,  self.L])
            # 추가 actuator (udim=12)
            if k == 8:  Bk = np.array([0, 1, 0])
            if k == 9:  Bk = np.array([-1,0,0])
            if k == 10: Bk = np.array([0,-1,0])
            if k == 11: Bk = np.array([1, 0,0])

            B_cols.append(delta[k] * Bk)
        B = np.column_stack(B_cols)

        qdot = F @ np.linalg.inv(M) @ B @ u
        return qdot

    def step_target(self, q, u, delta):
        t_span = [self.time, self.time+self.dt]
        sol = solve_ivp(
            fun=lambda t, y: self._dynamics_target(t, y, u, delta),
            t_span=t_span, y0=q, method="RK45",
            t_eval=[self.time+self.dt],
            rtol=1e-6, atol=1e-9
        )
        return sol.y[:, -1]

    # -------------------
    # Robot update
    # -------------------
    def _normalize_R(self, R):
        u, _, vT = np.linalg.svd(R)
        R_ortho = u @ vT
        if np.linalg.det(R_ortho) < 0:
            u[:, -1] *= -1
            R_ortho = u @ vT
        return R_ortho

    def _slot_local(self, idx):
        L = self.L
        r = self.r
        if idx == 0: return np.array([-L,-2*L]) + np.array([0,-r]), np.pi/2
        if idx == 1: return np.array([ L,-2*L]) + np.array([0,-r]), np.pi/2
        if idx == 2: return np.array([ 2*L,-L]) + np.array([ r, 0]), np.pi
        if idx == 3: return np.array([ 2*L, L]) + np.array([ r, 0]), np.pi
        if idx == 4: return np.array([ L, 2*L]) + np.array([0, r]), 3*np.pi/2
        if idx == 5: return np.array([-L, 2*L]) + np.array([0, r]), 3*np.pi/2
        if idx == 6: return np.array([-2*L, L]) + np.array([-r,0]), 0
        if idx == 7: return np.array([-2*L,-L]) + np.array([-r,0]), 0
        raise ValueError("Invalid slot idx")
    
    # -------------------
    # API
    # -------------------
    def apply_actions(self, vx_list, vy_list, omega_list):
        for i in range(self.num_robots):
            self.actions[i] = (vx_list[i], vy_list[i], omega_list[i])

    def set_mode(self, rid, mode, slot_id=None):
        self.modes[rid] = mode
        self.slot_ids[rid] = slot_id

    def step(self, target_u=None, target_delta=None):
        # 1) target integration
        if target_u is None:  # no external forces
            target_u = np.zeros(self.udim)
        if target_delta is None:
            target_delta = np.ones(self.udim)

        self.state_target = self.step_target(self.state_target, target_u, target_delta)
        self.time += self.dt

        # 2) robot update
        for i in range(self.num_robots):
            if self.modes[i] == "NAV":
                x, y, R = self.robots[i]["x"], self.robots[i]["y"], self.robots[i]["R"]
                vx, vy, omega = self.actions[i]

                x_new = x + self.dt * vx
                y_new = y + self.dt * vy

                dR = np.array([
                    [np.cos(omega*self.dt), -np.sin(omega*self.dt)],
                    [np.sin(omega*self.dt),  np.cos(omega*self.dt)]
                ])
                R_new = R @ dR
                R_new = self._normalize_R(R_new)

                self.robots[i] = {"x": x_new, "y": y_new, "R": R_new}

            elif self.modes[i] == "CON":
                obj_x, obj_y, obj_th = self.state_target
                R_obj = np.array([[np.cos(obj_th), -np.sin(obj_th)],
                                  [np.sin(obj_th),  np.cos(obj_th)]])
                slot_local, slot_ori = self._slot_local(self.slot_ids[i])
                slot_world = np.array([obj_x, obj_y]) + R_obj @ slot_local
                R_slot = np.array([
                    [np.cos(obj_th+slot_ori), -np.sin(obj_th+slot_ori)],
                    [np.sin(obj_th+slot_ori),  np.cos(obj_th+slot_ori)]
                ])
                self.robots[i] = {"x": slot_world[0], "y": slot_world[1], "R": R_slot}

        return self.get_state()

    def get_state(self):
        state_dict = {"robots": [], "target": {}}
        # target
        x, y, th = self.state_target
        R = [np.cos(th), -np.sin(th), 0,
             np.sin(th),  np.cos(th), 0,
             0, 0, 1]
        state_dict["target"] = {
            "position": [x, y, 0],
            "orientation_euler": [0, 0, th],
            "rotation_matrix": R
        }
        # robots
        for rob in self.robots:
            R2 = rob["R"]
            state_dict["robots"].append({
                "position": [rob["x"], rob["y"], 0],
                "rotation_matrix": [R2[0,0], R2[0,1], 0,
                                    R2[1,0], R2[1,1], 0,
                                    0, 0, 1]
            })
        return state_dict