import yaml
import numpy as np
from enum import IntEnum
import time

class Mode(IntEnum):
    CON = 0
    NAV = 1

class RobotCtrller:
    def __init__(self, ctrl_config_path="configs/ctrl_config.yaml", sim_config_path="configs/sim_config.yaml"):

        # Load YAML configuration
        with open(ctrl_config_path, "r") as f:
            self.ctrl_config = yaml.safe_load(f)
            self.kp_t = self.ctrl_config["gain"]["kp_t"]   # position gain
            self.kp_r = self.ctrl_config["gain"]["kp_r"]   # orientation gain

        self.kp_t = 5 # position gain
        self.kp_r = 3 # orientation gain

        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
            self.L = (self.sim_config["target"]["size"][0]) / 2
            self.r = self.sim_config["robot"]["radius"]

        # Initialize robot parameters
        self.num_robots = self.sim_config["robot"]["num"]
        self.delta_indicator = np.array(self.sim_config["robot"]["init_id"])
        # [a,b,...]: 1st robot → slot a, 2nd robot → slot b, etc.

        # Initialize external trajectories
        self.ext_trajs = None
        self.ext_traj_idx = 0

        # Interaction mode, Navigation mode
        self.mode = Mode.NAV
        self.wait_duration = 0.5  # seconds
        self.wait_timer_start = None
    
    # -------------------------
    # Mode switching law
    # -------------------------
    def mode_switch_law(self, trigger=False):
        if self.mode == Mode.CON:
            if trigger:
                self.mode = Mode.NAV
                print("CHANGED TO NAVIGATION MODE")
                self.wait_timer_start = None  # Reset timer
        elif self.mode == Mode.NAV:
            if self.ext_trajs is None:  # Navigation finished
                if self.wait_timer_start is None:
                    # First time hitting this condition
                    self.wait_timer_start = time.time()
                elif time.time() - self.wait_timer_start >= self.wait_duration:
                    # Waited long enough → switch to control
                    self.mode = Mode.CON
                    print("CHANGED TO CONTACT MODE")
                    self.wait_timer_start = None  # Reset for next time

    # -------------------------
    # Compute velocity commands
    # -------------------------
    def compute_actions(self, robot_states, pos_ds, ori_ds, trigger=False):
        """
        Kinematics-based controller.
        Args:
            robot_states : list of robot states from env
            pos_ds : desired positions (N,2)
            ori_ds : desired orientations (N,)
            trigger : external trigger for mode switching
        Returns:
            vxs, vys, omegas : lists of velocity commands
        """
        self.mode_switch_law(trigger)

        vxs, vys, omegas = [], [], []

        for idx, robot_state in enumerate(robot_states):
            pos = np.array(robot_state["position"][:2])
            R = np.array(robot_state["rotation_matrix"]).reshape(3, 3)
            yaw = np.arctan2(R[1,0], R[0,0])  # robot yaw angle

            pos_d = pos_ds[idx]
            ori_d = ori_ds[idx]

            # errors
            e_p = pos_d - pos
            e_th = np.arctan2(np.sin(ori_d - yaw), np.cos(ori_d - yaw))  # wrap [-pi,pi]


            if self.mode == Mode.NAV:
                # proportional control on position and orientation
                vx, vy = self.kp_t * e_p
                omega = self.kp_r * e_th
            else:
                vx, vy, omega = 0.0, 0.0, 0.0

            vxs.append(vx)
            vys.append(vy)
            omegas.append(omega)

        return vxs, vys, omegas 

    # -------------------------
    # Motion planner
    # -------------------------
    def motion_planner(self, obj_state, delta_indicator, ext_trajs=None):
        """
        Returns:
            pos_ds: (N, 2) desired positions for each robot
            ori_ds: (N,) desired orientations for each robot
        """
        # external trajectory (e.g., MAPF result)
        if ext_trajs is not None:
            self.ext_trajs = ext_trajs
            self.ext_traj_idx = 0
            self.delta_indicator = delta_indicator

        if self.ext_trajs is not None:
            traj_lens = [len(traj) for traj in self.ext_trajs["positions"].values()]
            traj_len = min(traj_lens)
            self.ext_traj_idx = min(self.ext_traj_idx, traj_len - 1)

            pos_ds = np.array([self.ext_trajs["positions"][i][self.ext_traj_idx] for i in range(self.num_robots)])
            ori_ds = np.array([self.ext_trajs["orientations"][i][self.ext_traj_idx] for i in range(self.num_robots)])

            self.ext_traj_idx += 1
            if self.ext_traj_idx >= (len(self.ext_trajs["positions"][0]) - 1):
                # Reset trajectory once done
                self.ext_trajs = None
                self.ext_traj_idx = 0
                # self.delta_indicator = delta_indicator

            return pos_ds, ori_ds
        else:
            pos_ds = []
            ori_ds = []

            obj_pos = np.array(obj_state["position"][:2]).reshape(2,)
            obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)
            obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))  # yaw

            # 2D rotation matrix from target orientation
            R = np.array([
                [np.cos(obj_ori), -np.sin(obj_ori)],
                [np.sin(obj_ori),  np.cos(obj_ori)]
            ])

            for idx in self.delta_indicator:
                # Relative position from object center before rotation
                if idx == 0:
                    local_pos = np.array([-self.L, -2*self.L]) + np.array([0, -self.r])
                    local_ori = np.pi / 2
                elif idx == 1:
                    local_pos = np.array([self.L, -2*self.L]) + np.array([0, -self.r])
                    local_ori = np.pi / 2
                elif idx == 2:
                    local_pos = np.array([2*self.L, -self.L]) + np.array([self.r, 0])
                    local_ori = np.pi
                elif idx == 3:
                    local_pos = np.array([2*self.L, self.L]) + np.array([self.r, 0])
                    local_ori = np.pi
                elif idx == 4:
                    local_pos = np.array([self.L, 2*self.L]) + np.array([0, self.r])
                    local_ori = 3*np.pi/2
                elif idx == 5:
                    local_pos = np.array([-self.L, 2*self.L]) + np.array([0, self.r])
                    local_ori = 3*np.pi/2
                elif idx == 6:
                    local_pos = np.array([-2*self.L, self.L]) + np.array([-self.r, 0])
                    local_ori = 0
                elif idx == 7:
                    local_pos = np.array([-2*self.L, -self.L]) + np.array([-self.r, 0])
                    local_ori = 0
                else:
                    raise ValueError(f"Invalid contact index: {idx}")

                # Rotate and translate local position based on target
                world_pos = obj_pos + R @ local_pos
                world_ori = (obj_ori + local_ori) % (2*np.pi)

                pos_ds.append(world_pos)
                ori_ds.append(world_ori)

            return np.array(pos_ds), np.array(ori_ds)
