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
            self.kp_t = self.ctrl_config["gain"]["kp_t"]
            self.kd_t = self.ctrl_config["gain"]["kd_t"]
            self.kp_r = self.ctrl_config["gain"]["kp_r"]
            self.kd_r = self.ctrl_config["gain"]["kd_r"]
            self.J = self.ctrl_config["system"]["J"]

        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
            self.L = (self.sim_config["target"]["size"][0])/2
            self.r = self.sim_config["robot"]["radius"]
            self.m = self.sim_config["mass"]["robot"]

        with open("configs/obj_ctrl_config.yaml", "r") as f:
            self.obj_ctrl_config = yaml.safe_load(f)
            self.uM = self.obj_ctrl_config["uM"]
            
        # Initialize robot parameters
        self.num_robots = self.sim_config["robot"]["num"]
        self.delta_indicator = np.arange(self.num_robots)
        # [a,b,...]: 1st robot in poisition a, 2nd robot in position b, etc.

        # Initialize external trajectories
        self.ext_trajs = None
        self.ext_traj_idx = 0

        # Interaction mode, Naviation mode
        self.mode = Mode.NAV
        self.wait_duration = 2.0  # seconds
        self.wait_timer_start = None
    
    def mode_switch_law(self, trigger = False):
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

    def compute_actions(self, robot_states, robot_pushs, pos_ds, ori_ds, trigger = False):
        
        self.mode_switch_law(trigger)
        
        forces_x = []
        forces_y = []
        torques = []

        for idx, robot_state in enumerate(robot_states):
            pos = np.array(robot_state["position"][:2]).reshape(2, 1)
            vel = np.array(robot_state["linear_velocity"][:2]).reshape(2, 1)
            rot = np.array(robot_state["rotation_matrix"]).reshape(3, 3)
            ang_vel = np.array(robot_state["angular_velocity"][2])

            pos_d = pos_ds[idx]
            vel_d = np.zeros((2,1))
            ori_d = ori_ds[idx]
            rot_d = self.rot_z(ori_d)
            ang_vel_d = np.zeros((3,1))

            e_p = pos_d.reshape(2,1) - pos
            e_v = vel_d - vel
            e_R = 1/2*self.vee(rot.T@rot_d - rot_d.T@rot)
            e_om = rot.T@rot_d@ang_vel_d - ang_vel

            force = self.m*rot[:2, :2].T@(self.kp_t*e_p + self.kd_t*e_v)
            torque = self.J*(self.kp_r*e_R[2] + self.kd_r*e_om[2])

            if self.mode == Mode.CON:
                force[0] += min((robot_pushs[idx],self.uM)) # additional pushing force
            
            forces_x.append(force[0])
            forces_y.append(force[1])
            torques.append(torque)

        return forces_x, forces_y, torques 
    
    def motion_planner(self, obj_state, delta_indicator, ext_trajs=None):
        """
        Returns:
            pos_ds: (N, 2) desired positions for each robot
            ori_ds: (N,) desired orientations for each robot
        """
        # ext_trajs should change driectly to None after having some none-None value
        if ext_trajs is not None:
            self.ext_trajs = ext_trajs
            self.ext_traj_idx = 0
            self.delta_indicator = delta_indicator

        if self.ext_trajs is not None:
            traj_lens = [len(traj) for traj in self.ext_trajs["positions"].values()]
            traj_len = min(traj_lens)
            self.ext_traj_idx = min(self.ext_traj_idx, traj_len - 1)

            # Use current external trajectory
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

            obj_pos = np.array(obj_state["position"][:2]).reshape(2,)               # (2,)
            obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)          # (3,3)
            obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))     # yaw
            # obj_vel = np.array(obj_state["linear_velocity"][:2]).reshape(2,)      # (2,)
            # obj_ang_vel = float(obj_state["angular_velocity"][2])                 # scalar

            # 2D rotation matrix from target orientation
            R = np.array([
                [np.cos(obj_ori), -np.sin(obj_ori)],
                [np.sin(obj_ori),  np.cos(obj_ori)]
            ])

            for idx in self.delta_indicator:
                # Relative position from object center before rotation
                if idx == 0:
                    local_pos = np.array([-self.L, -2 * self.L]) + np.array([0, -self.r])
                    local_ori = np.pi / 2
                elif idx == 1:
                    local_pos = np.array([self.L, -2 * self.L]) + np.array([0, -self.r])
                    local_ori = np.pi / 2
                elif idx == 2:
                    local_pos = np.array([2 * self.L, -self.L]) + np.array([self.r, 0])
                    local_ori = np.pi
                elif idx == 3:
                    local_pos = np.array([2 * self.L, self.L]) + np.array([self.r, 0])
                    local_ori = np.pi
                elif idx == 4:
                    local_pos = np.array([self.L, 2 * self.L]) + np.array([0, self.r])
                    local_ori = 3 * np.pi / 2
                elif idx == 5:
                    local_pos = np.array([-self.L, 2 * self.L]) + np.array([0, self.r])
                    local_ori = 3 * np.pi / 2
                elif idx == 6:
                    local_pos = np.array([-2 * self.L, self.L]) + np.array([-self.r, 0])
                    local_ori = 0
                elif idx == 7:
                    local_pos = np.array([-2 * self.L, -self.L]) + np.array([-self.r, 0])
                    local_ori = 0
                else:
                    raise ValueError(f"Invalid contact index: {idx}")

                # Rotate and translate local position based on target
                world_pos = obj_pos + R @ local_pos
                world_ori = (obj_ori + local_ori) % (2 * np.pi)

                pos_ds.append(world_pos)
                ori_ds.append(world_ori)

            return np.array(pos_ds), np.array(ori_ds)
    
    def rot_z(self,psi):
        return np.array([[np.cos(psi), -np.sin(psi), 0], 
                         [np.sin(psi), np.cos(psi), 0], 
                         [0, 0, 1]])
    
    def vee(self,mat):
        mat = (mat - mat.T)/2
        return np.array([mat[2][1], mat[0][2], mat[1][0]])