import yaml
import numpy as np

class RobotCtrller:
    def __init__(self, config_path="configs/ctrl_config.yaml"):

        # Load YAML configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.kp_t = self.config["gain"]["kp_t"]
            self.kd_t = self.config["gain"]["kd_t"]
            self.kp_r = self.config["gain"]["kp_r"]
            self.kd_r = self.config["gain"]["kd_r"]
            self.L = self.config["system"]["L"]
            self.m = self.config["system"]["m"]
            self.J = self.config["system"]["J"]

    def compute_actions(self, state):
        # target_pos = state["target"]["position"]
        # target_vel = state["target"]["linear_velocity"]
        

        pos_ds = np.array([[-self.L, -3*self.L], [self.L, -3*self.L], [3*self.L, -self.L], [3*self.L, self.L],
                 [self.L, 3*self.L], [-self.L, 3*self.L], [-3*self.L, self.L], [-3*self.L, -self.L]])
        
        ori_ds = np.array([np.pi/2, np.pi/2, np.pi, np.pi, 3*np.pi/2, 3*np.pi/2, 0, 0])
        
        forces_x = []
        forces_y = []
        torques = []

        for idx, robot_state in enumerate(state["robots"]):
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
            
            forces_x.append(force[0])
            forces_y.append(force[1])
            torques.append(torque)

        return forces_x, forces_y, torques
    
    def rot_z(self,psi):
        return np.array([[np.cos(psi), -np.sin(psi), 0], 
                         [np.sin(psi), np.cos(psi), 0], 
                         [0, 0, 1]])
    
    def vee(self,mat):
        mat = (mat - mat.T)/2
        return np.array([mat[2][1], mat[0][2], mat[1][0]])