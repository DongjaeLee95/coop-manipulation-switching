import yaml
import numpy as np
from scipy.linalg import block_diag
import math

class RobotCtrller_QS:
    def __init__(self, ctrl_config_path="configs/ctrl_config.yaml", sim_config_path="configs/sim_config.yaml"):

        # Load YAML configuration
        with open(ctrl_config_path, "r") as f:
            self.ctrl_config = yaml.safe_load(f)
            self.kp_t = self.ctrl_config["gain"]["kp_t"]
            self.kd_t = self.ctrl_config["gain"]["kd_t"]
            self.kp_r = self.ctrl_config["gain"]["kp_r"]
            self.kd_r = self.ctrl_config["gain"]["kd_r"]
            self.L = self.ctrl_config["system"]["L"]
            self.J = self.ctrl_config["system"]["J"]

        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
            self.m = self.sim_config["robot"]["mass"]
            self.mu_plane = self.sim_config["friction"]["plane"]
            self.mu_robot = self.sim_config["friction"]["robot"]

    def compute_action(self, state, ref):

        # pos_ds = np.array([[0.1,0.1]])
        # ori_ds = np.array([np.pi/2])
        # pos_d = pos_ds[0]
        # vel_d = np.zeros((2,1))
        # ori_d = ori_ds[0]
        # rot_d = self.rot_z(ori_d)
        # ang_vel_d = np.zeros((3,1))

        pos = np.array(state["position"][:2]).reshape(2, 1)
        vel = np.array(state["linear_velocity"][:2]).reshape(2, 1)
        rot = np.array(state["rotation_matrix"]).reshape(3, 3)
        ang_vel = np.array(state["angular_velocity"][2])

        pos_d = ref["position"]
        vel_d = ref["velocity"]
        ori_d = ref["yaw"]
        rot_d = self.rot_z(ori_d)
        ang_vel_d = ref["angular_velocity"]

        e_p = pos_d.reshape(2,1) - pos
        e_v = vel_d - vel
        e_R = 1/2*self.vee(rot.T@rot_d - rot_d.T@rot)
        e_om = rot.T@rot_d@ang_vel_d - ang_vel

        g = 9.8
        mu = math.sqrt(self.mu_plane*self.mu_robot)

        c = 1.0
        r = math.sqrt(5)*self.L

        cf = mu*self.m*g
        ctau = c*r*cf
        G = block_diag(rot[:2,:2], 1)@np.diag([1/cf,1/cf,1/ctau])

        # input = np.linalg.inv(G)@np.vstack((self.kp_t * e_p, [[self.kp_r * e_R[2]]]))
        input = np.linalg.inv(G)@np.vstack((self.kp_t*e_p + self.kd_t*e_v, [self.kp_r*e_R[2] + self.kd_r*e_om[2]]))

        force = input[:2]
        torque = input[-1]

        # force = self.m*rot[:2, :2].T@(self.kp_t*e_p)
        # torque = self.J*(self.kp_r*e_R[2])
        
        # forces_x.append(force[0])
        # forces_y.append(force[1])
        # torques.append(torque)

        return float(force[0]), float(force[1]), float(torque)
    
    def rot_z(self,psi):
        return np.array([[np.cos(psi), -np.sin(psi), 0], 
                         [np.sin(psi), np.cos(psi), 0], 
                         [0, 0, 1]])
    
    def vee(self,mat):
        mat = (mat - mat.T)/2
        return np.array([mat[2][1], mat[0][2], mat[1][0]])