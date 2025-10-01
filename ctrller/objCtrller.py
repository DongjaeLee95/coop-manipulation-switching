# based on simplified controller + QP-based controller
import yaml
import numpy as np
import math

class ObjCtrller:
    def __init__(self, integral_flag = False, dt = 1/240, obj_ctrl_config_path="configs/obj_ctrl_config.yaml", sim_config_path="configs/sim_config.yaml"):
        """
        Unified controller that combines high-level control and QP-based actuator allocation.
        Args:
            param : dict, system parameters
        """
        self.param = {}
        with open(obj_ctrl_config_path, "r") as f:
            self.obj_ctrl_config = yaml.safe_load(f)
            self.param["qdim"] = self.obj_ctrl_config["qdim"]
            self.param["udim"] = self.obj_ctrl_config["udim"]
            self.param["Kq"] = np.array(self.obj_ctrl_config["Kq"])
            self.param["Ki"] = np.array(self.obj_ctrl_config["Ki"])
            self.param["V_decay"] = self.obj_ctrl_config["V_decay"]
            self.param["c"] = self.obj_ctrl_config["c"]

        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
            self.param["L"] = (self.sim_config["target"]["size"][0])/2
            self.param["m"] = self.sim_config["mass"]["target"]
            # self.param["mu"] = math.sqrt(self.sim_config["friction"]["plane"] * 
            #                             self.sim_config["friction"]["target"])
            self.param["mu"] = min((self.sim_config["friction"]["plane"],
                                        self.sim_config["friction"]["target"]))
            self.param["g"] = self.sim_config["gravity"]

            self.param["r"] = math.sqrt(5)*self.param["L"]
            self.param["c_f"] = self.param["mu"]*self.param["m"]*self.param["g"]
            self.param["c_tau"] = self.param["c"]*self.param["r"]*self.param["c_f"]
        
        # TODO - integral term should be reinitialized whenever goes to NAV mode
        self.dt = dt
        self.integral_error = np.zeros((self.param["qdim"],1))
        self.integral_flag = integral_flag

    def compute(self, obj_state, x_d, delta):
        """
        Main compute function for the controller.

        Args:
            q : (3,) np.array, current state
            x_d : (6,) np.array, desired trajectory [q_d; qdot_d]
            delta : (udim,) np.array, actuator selection vector
            iter_idx : int, time step index

        Returns:
            u : (udim,) np.array, actuator input
            tau : (3,) np.array, virtual control torque
            V : float, Lyapunov value
            y : float, trigger_y
            x_max : float, max of trigger_x
        """
        # --- High-level control ---
        obj_pos = np.array(obj_state["position"][:2]).reshape(2,)   
        obj_rot = np.array(obj_state["rotation_matrix"]).reshape(3, 3)
        obj_ori = float(np.arctan2(obj_rot[1, 0], obj_rot[0, 0]))
        
        q = np.vstack([obj_pos.reshape(2, 1), [obj_ori]]) 

        psi = float(q[2])
        qdim = self.param["qdim"]
        q_d = x_d[0:qdim].reshape(-1,1)
        qdot_d = x_d[qdim:].reshape(-1,1)

        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
        F = np.block([[R, np.zeros((2, 1))],
                      [np.zeros((1, 2)), 1]])
        M = np.block([[self.param["c_f"] * np.eye(2), np.zeros((2, 1))],
                      [np.zeros((1, 2)), self.param["c_tau"]]])
        G = F @ np.linalg.inv(M)

        eq = q_d - q.reshape(-1, 1)
        if self.integral_flag:
            tau = M @ (F.T @ (qdot_d + self.param["Kq"] @ eq + self.param["Ki"] @ self.integral_error))
            self.integral_error += eq*self.dt
        else:
            tau = M @ (F.T @ (qdot_d + self.param["Kq"] @ eq))
        tau = tau.flatten()

        # --- QP controller (analytic solution) ---
        delta_dim = len(delta)
        Bbar = self._findBbar(delta)
        c = 2 * np.min(np.diag(self.param["Kq"]))
        V = float(0.5 * eq.T @ eq)

        trigger_x = Bbar.T @ G.T @ eq
        trigger_y = -(c - self.param["V_decay"]) * V + eq.T @ G @ tau.reshape(-1, 1)

        ubar = np.zeros(np.sum(delta, dtype=int))
        if trigger_y > 0:
            denominator = np.sum(np.maximum(0, trigger_x)**2)
            if denominator > 0:
                for i in range(len(ubar)):
                    ubar[i] = self._ReLU(trigger_x[i]) * trigger_y / denominator

        u = np.zeros(delta_dim)
        j_idx = [j for j in range(delta_dim) if delta[j] > 0]
        u[j_idx] = ubar

        y = float(trigger_y)
        x_max = float(np.max(trigger_x)) if len(trigger_x) > 0 else 0.0

        return u, tau, V, y, x_max

    def _findBbar(self, delta):
        """
        Builds the B matrix based on actuator selection (delta).

        Args:
            delta : (udim,) actuator enable vector

        Returns:
            Bbar : (3, num_active) matrix of active actuator directions
        """
        p = self.param
        B1d = delta[0] * np.array([0, 1, -p["L"]])
        B2d = delta[1] * np.array([0, 1,  p["L"]])
        B3d = delta[2] * np.array([-1, 0, -p["L"]])
        B4d = delta[3] * np.array([-1, 0,  p["L"]])
        B5d = delta[4] * np.array([0, -1, -p["L"]])
        B6d = delta[5] * np.array([0, -1,  p["L"]])
        B7d = delta[6] * np.array([1, 0, -p["L"]])
        B8d = delta[7] * np.array([1, 0,  p["L"]])

        if p["udim"] == 8:
            B = np.column_stack([B1d, B2d, B3d, B4d, B5d, B6d, B7d, B8d])
        elif p["udim"] == 12:
            B9d  = delta[8]  * np.array([0, 1, 0])
            B10d = delta[9]  * np.array([-1, 0, 0])
            B11d = delta[10] * np.array([0, -1, 0])
            B12d = delta[11] * np.array([1, 0, 0])
            B = np.column_stack([B1d, B2d, B3d, B4d, B5d, B6d, B7d, B8d,
                                 B9d, B10d, B11d, B12d])
        else:
            raise ValueError("Unsupported udim: {}".format(p["udim"]))

        j_idx = [j for j in range(p["udim"]) if delta[j] > 0]
        return B[:, j_idx]

    @staticmethod
    def _ReLU(x):
        return max(0, x)
