import numpy as np

class CtrllerQPOnly:
    def __init__(self, param):
        self.param = param

    def compute(self, q, x_d, tau, delta, iter_idx):
        """
        Args:
            q : (3,) np.array, current state
            x_d : (qdim,) np.array, desired [q_d]
            tau : (3,) np.array, torque input
            delta : (udim,) np.array, actuator selection vector
            iter_idx : int, iteration index

        Returns:
            u : (udim,) np.array, actuator input
            y : float, trigger_y value
            x_max : float, max(trigger_x)
        """
        psi = q[2]

        qdim = self.param["qdim"]
        q_d = x_d[0:qdim]

        R = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)]
        ])
        F = np.block([
            [R, np.zeros((2, 1))],
            [np.zeros((1, 2)), 1]
        ])
        M = np.block([
            [self.param["c_f"] * np.eye(2), np.zeros((2, 1))],
            [np.zeros((1, 2)), self.param["c_tau"]]
        ])
        G = F @ np.linalg.inv(M)

        eq = q_d - q.reshape(-1,1)

        delta_dim = len(delta)
        Bbar = self.findBbar(delta)
        c = 2*np.min(np.diag(self.param["Kq"]))
        V = float(0.5 * eq.T @ eq)

        # --- Analytic controller ---
        trigger_x = Bbar.T @ G.T @ eq
        trigger_y = -(c - self.param["V_decay"]) * V + eq.T @ G @ tau.reshape(-1,1)

        ubar = np.zeros(int(np.sum(delta)))
        if trigger_y > 0:
            denominator = np.sum(np.maximum(0, trigger_x)**2)
            if denominator > 0:
                for i in range(len(ubar)):
                    ubar[i] = self.ReLU(trigger_x[i]) * trigger_y / denominator
                    # ubar[i] = min(ubar[i], self.param["uM"])    

        # select indices from delta
        u = np.zeros(delta_dim)
        j_idx = [j for j in range(delta_dim) if delta[j] > 0]
        u[j_idx] = ubar

        y = trigger_y
        x_max = np.max(trigger_x) if len(trigger_x) > 0 else 0.0

        return u, float(y), float(x_max)

    def findBbar(self, delta):
        p = self.param
        B1d = delta[0] * np.array([0, 1, -p["L1"]])
        B2d = delta[1] * np.array([0, 1,  p["L2"]])
        B3d = delta[2] * np.array([-1, 0, -p["L3"]])
        B4d = delta[3] * np.array([-1, 0,  p["L4"]])
        B5d = delta[4] * np.array([0, -1, -p["L5"]])
        B6d = delta[5] * np.array([0, -1,  p["L6"]])
        B7d = delta[6] * np.array([1, 0, -p["L7"]])
        B8d = delta[7] * np.array([1, 0,  p["L8"]])

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
            raise ValueError("Unsupported udim")

        j_idx = [j for j in range(p["udim"]) if delta[j] > 0]
        return B[:, j_idx]

    @staticmethod
    def ReLU(x):
        return max(0, x)
