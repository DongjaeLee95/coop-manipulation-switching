import numpy as np

class HighLevelControl:
    def __init__(self, param):
        """
        Initialize high level controller with system parameters.
        Args:
            param : dict, system parameters
        """
        self.param = param

    def compute(self, q, x_d):
        """
        Python version of high_level_control.m

        Args:
            q : (3,) np.array, current state (position + orientation)
            x_d : (6,) np.array, desired trajectory [q_d; qdot_d]

        Returns:
            tau : (3,) np.array, control input
            V_lyap : float, Lyapunov function value
        """
        psi = q[2]

        qdim = self.param["qdim"]
        q_d = x_d[0:qdim]
        qdot_d = x_d[qdim:]

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

        eq = q_d - q.reshape(-1,1)

        tau = M @ (F.T @ (qdot_d + self.param["Kq"] @ eq))
        V_lyap = float(0.5 * eq.T @ eq)

        return tau.flatten(), V_lyap
