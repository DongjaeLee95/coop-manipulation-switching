import numpy as np

class HighLevelControl:
    def __init__(self, param):
        """
        Initialize high level controller with system parameters.
        Args:
            param : dict, system parameters
        """
        self.param = param

    def compute(self, q, v, x_d):
        """
        Python version of high_level_control.m

        Args:
            q : (3,) np.array, current state (position + orientation)
            v : (3,) np.array, current velocity
            x_d : (9,) np.array, desired trajectory [q_d; qdot_d; qddot_d]

        Returns:
            tau : (3,) np.array, control input
            V_lyap : float, Lyapunov function value
        """
        psi = q[2]
        v_x, v_y, om = v

        qdim = self.param["qdim"]
        q_d = x_d[0:qdim]
        qdot_d = x_d[qdim:2*qdim]
        qddot_d = x_d[2*qdim:]

        R = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)]
        ])
        F = np.block([
            [R, np.zeros((2, 1))],
            [np.zeros((1, 2)), 1]
        ])
        M = np.block([
            [self.param["m"] * np.eye(2), np.zeros((2, 1))],
            [np.zeros((1, 2)), self.param["J"]]
        ])

        eq = q_d - q.reshape(-1,1)
        qdot = v.reshape(-1,1)

        v_d = qdot_d + self.param["Kq"] @ eq
        ev = v_d - v.reshape(-1,1)
        vdot_d = qddot_d + self.param["Kq"] @ (qdot_d - qdot)

        tau = F.T @ (M @ (vdot_d + self.param["Kv"] @ ev))
        V_lyap = float(0.5 * (eq.T @ eq + ev.T @ ev))

        return tau, V_lyap
