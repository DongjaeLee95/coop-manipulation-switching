import numpy as np
from scipy.integrate import solve_ivp

class Dynamics:
    def __init__(self, param):
        self.param = param

    def step(self, x, u, delta, time):
        """
        One integration step for dynamics using scipy.solve_ivp (ode45 equivalent).
        
        Args:
            x : (6,) np.array, state [q; v]
            u : (udim,) np.array, input
            delta : (udim,) np.array, actuator selection
            time : float, current time

        Returns:
            x_next : (6,) np.array, state at time + DT
        """
        dt = self.param["DT"]
        t_span = [time, time + dt]

        sol = solve_ivp(
            fun=lambda t, y: self.EOM(t, y, u, delta),
            t_span=t_span,
            y0=x,
            method="RK45",   # similar to MATLAB ode45
            rtol=1e-6,
            atol=1e-9
        )

        # return the last solution
        x_next = sol.y[:, -1]
        return x_next

    def EOM(self, _, Y, u, delta):
        """
        Equation of Motion (EOM).
        Args:
            Y : (6,) np.array, [q; v]
            u : (udim,) np.array
            delta : (udim,) np.array
        Returns:
            dydt : (6,) np.array
        """
        q = Y[0:3]
        v = Y[3:6]

        psi = q[2]
        v_x, v_y, om = v

        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
        F = np.block([
            [R, np.zeros((2, 1))],
            [np.zeros((1, 2)), 1]
        ])
        M = np.block([
            [self.param["m"] * np.eye(2), np.zeros((2, 1))],
            [np.zeros((1, 2)), self.param["J"]]
        ])

        # Build G matrix depending on udim
        B1d = delta[0] * np.array([0, 1, -self.param["L1"]])
        B2d = delta[1] * np.array([0, 1,  self.param["L2"]])
        B3d = delta[2] * np.array([-1, 0, -self.param["L3"]])
        B4d = delta[3] * np.array([-1, 0,  self.param["L4"]])
        B5d = delta[4] * np.array([0, -1, -self.param["L5"]])
        B6d = delta[5] * np.array([0, -1,  self.param["L6"]])
        B7d = delta[6] * np.array([1, 0, -self.param["L7"]])
        B8d = delta[7] * np.array([1, 0,  self.param["L8"]])

        if self.param["udim"] == 8:
            B = np.column_stack([B1d, B2d, B3d, B4d, B5d, B6d, B7d, B8d])
        elif self.param["udim"] == 12:
            B9d  = delta[8]  * np.array([0, 1, 0])
            B10d = delta[9]  * np.array([-1, 0, 0])
            B11d = delta[10] * np.array([0, -1, 0])
            B12d = delta[11] * np.array([1, 0, 0])
            B = np.column_stack([B1d, B2d, B3d, B4d, B5d, B6d, B7d, B8d, B9d, B10d, B11d, B12d])

        qdot = v
        G = np.linalg.solve(M, B)  # M \ B in MATLAB
        vdot = F @ (G @ u)

        dydt = np.zeros(2 * self.param["qdim"])
        dydt[0:self.param["qdim"]] = qdot
        dydt[self.param["qdim"]:] = vdot

        return dydt        
