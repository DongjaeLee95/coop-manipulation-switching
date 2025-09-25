import numpy as np
import time
from docplex.mp.model import Model

class ICSMILP:
    def __init__(self, param):
        self.param = param

    def compute(self, q, v, x_d, tau, delta_prev, u_prev, iter_idx):
        """
        Python version of ICS_MILP.m

        Returns:
            u : (udim,) np.array
            delta : (udim,) np.array
            trigger_flag : bool
            compt_time : float
            rho : float
        """
        psi = q[2]

        qdim = self.param["qdim"]
        q_d = x_d[0:qdim]
        qdot_d = x_d[qdim:2*qdim]

        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
        F = np.block([[R, np.zeros((2, 1))],
                      [np.zeros((1, 2)), 1]])
        M = np.block([[self.param["m"] * np.eye(2), np.zeros((2, 1))],
                      [np.zeros((1, 2)), self.param["J"]]])

        eq = q_d - q.reshape(-1,1)
        v_d = qdot_d + self.param["Kq"] @ eq
        ev = v_d - v.reshape(-1,1)

        Bbar = self.findBbar(delta_prev)
        Gbar = self.findGbar(delta_prev)
        ubar = self.findubar(u_prev, delta_prev)

        c = 2 * min(np.min(np.diag(self.param["Kq"])),
                    np.min(np.diag(self.param["Kv"]))) - 1
        V = float(0.5 * (eq.T @ eq + ev.T @ ev))

        trigger_x = Bbar.T @ np.linalg.inv(M) @ ev
        trigger_y = -(c - self.param["V_decay"]) * V + ev.T @ np.linalg.inv(M) @ F @ tau
        trigger_cond = (trigger_y > 0) and (np.max(trigger_x) <= 0)

        # additonal condition
        ubar_new = np.zeros(np.sum(delta_prev, dtype=int))
        if not trigger_cond and trigger_y > 0:
            denominator = np.sum(np.maximum(0, trigger_x)**2)
            if denominator > 0:
                for i in range(len(ubar_new)):
                    ubar_new[i] = self.ReLU(trigger_x[i]) * trigger_y / denominator

        if np.max(ubar_new) > self.param["uM"]:
            trigger_cond = True

        # --- Optimization step ---
        if trigger_cond:
            print("MILP triggered")
            mdl = Model(name="ICS_MILP")
            u = mdl.continuous_var_list(self.param["udim"], lb=0, name="u")
            delta = mdl.binary_var_list(self.param["udim"], name="delta")
            z = mdl.continuous_var_list(self.param["udim"], lb=0, name="z")
            rho = mdl.continuous_var(lb=0, name="rho")

            # Objective: minimize -rho  <=> maximize rho
            mdl.maximize(rho)

            uM = self.param["uM"]

            # Constraint: cV >= e_v^T F (tau - inv(M) * sum(b_i*z_i)) + rho
            B_full = self.findB(np.ones(self.param["udim"]))
            lhs = ev.T @ np.linalg.inv(M) @ F @ tau - ev.T @ np.linalg.inv(M) @ B_full @ np.array(z)
            mdl.add_constraint(c * V >= lhs.item() + rho)

            # Constraints relating u, delta, z
            for i in range(self.param["udim"]):
                mdl.add_constraint(z[i] <= u[i])
                mdl.add_constraint(u[i] <= uM * (1 - delta[i]) + z[i])
                mdl.add_constraint(z[i] <= uM * delta[i])

            # sum(delta) = sum(delta_prev)
            mdl.add_constraint(mdl.sum(delta) == int(np.sum(delta_prev)))

            # Solve
            start = time.time()
            sol = mdl.solve(log_output=False)
            compt_time = time.time() - start

            if sol is None:
                print("[ICS_MILP] No solution found.")
                return np.zeros(self.param["udim"]), delta_prev, False, compt_time, np.nan

            u_val = np.array([sol[u[i]] for i in range(self.param["udim"])])
            delta_val = np.array([round(sol[delta[i]]) for i in range(self.param["udim"])])
            rho_val = sol[rho]

            return u_val, delta_val, True, compt_time, rho_val
        else:
            # Not triggered → keep previous delta, reset u
            return np.zeros(self.param["udim"]), delta_prev, False, np.nan, np.nan

    # === Helper functions ===
    def findB(self, delta):
        p = self.param
        B1d = delta[0]*np.array([0,1,-p["L1"]])
        B2d = delta[1]*np.array([0,1, p["L2"]])
        B3d = delta[2]*np.array([-1,0,-p["L3"]])
        B4d = delta[3]*np.array([-1,0, p["L4"]])
        B5d = delta[4]*np.array([0,-1,-p["L5"]])
        B6d = delta[5]*np.array([0,-1, p["L6"]])
        B7d = delta[6]*np.array([1,0,-p["L7"]])
        B8d = delta[7]*np.array([1,0, p["L8"]])
        return np.column_stack([B1d, B2d, B3d, B4d, B5d, B6d, B7d, B8d])

    def findubar(self, u, delta):
        j_idx = [j for j in range(len(delta)) if delta[j] > 0]
        return u[j_idx]

    def findBbar(self, delta):
        B = self.findB(delta)
        j_idx = [j for j in range(len(delta)) if delta[j] > 0]
        return B[:, j_idx]

    def findGbar(self, delta):
        M = np.diag([self.param["m"], self.param["m"], self.param["J"]])
        return np.linalg.inv(M) @ self.findBbar(delta)

    @staticmethod
    def ReLU(x):
        return max(0, x)
