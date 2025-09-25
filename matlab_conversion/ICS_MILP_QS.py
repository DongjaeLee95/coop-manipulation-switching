import numpy as np
import time
from docplex.mp.model import Model

class ICSMILP:
    def __init__(self, param):
        self.param = param
        self.mdl = Model(name="ICS_MILP")
        self.u = self.mdl.continuous_var_list(param["udim"], lb=0, name="u")
        self.delta = self.mdl.binary_var_list(param["udim"], name="delta")
        self.z = self.mdl.continuous_var_list(param["udim"], lb=0, name="z")
        self.rho = self.mdl.continuous_var(lb=0, name="rho")

        self.mdl.maximize(self.rho)

        uM = param["uM"]
        for i in range(param["udim"]):
            self.mdl.add_constraint(self.z[i] <= self.u[i])
            self.mdl.add_constraint(self.u[i] <= uM * (1 - self.delta[i]) + self.z[i])
            self.mdl.add_constraint(self.z[i] <= uM * self.delta[i])

        # Placeholder for dynamic constraint
        self.dynamic_constraint = None

        # Sum(delta) = sum(delta_prev) → added dynamically in compute()
        self.delta_sum_constraint = None

    def compute(self, q, x_d, tau, delta_prev, u_prev, iter_idx):
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
        qdot_d = x_d[qdim:]

        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi),  np.cos(psi)]])
        F = np.block([[R, np.zeros((2, 1))],
                      [np.zeros((1, 2)), 1]])
        M = np.block([[self.param["c_f"] * np.eye(2), np.zeros((2, 1))],
                      [np.zeros((1, 2)), self.param["c_tau"]]])
        G = F @ np.linalg.inv(M)

        eq = q_d - q.reshape(-1,1)

        Bbar = self.findBbar(delta_prev)

        c = 2 * np.min(np.diag(self.param["Kq"]))
        V = float(0.5 * eq.T @ eq)

        trigger_x = Bbar.T @ G.T @ eq
        trigger_y = -(c - self.param["V_decay"]) * V + eq.T @ G @ tau.reshape(-1,1)
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

            # Remove previous dynamic constraints
            if self.dynamic_constraint is not None:
                self.mdl.remove(self.dynamic_constraint)
            if self.delta_sum_constraint is not None:
                self.mdl.remove(self.delta_sum_constraint)

            B_full = self.findB(np.ones(self.param["udim"]))
            lhs = eq.T @ np.linalg.inv(M) @ F @ tau.reshape(-1,1) - eq.T @ np.linalg.inv(M) @ B_full @ np.array(self.z)
            c = 2 * np.min(np.diag(self.param["Kq"]))
            V = float(0.5 * eq.T @ eq)

            self.dynamic_constraint = self.mdl.add_constraint((c - self.param["V_decay"]) * V >= lhs.item() + self.rho)
            self.delta_sum_constraint = self.mdl.add_constraint(self.mdl.sum(self.delta) == int(np.sum(delta_prev)))

            start = time.time()
            sol = self.mdl.solve(log_output=False)
            compt_time = time.time() - start

            if sol is None:
                print("[ICS_MILP] No solution found.")
                return np.zeros(self.param["udim"]), delta_prev, False, compt_time, np.nan

            u_val = np.array([sol[self.u[i]] for i in range(self.param["udim"])])
            delta_val = np.array([round(sol[self.delta[i]]) for i in range(self.param["udim"])])
            rho_val = sol[self.rho]

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
        M = np.diag([self.param["c_f"], self.param["c_f"], self.param["c_tau"]])
        return np.linalg.inv(M) @ self.findBbar(delta)

    @staticmethod
    def ReLU(x):
        return max(0, x)
