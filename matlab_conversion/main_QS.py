import numpy as np
import matplotlib.pyplot as plt
import math

from high_level_control_QS import HighLevelControl
from ICS_MILP_QS import ICSMILP
from ctrller_QP_only_QS import CtrllerQPOnly
from dynamics_QS import Dynamics

def main_QS():
    # === Parameters ===
    param = {
        "DT": 0.001,
        "m": 1,
        "mu": 0.5,
        "g": 9.81,
        "c": 1,
        "qdim": 3,
        "vdim": 3,
        "udim": 8,
        "bt": 0,
        "br": 0,
        "uM": 5,
        "V_decay": 4, # 0.01
        "L": 0.1,
        "L1": 0.1, "L2": 0.1, "L3": 0.1, "L4": 0.1,
        "L5": 0.1, "L6": 0.1, "L7": 0.1, "L8": 0.1,
        "Kq": 3 * np.diag([1, 1, 1]),
    }

    param["r"] = math.sqrt(5)*param["L"]
    param["c_f"] = param["mu"]*param["m"]*param["g"]
    param["c_tau"] = param["c"]*param["r"]*param["mu"]*param["m"]*param["g"]

    udim = param["udim"]
    qdim = param["qdim"]
    vdim = param["vdim"]

    # === Simulation setup ===
    ts = np.arange(0, 5 + param["DT"], param["DT"])
    print("len(ts):",len(ts))
    q = np.zeros((qdim, len(ts)))

    # q_d = 0.1 * np.diag([1, 1, -1]) @ np.ones((qdim, len(ts)))
    q_d = np.diag([0, 0, np.deg2rad(30)]) @ np.ones((qdim, len(ts)))
    qdot_d = np.zeros((qdim, len(ts)))

    delta_init = np.zeros((udim,))
    delta_init[0] = 1
    delta_init[1] = 1
    # delta_init[2] = 1
    # delta_init[3] = 1
    # delta_init[4] = 1
    # delta_init[5] = 1
    # delta_init[6] = 1
    # delta_init[7] = 1

    tau = np.zeros((3, len(ts)))
    delta = np.zeros((udim, len(ts)))
    u = np.zeros((udim, len(ts)))
    y = np.zeros(len(ts))
    x_max = np.zeros(len(ts))
    u_ics = np.zeros((udim, len(ts)))
    trigger_flag = np.zeros(len(ts))
    compt_time = np.zeros(len(ts))
    rho = np.zeros(len(ts))
    V_lyap = np.zeros(len(ts))
    V_lyap_compare = np.zeros(len(ts))

    delta[:,0] = delta_init
    for i in range(len(ts)):
        delta[:, i] = delta_init

    # === Instantiate controllers/dynamics ===
    hlc = HighLevelControl(param)
    ics = ICSMILP(param)
    ctrl_qp = CtrllerQPOnly(param)
    dyn = Dynamics(param)

    # === Simulation loop ===
    for i in range(len(ts)):
        if i > 0:
            delta_prev = delta[:, i-1]
            u_prev = u[:, i-1]
        else:
            delta_prev = delta[:, 0]
            u_prev = u[:, 0]

        iter_idx = i
        tau[:, i], V_lyap[i] = hlc.compute(
            q[:, i],
            np.hstack([q_d[:, i], qdot_d[:, i]]).reshape(-1, 1)
        )
        if i == 0:
            V_lyap_compare[i] = V_lyap[i]
        else:
            V_lyap_compare[i] = np.exp(-param["V_decay"]*ts[i]) * V_lyap_compare[0]

        u_ics[:, i], delta[:, i], trigger_flag[i], compt_time[i], rho[i] = \
            ics.compute(
                q[:, i],
                np.hstack([q_d[:, i], qdot_d[:, i]]).reshape(-1, 1),
                tau[:, i], delta_prev, u_prev, iter_idx
            )

        u[:, i], y[i], x_max[i] = ctrl_qp.compute(
            q[:, i],
            np.hstack([q_d[:, i], qdot_d[:, i]]).reshape(-1, 1),
            tau[:, i], delta[:, i], iter_idx
        )

        x_next = dyn.step(
            q[:, i],
            u[:, i], delta[:, i], ts[i]
        )

        if i < len(ts)-1:
            print(f"step: {i}")
            q[:, i+1] = x_next[:qdim]

    # === Plotting ===
    linewidth = 2

    # --- Fig 1: states ---
    fig1, axs = plt.subplots(3, 1, figsize=(6, 6))
    axs[0].plot(ts, q[0, :], 'k-', linewidth=linewidth)
    axs[0].plot(ts, q[1, :], 'b-', linewidth=linewidth)
    axs[0].plot(ts, q[2, :], 'r-', linewidth=linewidth)
    axs[0].plot(ts, q_d[0, :], 'k--', linewidth=linewidth)
    axs[0].plot(ts, q_d[1, :], 'b--', linewidth=linewidth)
    axs[0].plot(ts, q_d[2, :], 'r--', linewidth=linewidth)
    axs[0].legend(['p_x','p_y','theta'])
    axs[0].set_ylabel("q")

    axs[1].plot(ts, tau[0,:], 'k-', linewidth=linewidth)
    axs[1].plot(ts, tau[1,:], 'b-', linewidth=linewidth)
    axs[1].plot(ts, tau[2,:], 'r-', linewidth=linewidth)
    axs[1].set_ylabel("tau")
    
    axs[2].plot(ts, V_lyap, 'k-', linewidth=linewidth)
    axs[2].plot(ts, V_lyap_compare, 'r--', linewidth=linewidth)
    axs[2].set_ylabel("V")
    axs[2].set_xlabel("time [s]")

    plt.tight_layout()

    # --- Fig 2: inputs ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 6))
    axs2[0].plot(ts, u.T, '.', markersize=2)  # plot all inputs
    axs2[0].set_ylabel("u")

    axs2[1].plot(ts, delta.T, linewidth=linewidth)
    axs2[1].set_ylabel("delta")

    axs2[2].plot(ts, trigger_flag, 'k-', linewidth=2)
    axs2[2].set_ylabel("triggered")
    axs2[2].set_xlabel("time [s]")

    plt.tight_layout()

    # --- Fig 3: computation time and rho ---
    fig3, axs3 = plt.subplots(2, 1, figsize=(6, 4))
    axs3[0].plot(ts, compt_time, '.', markersize=2)
    axs3[0].set_ylabel("compt time [s]")

    axs3[1].plot(ts, rho, '.', markersize=2)
    axs3[1].plot(ts, np.zeros_like(ts), 'r-', linewidth=linewidth)
    axs3[1].set_ylabel("rho")
    axs3[1].set_xlabel("time [s]")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main_QS()
