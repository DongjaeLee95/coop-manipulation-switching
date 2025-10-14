import numpy as np
from enum import IntEnum

from envs.pushing_env_ode import PushingEnvODE
from ctrller.robotCtrller_kin import RobotCtrller
from visualizer.logvisualizer_ode import LogVisualizer
from logger.simulation_logger_ode import SimulationLogger
from ctrller.objCtrller import ObjCtrller
from ctrller.switching_law import SwitingLaw
from planner.robot_slot_planner import RobotSlotPlanner


class Mode(IntEnum):
    CON = 0
    NAV = 1


def main():
    # -------------------------------
    # Environment & Controller Setup
    # -------------------------------
    # dt = 1 / 240
    dt = 1 / 200
    env = PushingEnvODE(sim_config_path="configs/sim_config.yaml", dt=dt)
    ctrl = RobotCtrller(
        ctrl_config_path="configs/ctrl_config.yaml",
        sim_config_path="configs/sim_config.yaml"
    )
    obj_ctrl = ObjCtrller(
        obj_ctrl_config_path="configs/obj_ctrl_config.yaml",
        sim_config_path="configs/sim_config.yaml"
    )
    switching_law = SwitingLaw(
        obj_ctrl_config_path="configs/obj_ctrl_config.yaml",
        sim_config_path="configs/sim_config.yaml"
    )
    slot_planner = RobotSlotPlanner(sim_config_path="configs/sim_config.yaml")

    logger = SimulationLogger(prefix="simul_ode")
    logger.log_environment(env)

    # -------------------------------
    # Simulation Loop
    # -------------------------------
    # try:
    t = 0.0
    tf = 20.0

    delta_indicator = np.array(env.robot_init_id)
    delta = np.zeros(8)
    for i in delta_indicator:
        delta[i] = 1

    # desired target pose
    # obj_d = np.array([0, 0, np.deg2rad(90), 0.0, 0.0, 0.0])
    obj_d = np.array([1, 1, 1, 0.0, 0.0, 0.0])
    switch_trigger = False

    while True:
        if t > tf:
            break
        state = env.get_state()

        # TODO - MILP가 solution이 없다는데.. 말이안되는데? -- 뭔가 버그다, 90도 돌때 안되는데 matlab에서는 됨
        _, delta, switch_trigger, compt_time, rho = switching_law.compute(state["target"], obj_d, delta)
        u, _, V, _, _ = obj_ctrl.compute(state["target"], obj_d, delta)
        
        if switch_trigger:
            delta_indicator, ext_trajs = slot_planner.compute(state["robots"], state["target"], 
                                                              delta, delta_indicator)
        else:
            ext_trajs = None

        # robot motion planner + controller
        pos_ds, ori_ds = ctrl.motion_planner(state["target"], delta_indicator, ext_trajs)
        vx_s, vy_s, omegas = ctrl.compute_actions(state["robots"], pos_ds, ori_ds, switch_trigger)

        if ctrl.mode == Mode.NAV:
            u = np.zeros(len(delta))
        for rid in range(env.num_robots):
            if ctrl.mode == Mode.NAV:
                env.set_mode(rid, "NAV")
            if ctrl.mode == Mode.CON:
                env.set_mode(rid, "CON", delta_indicator[rid])

        env.apply_actions(vx_s, vy_s, omegas)
        # TODO - we need delta update
        env.step(u,delta)

        # -------------------------------
        # Logging
        # -------------------------------
        logger.log_step(
            t,
            state,
            {
                "vx": vx_s,
                "vy": vy_s,
                "omega": omegas,
                "ctrl_mode": int(ctrl.mode)
            },
            u,
            {
                "V_lyap": V,
                "delta": delta,
                "delta_indicator": delta_indicator,
                "trigger": switch_trigger,
                "MILP_compt_time": compt_time,
                "MILP_rho": rho
            },
            ext_trajs,
            obj_d
        )
        print("t: ", t)
        t += dt
        switch_trigger = False

    # except KeyboardInterrupt:
    print("Simulation terminated!")
    logger.save()

    # offline visualization (optional)
    visualizer = LogVisualizer(log_path=logger.log_path, sim_config="configs/sim_config.yaml")
    
    # visualizer.plot_animation(dt=dt, save_path="results/simulation.mp4")
    visualizer.plot_animation(dt=dt, save_path=None)
    
    visualizer.plot_target_box()
    visualizer.plot_u_with_mode()
    visualizer.plot_switching_data()
    visualizer.show()

    


if __name__ == "__main__":
    main()
