from envs.pushing_env import PushingEnv
from ctrller.robotCtrller import RobotCtrller
from visualizer.logvisualizer import LogVisualizer
from logger.simulation_logger import SimulationLogger
from ctrller.objCtrller import ObjCtrller
from enum import IntEnum
import numpy as np

class Mode(IntEnum):
    CON = 0
    NAV = 1

def deg2rad(ang):
    return ang*np.pi/180

def main():
    env = PushingEnv(gui=True, sim_config_path="configs/sim_config.yaml")
    ctrl = RobotCtrller(ctrl_config_path="configs/ctrl_config.yaml", sim_config_path="configs/sim_config.yaml")
    obj_ctrl = ObjCtrller(obj_ctrl_config_path="configs/obj_ctrl_config.yaml", sim_config_path="configs/sim_config.yaml")

    logger = SimulationLogger()

    logger.log_environment(env)

    try:
        # print simulation setting
        # env.print_dynamics_info(env.target_box)
        # env.print_dynamics_info(env.robots[0])

        t = 0.0
        dt = 1 / 240
 
        # initialize delta's
        delta_indicator = np.array(env.robot_init_id)
        delta = np.zeros(8)
        for i in delta_indicator:
            delta[i] = 1

        # set object desired pos, orientation
        obj_d = np.array([1.0, 1.0, deg2rad(90), 0.0, 0.0, 0.0])

        while True:
            state = env.get_state()
            # TODO - put multi-agent path finder module
            ext_trajs = None
            # MILP-based switching law
                # output: delta_indicator, delta

            # object controller - if successfully moved to the designated contact point
            u, tau, V, _, _ = obj_ctrl.compute(state["target"], obj_d, delta)

            trigger = False
            pos_ds, ori_ds = ctrl.motion_planner(state["target"], ext_trajs)
            forces_x, forces_y, torques = ctrl.compute_actions(state["robots"], u, pos_ds, ori_ds, trigger)
            if ctrl.mode == Mode.NAV:
                u = np.zeros(env.num_robots)

            env.apply_actions(forces_x, forces_y, torques)
            env.step()
            
            logger.log_step(t, state, {
                "forces_x": forces_x,
                "forces_y": forces_y,
                "torques": torques,
                "u": u,
                "ctrl_mode": int(ctrl.mode)  # convert enum to int for logging
            }, V)

            t += dt

    except KeyboardInterrupt:
        print("Simulation terminated!")
        logger.save()
        visualizer = LogVisualizer(log_path=logger.log_path)
        # visualizer.plot_robot1()
        visualizer.plot_target_box()
        visualizer.plot_u_with_mode()

    env.close()

if __name__ == "__main__":
    main()
