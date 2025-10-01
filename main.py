from envs.pushing_env import PushingEnv
from ctrller.robotCtrller import RobotCtrller
from visualizer.logvisualizer import LogVisualizer
from logger.simulation_logger import SimulationLogger
from ctrller.objCtrller import ObjCtrller
from ctrller.switching_law import SwitingLaw
from planner.robot_slot_planner import RobotSlotPlanner
from enum import IntEnum
import numpy as np

class Mode(IntEnum):
    CON = 0
    NAV = 1

def main():
    
    t = 0.0
    dt = 1 / 240

    env = PushingEnv(gui=True, sim_config_path="configs/sim_config.yaml")
    ctrl = RobotCtrller(ctrl_config_path="configs/ctrl_config.yaml", 
                        sim_config_path="configs/sim_config.yaml")
    obj_ctrl = ObjCtrller(dt, True, obj_ctrl_config_path="configs/obj_ctrl_config.yaml", 
                          sim_config_path="configs/sim_config.yaml")
    switching_law = SwitingLaw(obj_ctrl_config_path="configs/obj_ctrl_config.yaml", 
                               sim_config_path="configs/sim_config.yaml")
    slot_planner = RobotSlotPlanner(sim_config_path="configs/sim_config.yaml")

    logger = SimulationLogger()
    logger.log_environment(env)

    try:
        # print simulation setting
        # env.print_dynamics_info(env.target_box)
        # env.print_dynamics_info(env.robots[0])
 
        # initialize delta's
        delta_indicator = np.array(env.robot_init_id)
        delta = np.zeros(8)
        for i in delta_indicator:
            delta[i] = 1

        # set object desired pos, orientation
        obj_d = np.array([1, 1, np.deg2rad(30), 0.0, 0.0, 0.0])

        while True:
            state = env.get_state()
            
            # switch_trigger = False
            # compt_time = np.nan
            # rho = np.nan
            # switching law & object(target) controller
            _, delta, switch_trigger, compt_time, rho = switching_law.compute(state["target"], obj_d, delta)
            u, _, V, _, _ = obj_ctrl.compute(state["target"], obj_d, delta)

            # multi-agent path finding (collision-free)
            if switch_trigger:
                delta_indicator, ext_trajs = slot_planner.compute(state["robots"], state["target"], 
                                                                    delta, delta_indicator)
            else:
                ext_trajs = None

            # robot motion generator & robot controller
            pos_ds, ori_ds = ctrl.motion_planner(state["target"], delta_indicator, ext_trajs)
            forces_x, forces_y, torques = ctrl.compute_actions(state["robots"], u, pos_ds, ori_ds, switch_trigger)
            if ctrl.mode == Mode.NAV:
                u = np.zeros(len(delta))

            env.apply_actions(forces_x, forces_y, torques)
            env.step()
            
            logger.log_step(t, state, {
                "forces_x": forces_x,
                "forces_y": forces_y,
                "torques": torques,
                "ctrl_mode": int(ctrl.mode)  # convert enum to int for logging
            }, u, {
                "V_lyap": V,
                "delta": delta,
                "delta_indicator": delta_indicator,
                "trigger": switch_trigger,
                "MILP_compt_time": compt_time,
                "MILP_rho": rho
            })

            t += dt

    except KeyboardInterrupt:
        print("Simulation terminated!")
        logger.save()
        visualizer = LogVisualizer(log_path=logger.log_path)
        # visualizer.plot_robot1()
        visualizer.plot_target_box()
        visualizer.plot_u_with_mode()
        visualizer.show()

    env.close()

if __name__ == "__main__":
    main()
