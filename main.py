from envs.pushing_env import PushingEnv
from ctrller.robotCtrller import RobotCtrller
from visualizer.logvisualizer import LogVisualizer
from logger.simulation_logger import SimulationLogger
from ctrller.objCtrller import ObjCtrller

def main():
    env = PushingEnv(gui=True, sim_config_path="configs/sim_config.yaml")
    ctrl = RobotCtrller(ctrl_config_path="configs/ctrl_config.yaml", sim_config_path="configs/sim_config.yaml")
    obj_ctrl = ObjCtrller(obj_ctrl_config_path="configs/obj_ctrl_config.yaml", sim_config_path="configs/sim_config.yaml")

    logger = SimulationLogger()

    logger.log_environment(env)

    try:
        # print simulation setting
        env.print_dynamics_info(env.target_box)
        env.print_dynamics_info(env.robots[0])

        t = 0.0
        dt = 1 / 240

        while True:
            state = env.get_state()
            # TODO - put multi-agent path finder module
            ext_trajs = None
            pos_ds, ori_ds = ctrl.motion_planner(state["target"], ext_trajs)
            forces_x, forces_y, torques = ctrl.compute_actions(state["robots"], pos_ds, ori_ds)
            env.apply_actions(forces_x, forces_y, torques)
            env.step()
            
            logger.log_step(t, state, {
                "forces_x": forces_x,
                "forces_y": forces_y,
                "torques": torques
            })

            t += dt

    except KeyboardInterrupt:
        print("Simulation terminated!")
        logger.save()
        visualizer = LogVisualizer(log_path=logger.log_path)
        visualizer.plot_robot1()
        visualizer.plot_target_box()

    env.close()

if __name__ == "__main__":
    main()
