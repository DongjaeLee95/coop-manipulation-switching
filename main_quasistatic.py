from envs.quasistatic_env import QuasiStaticEnv
from ctrller.robotCtrller_quasistatic import RobotCtrller_QS
from visualizer.logvisualizer_quasistatic import LogVisualizer_QS
from logger.simulation_logger_quasistatic import SimulationLogger_QS
from trajectory_gen import Traj_gen
import numpy as np

def main():
    env = QuasiStaticEnv(gui=True, sim_config_path="configs/sim_config_quasistatic.yaml", ctrl_config_path="configs/ctrl_config_quasistatic.yaml")
    ctrl = RobotCtrller_QS(ctrl_config_path="configs/ctrl_config_quasistatic.yaml", sim_config_path="configs/sim_config_quasistatic.yaml")
    logger = SimulationLogger_QS()

    logger.log_environment(env)

    try:
        # print simulation setting
        env.print_dynamics_info(env.robot)

        t = 0.0
        dt = 1 / 240
        trajectory = Traj_gen(radius=0.5, omega=0.2, yaw=0.0)

        while True:
            state = env.get_state()
            ref = trajectory.get_reference(t)
            force_x, force_y, torque = ctrl.compute_action(state, ref)  # single robot
            env.apply_action(force_x, force_y, torque)
            env.step()

            logger.log_step(
                t,
                [state],  # robot_state
                [{
                    "position": ref["position"].flatten().tolist(),
                    "yaw": float(ref["yaw"])
                }],
                {
                    "forces_x": [force_x],
                    "forces_y": [force_y],
                    "torques": [torque]
                }
            )

            t += dt

    except KeyboardInterrupt:
        print("Simulation terminated!")
        logger.save()
        visualizer = LogVisualizer_QS(log_path=logger.log_path)
        visualizer.plot_robot()

    env.close()

if __name__ == "__main__":
    main()
