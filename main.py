from envs.pushing_env import PushingEnv
from ctrller.robotCtrller import RobotCtrller

def dummy_controller(state):
    # Just push forward with 10N for every robot
    # return [1.0 for _ in state["robots"]]
    return [0.0 for _ in state["robots"]], [3.0 for _ in state["robots"]], [0.0 for _ in state["robots"]]

def main():
    env = PushingEnv(gui=True, sim_config_path="configs/sim_config.yaml", ctrl_config_path="configs/ctrl_config.yaml")
    ctrl = RobotCtrller(config_path="configs/ctrl_config.yaml")

    try:
        # print simulation setting
        env.print_dynamics_info(env.target_box)
        env.print_dynamics_info(env.robots[0])
        while True:
            state = env.get_state()
            # print(state)
            # forces_x, forces_y, torques = dummy_controller(state)
            forces_x, forces_y, torques = ctrl.compute_actions(state)
            env.apply_actions(forces_x, forces_y, torques)
            env.step()
            for robot in env.robots:
                env.draw_body_frame(robot)
    except KeyboardInterrupt:
        print("Simulation terminated!")

    env.close()

if __name__ == "__main__":
    main()
