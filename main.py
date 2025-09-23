from envs.pushing_env import PushingEnv

def dummy_controller(state):
    # Just push forward with 10N for every robot
    # return [10.0 for _ in state["robots"]]
    return [0.0 for _ in state["robots"]]

def main():
    env = PushingEnv(gui=True, config_path="configs/sim_config.yaml")

    try:
        while True:
            state = env.get_state()
            actions = dummy_controller(state)
            env.apply_actions(actions)
            env.step()
    except KeyboardInterrupt:
        print("Simulation terminated!")

    env.close()

if __name__ == "__main__":
    main()
