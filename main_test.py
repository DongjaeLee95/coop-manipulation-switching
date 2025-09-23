from envs.pushing_env import PushingEnv

def main():
    env = PushingEnv(gui=True)

    try:
        while True:
            env.step_manual()
    except KeyboardInterrupt:
        print("Simulation terminated!")

    env.close()

if __name__ == "__main__":
    main()
