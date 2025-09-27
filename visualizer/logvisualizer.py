import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

class LogVisualizer:
    def __init__(self, log_path="logs/simulation_log.pkl"):
        with open(log_path, "rb") as f:
            self.data = pickle.load(f)

        self.time = []
        self.pos_x = []
        self.pos_y = []
        self.yaw = []
        self.force_x = []
        self.force_y = []
        self.torque = []
        self.vel_x = []
        self.vel_y = []
        self.ang_vel = []

        self.target_pos_x = []
        self.target_pos_y = []
        self.target_yaw = []
        self.target_vel_x = []
        self.target_vel_y = []
        self.target_ang_vel = []

        self.u_matrix = []        # Will store control inputs per time step
        self.ctrl_modes = []      # Will store mode (0=CON, 1=NAV) per time step

        self.num_robots = len(self.data["steps"][0]["robots"])

        self._extract_data()

    def _quat_to_yaw(self, quat):
        x, y, z, w = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _extract_data(self):
        for step in self.data["steps"]:
            self.time.append(step["time"])
            robot = step["robots"][0]
            action = step["actions"][0]
            target = step["target"]

            # Robot 1 data
            self.pos_x.append(robot["position"][0])
            self.pos_y.append(robot["position"][1])
            self.yaw.append(self._quat_to_yaw(robot["orientation_quat"]))
            self.force_x.append(action["force_x"])
            self.force_y.append(action["force_y"])
            self.torque.append(action["torque"])
            self.vel_x.append(robot["linear_velocity"][0])
            self.vel_y.append(robot["linear_velocity"][1])
            self.ang_vel.append(robot["angular_velocity"][2])

            # Target box data
            self.target_pos_x.append(target["position"][0])
            self.target_pos_y.append(target["position"][1])
            self.target_yaw.append(self._quat_to_yaw(target["orientation_quat"]))
            self.target_vel_x.append(target["linear_velocity"][0])
            self.target_vel_y.append(target["linear_velocity"][1])
            self.target_ang_vel.append(target["angular_velocity"][2])

            # Extract u for all robots in this step
            u_values = [a.get("u", 0.0) for a in step["actions"]]
            self.u_matrix.append(u_values)

            # Extract controller mode (default to 0 if not present)
            self.ctrl_modes.append(step.get("ctrl_mode", 0))

        self.u_matrix = np.array(self.u_matrix)
        self.ctrl_modes = np.array(self.ctrl_modes)

    def plot_robot1(self):
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        axs = axs.flatten()

        axs[0].plot(self.time, self.pos_x); axs[0].set_title("Position X [m]")
        axs[1].plot(self.time, self.pos_y); axs[1].set_title("Position Y [m]")
        axs[2].plot(self.time, np.degrees(self.yaw)); axs[2].set_title("Yaw Angle [deg]")

        axs[3].plot(self.time, self.force_x); axs[3].set_title("Force X [N]")
        axs[4].plot(self.time, self.force_y); axs[4].set_title("Force Y [N]")
        axs[5].plot(self.time, self.torque);  axs[5].set_title("Torque [Nm]")

        axs[6].plot(self.time, self.vel_x); axs[6].set_title("Velocity X [m/s]")
        axs[7].plot(self.time, self.vel_y); axs[7].set_title("Velocity Y [m/s]")
        axs[8].plot(self.time, self.ang_vel); axs[8].set_title("Angular Velocity [rad/s]")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_target_box(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        axs = axs.flatten()

        axs[0].plot(self.time, self.target_pos_x); axs[0].set_title("Target Position X [m]")
        axs[1].plot(self.time, self.target_pos_y); axs[1].set_title("Target Position Y [m]")
        axs[2].plot(self.time, np.degrees(self.target_yaw)); axs[2].set_title("Target Yaw Angle [deg]")

        axs[3].plot(self.time, self.target_vel_x); axs[3].set_title("Target Velocity X [m/s]")
        axs[4].plot(self.time, self.target_vel_y); axs[4].set_title("Target Velocity Y [m/s]")
        axs[5].plot(self.time, self.target_ang_vel); axs[5].set_title("Target Angular Velocity [rad/s]")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_u_with_mode(self):
        if "u" not in self.data["steps"][0]["actions"][0]:
            print("No 'u' data found in the logs. Skipping u plot.")
            return

        time = self.time
        u_matrix = self.u_matrix
        ctrl_modes = self.ctrl_modes

        num_cols = 2
        num_rows = math.ceil(self.num_robots / num_cols)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows), sharex=True)
        axs = axs.flatten()  # Make indexing uniform

        for i in range(self.num_robots):
            axs[i].plot(time, u_matrix[:, i], label=f"u_{i}")
            axs[i].set_ylabel(f"u_{i}")
            axs[i].grid(True)

            # Shade background where ctrl_mode == 1 (NAV)
            in_nav = False
            start_t = 0
            for j in range(len(ctrl_modes)):
                if ctrl_modes[j] == 1 and not in_nav:
                    start_t = time[j]
                    in_nav = True
                elif ctrl_modes[j] != 1 and in_nav:
                    end_t = time[j]
                    axs[i].axvspan(start_t, end_t, color='gray', alpha=0.3)
                    in_nav = False
            if in_nav:
                axs[i].axvspan(start_t, time[-1], color='gray', alpha=0.3)

            axs[i].legend()

        # Hide any unused subplots if num_robots is odd
        for j in range(self.num_robots, len(axs)):
            fig.delaxes(axs[j])

        axs[-1].set_xlabel("Time [s]")
        plt.suptitle("Control Inputs (u) with NAV Mode Highlighted")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()