import pickle
import numpy as np
import matplotlib.pyplot as plt

class LogVisualizer_QS:
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
        self.ref_x = []
        self.ref_y = []
        self.ref_yaw = []

        self._extract_data()

    def _quat_to_yaw(self, quat):
        x, y, z, w = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _extract_data(self):
        for step in self.data["steps"]:
            self.time.append(step["time"])
            robot = step["robot"][0]  # unpack the single robot from the list
            action = step["actions"][0]
            ref_list = step.get("reference", [])
            ref = ref_list[0] if isinstance(ref_list, list) and len(ref_list) > 0 else {}


            self.pos_x.append(robot["position"][0])
            self.pos_y.append(robot["position"][1])
            self.yaw.append(self._quat_to_yaw(robot["orientation_quat"]))
            self.force_x.append(action["force_x"])
            self.force_y.append(action["force_y"])
            self.torque.append(action["torque"])
            self.vel_x.append(robot["linear_velocity"][0])
            self.vel_y.append(robot["linear_velocity"][1])
            self.ang_vel.append(robot["angular_velocity"][2])

            ref_pos = ref.get("position", [None, None])
            self.ref_x.append(ref_pos[0])
            self.ref_y.append(ref_pos[1])
            self.ref_yaw.append(ref.get("yaw", None))


    def plot_robot(self):
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        axs = axs.flatten()

        # Position
        axs[0].plot(self.time, self.pos_x, label="Actual")
        axs[0].plot(self.time, self.ref_x, '--', label="Reference")
        axs[0].set_title("Position X [m]")

        axs[1].plot(self.time, self.pos_y, label="Actual")
        axs[1].plot(self.time, self.ref_y, '--', label="Reference")
        axs[1].set_title("Position Y [m]")

        axs[2].plot(self.time, np.degrees(self.yaw), label="Actual")
        axs[2].plot(self.time, np.degrees(self.ref_yaw), '--', label="Reference")
        axs[2].set_title("Yaw Angle [deg]")

        # Forces
        axs[3].plot(self.time, self.force_x); axs[3].set_title("Force X [N]")
        axs[4].plot(self.time, self.force_y); axs[4].set_title("Force Y [N]")
        axs[5].plot(self.time, self.torque);  axs[5].set_title("Torque [Nm]")

        # Velocities
        axs[6].plot(self.time, self.vel_x); axs[6].set_title("Velocity X [m/s]")
        axs[7].plot(self.time, self.vel_y); axs[7].set_title("Velocity Y [m/s]")
        axs[8].plot(self.time, self.ang_vel); axs[8].set_title("Angular Velocity [rad/s]")

        # Add labels and grid
        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(True)
            if ax.get_lines():  # only add legend if there's something to show
                ax.legend()

        plt.tight_layout()
        plt.show()

