import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import math

class LogVisualizer:
    def __init__(self, log_path="logs/simulation_log.pkl", sim_config="configs/sim_config.yaml"):
        with open(log_path, "rb") as f:
            self.data = pickle.load(f)

        with open(sim_config, "r") as f:
            self.sim_config = yaml.safe_load(f)

        self.time = []
        self.robots_x = []   # 모든 로봇 trajectory x
        self.robots_y = []   # 모든 로봇 trajectory y
        self.robots_yaw = [] # 모든 로봇 yaw

        self.target_pos_x = []
        self.target_pos_y = []
        self.target_yaw = []
        self.target_vel_x = []
        self.target_vel_y = []
        self.target_ang_vel = []

        self.V_lyap = []
        self.delta = []
        self.delta_indicator = []
        self.trigger = []
        self.MILP_compt_time = []
        self.MILP_rho = []

        self.u_matrix = []        # Will store control inputs per time step
        self.ctrl_modes = []      # Will store mode (0=CON, 1=NAV) per time step
        self.num_robots = len(self.data["steps"][0]["robots"])

        self.ext_trajs = []

        self._extract_data()

    def show(self):
        plt.show()

    def _rotation_to_yaw(self, R_flat):
        # rotation_matrix는 pybullet 스타일로 flat (9,)일 수 있음
        R = np.array(R_flat).reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])

    def _extract_data(self):
        for step in self.data["steps"]:
            self.time.append(step["time"])
            robots = step["robots"]
            target = step["target"]
            switching = step["switching"]
            u = step["obj_actions"]

            # 로봇들 위치 저장
            self.robots_x.append([r["position"][0] for r in robots])
            self.robots_y.append([r["position"][1] for r in robots])
            self.robots_yaw.append([
                np.arctan2(r["rotation_matrix"][3], r["rotation_matrix"][0])
                for r in robots
            ])

            # Target box data
            self.target_pos_x.append(target["position"][0])
            self.target_pos_y.append(target["position"][1])
            self.target_yaw.append(np.arctan2(target["rotation_matrix"][3], target["rotation_matrix"][0]))

            # switching law-related
            self.V_lyap.append(switching["V_lyap"])
            self.delta.append(switching["delta"])
            self.delta_indicator.append(switching["delta_indicator"])
            self.trigger.append(switching["trigger"])
            self.MILP_compt_time.append(switching["MILP_compt_time"])
            self.MILP_rho.append(switching["MILP_rho"])

            # Extract u for all robots in this step
            self.u_matrix.append(u)

            # Controller mode
            self.ctrl_modes.append(step.get("ctrl_mode", 0))

            # external trajectory
            self.ext_trajs.append(step.get("ext_trajs", None))

        self.u_matrix = np.array(self.u_matrix)
        self.ctrl_modes = np.array(self.ctrl_modes)
        self.delta = np.array(self.delta)
        self.delta_indicator = np.array(self.delta_indicator)
        self.robots_x = np.array(self.robots_x)
        self.robots_y = np.array(self.robots_y)
        self.robots_yaw = np.array(self.robots_yaw)
        self.target_pos_x = np.array(self.target_pos_x)
        self.target_pos_y = np.array(self.target_pos_y)
        self.target_yaw = np.array(self.target_yaw)

        self.MILP_compt_time = np.array(self.MILP_compt_time)
        self.MILP_rho = np.array(self.MILP_rho)

    def plot_target_box(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        axs = axs.flatten()

        axs[0].plot(self.time, self.target_pos_x); axs[0].set_title("Target Position X [m]")
        axs[0].set_xlim(self.time[0], self.time[-1])
        axs[1].plot(self.time, self.target_pos_y); axs[1].set_title("Target Position Y [m]")
        axs[1].set_xlim(self.time[0], self.time[-1])
        axs[2].plot(self.time, np.degrees(self.target_yaw)); axs[2].set_title("Target Yaw Angle [deg]")
        axs[2].set_xlim(self.time[0], self.time[-1])

        # axs[3].plot(self.time, self.target_vel_x); axs[3].set_title("Target Velocity X [m/s]")
        # axs[4].plot(self.time, self.target_vel_y); axs[4].set_title("Target Velocity Y [m/s]")
        # axs[5].plot(self.time, self.target_ang_vel); axs[5].set_title("Target Angular Velocity [rad/s]")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(True)

        plt.tight_layout()        

    def plot_u_with_mode(self):
        time = self.time
        u_matrix = self.u_matrix              # shape: (T, num_slots)
        delta_ind = self.delta_indicator      # shape: (T, num_robots)

        # 각 time-step에서 로봇별 힘 u를 뽑기
        u_robot_mat = []
        for j in range(len(time)):
            u_robot_mat.append([u_matrix[j, delta_ind[j][i]] for i in range(self.num_robots)])
        u_robot_mat = np.array(u_robot_mat)   # shape: (T, num_robots)

        ctrl_modes = self.ctrl_modes

        num_cols = 2
        num_rows = math.ceil(self.num_robots / num_cols)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows), sharex=True)
        axs = axs.flatten()

        for i in range(self.num_robots):
            axs[i].plot(time, u_robot_mat[:, i], label=f"Robot {i} force u")
            axs[i].set_ylabel(f"u_{i}")
            axs[i].set_xlim(self.time[0], self.time[-1])
            axs[i].grid(True)

            # NAV 모드 shading
            in_nav = False
            start_t = 0
            for j in range(len(ctrl_modes)):
                if ctrl_modes[j] == 1 and not in_nav:
                    start_t = time[j]
                    in_nav = True
                elif ctrl_modes[j] != 1 and in_nav:
                    axs[i].axvspan(start_t, time[j], color='gray', alpha=0.3)
                    in_nav = False
            if in_nav:
                axs[i].axvspan(start_t, time[-1], color='gray', alpha=0.3)

            axs[i].legend()

        # 사용 안 한 subplot 숨기기
        for j in range(self.num_robots, len(axs)):
            fig.delaxes(axs[j])

        axs[-1].set_xlabel("Time [s]")
        plt.suptitle("Robot Control Forces (u) with NAV Mode Highlighted")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
    # -------------------------------
    # Trajectory Animation with Orientation
    # -------------------------------
    def plot_animation(self, dt, interval=50, save_path = None):
        
        frame_step = 10

        robot_radius = self.sim_config["robot"]["radius"]
        target_size = self.sim_config["target"]["size"][0]
        colors = ["r", "g", "b", "c", "m", "y"]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_xlim(min(self.robots_x.min(), self.target_pos_x.min()) - 1,
                    max(self.robots_x.max(), self.target_pos_x.max()) + 1)
        ax.set_ylim(min(self.robots_y.min(), self.target_pos_y.min()) - 1,
                    max(self.robots_y.max(), self.target_pos_y.max()) + 1)

        # robots (circle + heading)
        robot_circles = [plt.Circle((0, 0), robot_radius, fc="blue", alpha=0.5) for _ in range(self.num_robots)]
        robot_headings = [ax.plot([], [], "k-")[0] for _ in range(self.num_robots)]
        for c in robot_circles:
            ax.add_patch(c)

        # target (rectangle)
        target_rect = plt.Rectangle((-target_size, -target_size), 2*target_size, 2*target_size,
                            fc="red", alpha=0.3)
        ax.add_patch(target_rect)

        # ext_trajs (라인, 시작점, 목표점)
        traj_lines = [ax.plot([], [], linestyle="--", color=colors[i % len(colors)])[0]
                    for i in range(self.num_robots)]
        traj_starts = [ax.scatter([], [], color=colors[i % len(colors)], marker="o")
                    for i in range(self.num_robots)]
        traj_goals = [ax.scatter([], [], color=colors[i % len(colors)], marker="x")
                    for i in range(self.num_robots)]

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init():
            for c in robot_circles:
                c.set_center((0, 0))
            for h in robot_headings:
                h.set_data([], [])
            target_rect.set_xy((-target_size, -target_size))
            target_rect.angle = 0
            time_text.set_text("")
            return robot_circles + robot_headings + [target_rect, time_text]

        def update(frame):
            # robots
            for i, (c, h) in enumerate(zip(robot_circles, robot_headings)):
                x = self.robots_x[frame, i]
                y = self.robots_y[frame, i]
                yaw = self.robots_yaw[frame, i]
                c.set_center((x, y))
                # heading line
                hx = [x, x + robot_radius * np.cos(yaw)]
                hy = [y, y + robot_radius * np.sin(yaw)]
                h.set_data(hx, hy)

            # target
            tx = self.target_pos_x[frame]
            ty = self.target_pos_y[frame]
            tyaw = self.target_yaw[frame]
            transf = transforms.Affine2D().rotate(tyaw).translate(tx, ty) + ax.transData
            target_rect.set_transform(transf)

            # ext_trajs (ctrl_mode == NAV일 때만)
            if self.ctrl_modes[frame] == 1:  # NAV
                recent_ext = None
                for f in range(frame, -1, -1):
                    if self.ext_trajs[f] is not None:
                        recent_ext = self.ext_trajs[f]
                        break
                if recent_ext is not None:
                    for i, line in enumerate(traj_lines):
                        traj = np.array(recent_ext["positions"][i])
                        line.set_data(traj[:, 0], traj[:, 1])
                        traj_starts[i].set_offsets(traj[0])
                        traj_goals[i].set_offsets(traj[-1])
            else:  # CON
                for line in traj_lines:
                    line.set_data([], [])
                for s in traj_starts + traj_goals:
                    s.set_offsets(np.empty((0, 2)))

            time_text.set_text(f"t = {self.time[frame]:.2f}s")
            return robot_circles + robot_headings + [target_rect, time_text] + traj_lines + traj_starts + traj_goals

        ani = animation.FuncAnimation(fig, update, 
                                      frames=range(0, len(self.time), frame_step),
                                      init_func=init, blit=True, interval=interval)
        ax.legend(["Robot heading", "Target"])

        fps = int(1.0/(dt * frame_step))
         # Save if path is given
        if save_path is not None:
            ani.save(save_path, writer="ffmpeg", fps=fps)  # interval [ms] → fps
            print(f"Animation saved to {save_path}")

        plt.show()
        return ani

    # -------------------------------
    # Switching Law Data
    # -------------------------------
    def plot_switching_data(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        axs = axs.flatten()

        axs[0].plot(self.time, self.V_lyap); axs[0].set_title("Lyapunov Function V")
        axs[0].set_xlim(self.time[0], self.time[-1])
        axs[1].plot(self.time, self.delta_indicator); axs[1].set_title("Delta")
        axs[1].set_xlim(self.time[0], self.time[-1])
        axs[2].plot(self.time, self.trigger); axs[2].set_title("Switch Trigger")
        axs[2].set_xlim(self.time[0], self.time[-1])
        # MILP computation time
        axs[3].plot(
            self.time,
            [np.nan if v is None else v for v in self.MILP_compt_time],
            marker="o", linestyle="None"
        )
        axs[3].set_title("MILP Computation Time [s]")
        axs[3].set_xlim(self.time[0], self.time[-1])

        # MILP rho
        axs[4].plot(
            self.time,
            [np.nan if v is None else v for v in self.MILP_rho],
            marker="o", linestyle="None"
        )
        axs[4].set_title("MILP rho")
        axs[4].set_xlim(self.time[0], self.time[-1])

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.grid(True)

        plt.tight_layout()