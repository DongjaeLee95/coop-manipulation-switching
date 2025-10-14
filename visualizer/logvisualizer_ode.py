import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib as mpl
import math

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']  # MATLAB과 유사
plt.rcParams.update({
    "text.usetex": True,                # LaTeX 엔진 사용
    "font.family": "serif",             # LaTeX 폰트
    "text.latex.preamble": r"\usepackage{amsmath}"  # AMSmath 사용
})

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

        self.obj_d_pos_x = []
        self.obj_d_pos_y = []
        self.obj_d_yaw = []

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

            # desired object pose
            self.obj_d_pos_x.append(step["obj_d"][0])
            self.obj_d_pos_y.append(step["obj_d"][1])
            self.obj_d_yaw.append(step["obj_d"][2])

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
        self.obj_d_pos_x = np.array(self.obj_d_pos_x)
        self.obj_d_pos_y = np.array(self.obj_d_pos_y)
        self.obj_d_yaw = np.array(self.obj_d_yaw)

    def plot_target_box(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        axs = axs.flatten()
        for ax in axs:
            ax.grid(True)

        axs[0].plot(self.time, self.target_pos_x); axs[0].set_title("Target Position X [m]", fontsize=16)
        axs[0].set_xlim(self.time[0], self.time[-1])
        axs[1].plot(self.time, self.target_pos_y); axs[1].set_title("Target Position Y [m]", fontsize=16)
        axs[1].set_xlim(self.time[0], self.time[-1])
        axs[2].plot(self.time, np.degrees(self.target_yaw)); axs[2].set_title("Target Yaw Angle [deg]" , fontsize=16)
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
        for ax in axs:
            ax.grid(True)

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
        ax.grid(True)
        ax.set_aspect("equal")
        ax.set_xlim(min(self.robots_x.min(), self.target_pos_x.min() - target_size) - 0.5,
                    max(self.robots_x.max(), self.target_pos_x.max() + target_size) + 0.5)
        ax.set_ylim(min(self.robots_y.min(), self.target_pos_y.min() - target_size) - 0.5,
                    max(self.robots_y.max(), self.target_pos_y.max() + target_size) + 0.5)
        tick_fontsize = 16
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

        # robots (circle + heading)
        # robot_circles = [plt.Circle((0, 0), robot_radius, fc="blue", alpha=0.5) for _ in range(self.num_robots)]
        robot_circles = [plt.Circle((0, 0), robot_radius, fc="none", ec="k", linestyle="-", alpha=0.6, linewidth=2.0) \
                         for _ in range(self.num_robots)]
        robot_headings = [ax.plot([], [], "k-")[0] for _ in range(self.num_robots)]
        for c in robot_circles:
            ax.add_patch(c)

        # target (rectangle)
        target_rect = plt.Rectangle((-target_size, -target_size), 2*target_size, 2*target_size,
                            fc="none", ec="k", linestyle="-", alpha=0.6, linewidth=2.0)
        ax.add_patch(target_rect)

        # target orientation
        target_x_axis, = ax.plot([],[], "b-", linewidth=2.5, label="Target x-axis")
        target_y_axis, = ax.plot([],[], "r-", linewidth=2.5, label="Target y-axis")

        # desired target
        # desired_rect = plt.Rectangle((-target_size, -target_size), 2*target_size, 2*target_size,
        #                     fc="none", ec="k", linestyle="--", alpha=0.6)
        # ax.add_patch(desired_rect)
        desired_circle = plt.Circle((0,0), robot_radius, 
                                fc='none', ec="r", linestyle='-', linewidth=1.5, alpha=0.7)
        
        # target origin
        target_circle = plt.Circle((0,0), robot_radius/2, 
                                fc='k', alpha=0.8)

        desired_x_axis, = ax.plot([],[], "b-", linewidth=8.0, alpha=0.3, label="Desired x-axis")
        desired_y_axis, = ax.plot([],[], "r-", linewidth=8.0, alpha=0.3, label="Desired y-axis")


        # ext_trajs (라인, 시작점, 목표점)
        # traj_lines = [ax.plot([], [], linestyle="--", color=colors[i % len(colors)])[0]
        #             for i in range(self.num_robots)]
        # traj_starts = [ax.scatter([], [], color=colors[i % len(colors)], marker="o")
        #             for i in range(self.num_robots)]
        # traj_goals = [ax.scatter([], [], color=colors[i % len(colors)], marker="x")
        #             for i in range(self.num_robots)]
        traj_lines = [ax.plot([], [], linestyle="--", color='k')[0]
                    for i in range(self.num_robots)]
        traj_starts = [ax.scatter([], [], color='k', marker="o")
                    for i in range(self.num_robots)]
        traj_goals = [ax.scatter([], [], color='k', marker="x")
                    for i in range(self.num_robots)]

        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=18)

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
            target_circle.set_center((tx,ty))
            ax.add_patch(target_circle)

            # draw body frame axes for current target
            axis_len = target_size*0.8
            R = np.array([[np.cos(tyaw), -np.sin(tyaw)],
                         [np.sin(tyaw), np.cos(tyaw)]])
            x_axis = np.array([tx,ty]) + R @ np.array([axis_len/2,0])
            y_axis = np.array([tx,ty]) + R @ np.array([0,axis_len/2])
            target_x_axis.set_data([tx, x_axis[0]], [ty, x_axis[1]])
            target_y_axis.set_data([tx, y_axis[0]], [ty, y_axis[1]])

            # desired target
            dx = self.obj_d_pos_x[frame]
            dy = self.obj_d_pos_y[frame]
            dyaw = self.obj_d_yaw[frame]
            desired_circle.set_center((dx,dy))
            ax.add_patch(desired_circle) 
            # dtransf = transforms.Affine2D().rotate(dyaw).translate(dx,dy) + ax.transData
            # desired_rect.set_transform(dtransf)

            R_d = np.array([[np.cos(dyaw), -np.sin(dyaw)],
                         [np.sin(dyaw), np.cos(dyaw)]])
            dx_axis = np.array([dx,dy]) + R_d @ np.array([axis_len,0])
            dy_axis = np.array([dx,dy]) + R_d @ np.array([0,axis_len])
            desired_x_axis.set_data([dx, dx_axis[0]], [dy, dx_axis[1]])
            desired_y_axis.set_data([dx, dy_axis[0]], [dy, dy_axis[1]])

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
            return robot_circles + robot_headings + [target_rect, time_text,
                                                    target_x_axis, target_y_axis,
                                                    target_circle, desired_circle, desired_x_axis, desired_y_axis] \
                                 + traj_lines + traj_starts + traj_goals
            # return robot_circles + robot_headings + [target_rect, time_text,
            #                                         target_x_axis, target_y_axis,
            #                                         desired_rect, desired_x_axis, desired_y_axis] \
            #                      + traj_lines + traj_starts + traj_goals

        ani = animation.FuncAnimation(fig, update, 
                                      frames=range(0, len(self.time), frame_step),
                                      init_func=init, blit=True, interval=interval)
        # ax.legend(["Robot heading", "Target"])

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
        for ax in axs:
            ax.grid(True)

        axs[0].plot(self.time, self.V_lyap); axs[0].set_title("Lyapunov Function V" , fontsize=16)
        axs[0].set_xlim(self.time[0], self.time[-1])
        axs[1].plot(self.time, self.delta_indicator); axs[1].set_title("Delta" , fontsize=16)
        axs[1].set_xlim(self.time[0], self.time[-1])
        axs[2].plot(self.time, self.trigger); axs[2].set_title("Switch Trigger" , fontsize=16)
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

    # -------------------------------
    # Figures for paper
    # -------------------------------
    def plot_paper(self):
        fig, axs = plt.subplots(3, 1, figsize=(7, 7))
        axs = axs.flatten()
        for ax in axs:
            ax.grid(True)
        axs[0].plot(self.time, self.V_lyap, color='k')
        axs[0].set_yscale('log')
        axs[0].set_ylabel(r"$log(V)$", fontsize=16)

        u_robot_mat = []
        for j in range(len(self.time)):
            u_robot_mat.append([self.u_matrix[j, self.delta_indicator[j][i]] for i in range(self.num_robots)])
        u_robot_mat = np.array(u_robot_mat)   # shape: (T, num_robots)

        axs[1].plot(self.time, u_robot_mat[:, 0], color='k')
        axs[1].plot(self.time, u_robot_mat[:, 1], color='r', linestyle="--")
        axs[1].plot(self.time, u_robot_mat[:, 2], color='b', linestyle=":")
        axs[1].axhline(10, color="k", linestyle="--", lw=1.2)
        axs[1].set_ylabel(r"$u$", fontsize=16)
        axs[1].set_ylim(0, 10.5)

        # axs[1].set_ylabel(r"$u_{1}$ [N]", fontsize=16)
        axs[2].plot(self.time, self.delta_indicator[:,0], color='k')
        axs[2].plot(self.time, self.delta_indicator[:,1], color='r', linestyle='--')
        axs[2].plot(self.time, self.delta_indicator[:,2], color='b', linestyle=':')
        axs[2].set_ylabel(r"$\delta_{num}$", fontsize=16)
        axs[2].set_xlabel(r"Time [s]", fontsize=16)
        
        # x-axis
        for i in range(3):
            axs[i].set_xticks(np.linspace(self.time[0], self.time[-1], 11))
            axs[i].tick_params(axis='x', labelsize=13)
            axs[i].tick_params(axis='y', labelsize=13)
            axs[i].set_xlim(self.time[0], self.time[-1])

        # NAV 모드 shading
        for i in range(3):
            in_nav = False
            start_t = 0
            ctrl_modes = self.ctrl_modes
            for j in range(len(ctrl_modes)):
                if ctrl_modes[j] == 1 and not in_nav:
                    start_t = self.time[j]
                    in_nav = True
                elif ctrl_modes[j] != 1 and in_nav:
                    axs[i].axvspan(start_t, self.time[j], color='gray', alpha=0.3)
                    in_nav = False
            if in_nav:
                axs[i].axvspan(start_t, self.time[-1], color='gray', alpha=0.3)
        
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.savefig("results/paper_plot.png", dpi=600, bbox_inches='tight', pad_inches=0.05)


    def plot_snapshot_overlay(self, snapshot_times):
        """
        Overlay multiple simulation states (robots, target, desired target)
        at user-specified times on a single figure.
        """

        # TODO - 한 time stamp 3곳에서만 그림 그리고, 한곳에서는 로봇 위치 바뀜에 따라 경로 어떻게 되는지 visualize하자

        robot_radius = self.sim_config["robot"]["radius"]
        target_size = self.sim_config["target"]["size"][0]

        # 색상 팔레트 (시점별 구분)
        colors = plt.cm.plasma(np.linspace(0, 1, len(snapshot_times)))
        # line_styles = ["-", "--", "-.", ":", "-", "--"]
        line_styles = ["-"]

        # 시점 -> 프레임 인덱스로 변환
        time_array = np.array(self.time)
        indices = []
        for t in snapshot_times:
            idx = (np.abs(time_array - t)).argmin()
            indices.append(idx)

        # Figure 생성
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(min(self.robots_x.min(), self.target_pos_x.min() - target_size) - 0.5,
                    max(self.robots_x.max(), self.target_pos_x.max() + target_size) + 0.5)
        ax.set_ylim(min(self.robots_y.min(), self.target_pos_y.min() - target_size) - 0.5,
                    max(self.robots_y.max(), self.target_pos_y.max() + target_size) + 0.5)

        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Y [m]", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # 각 snapshot 시점에서 로봇/타깃/desired 그리기
        for color, idx, t in zip(colors, indices, snapshot_times):
            color = 'k'
            label_suffix = f"t = {t:.2f}s"

            # --- 로봇 ---
            for i in range(self.num_robots):
                x, y, yaw = self.robots_x[idx, i], self.robots_y[idx, i], self.robots_yaw[idx, i]
                circle = plt.Circle((x, y), robot_radius, fc='none', ec=color, lw=2.0)
                ax.add_patch(circle)
                # heading
                hx = [x, x + robot_radius * np.cos(yaw)]
                hy = [y, y + robot_radius * np.sin(yaw)]
                ax.plot(hx, hy, color=color, lw=2)

            # --- target (현재) ---
            tx, ty, tyaw = self.target_pos_x[idx], self.target_pos_y[idx], self.target_yaw[idx]
            transf = transforms.Affine2D().rotate(tyaw).translate(tx, ty) + ax.transData
            target_rect = plt.Rectangle(
                (-target_size, -target_size),
                2 * target_size, 2 * target_size,
                fc="none", ec=color, lw=2,
                linestyle=line_styles[idx % len(line_styles)],
                transform=transf
            )
            ax.add_patch(target_rect)
            target_circle = plt.Circle((tx, ty), robot_radius/2, fc='k')
            ax.add_patch(target_circle)

            # 타깃의 로컬 축 (x, y)
            axis_len = target_size * 0.8
            R = np.array([[np.cos(tyaw), -np.sin(tyaw)],
                        [np.sin(tyaw), np.cos(tyaw)]])
            x_axis = np.array([tx, ty]) + R @ np.array([axis_len/2, 0])
            y_axis = np.array([tx, ty]) + R @ np.array([0, axis_len/2])
            ax.plot([tx, x_axis[0]], [ty, x_axis[1]], color='b', lw=2)
            ax.plot([tx, y_axis[0]], [ty, y_axis[1]], color='r', lw=2)

            # --- desired target ---
            dx, dy, dyaw = self.obj_d_pos_x[idx], self.obj_d_pos_y[idx], self.obj_d_yaw[idx]
            Rd = np.array([[np.cos(dyaw), -np.sin(dyaw)],
                        [np.sin(dyaw), np.cos(dyaw)]])
            dx_axis = np.array([dx, dy]) + Rd @ np.array([axis_len, 0])
            dy_axis = np.array([dx, dy]) + Rd @ np.array([0, axis_len])
            desired_circle = plt.Circle((dx, dy), robot_radius, fc='none', ec='r', lw=1.5, linestyle='-')
            ax.add_patch(desired_circle)
            ax.plot([dx, dx_axis[0]], [dy, dx_axis[1]], color='b', lw=8, alpha=0.1)
            ax.plot([dx, dy_axis[0]], [dy, dy_axis[1]], color='r', lw=8, alpha=0.1)
            # ax.scatter(dx, dy, color=color, marker='x', s=80, label=label_suffix)

        ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
        ax.set_title("Snapshots of Robots and Target at Selected Times", fontsize=15)
        plt.tight_layout()
        plt.show()


    def plot_combined_animation(self, dt, interval=50, window_sec=5.0, save_path=None):

        time = np.array(self.time)
        V = np.array(self.V_lyap)
        ctrl_modes = np.array(self.ctrl_modes)
        u_matrix = np.array(self.u_matrix)
        delta_ind = np.array(self.delta_indicator)
        num_robots = self.num_robots

        # 로봇별 제어 입력 계산
        u_robot_mat = np.zeros((len(time), num_robots))
        for j in range(len(time)):
            for i in range(num_robots):
                u_robot_mat[j, i] = u_matrix[j, delta_ind[j][i]]

        # ----------------- Figure 설정 -----------------
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1.0, 1.0])
        ax_main = fig.add_subplot(gs[:, 0])
        ax_v = fig.add_subplot(gs[0, 1])
        ax_u = fig.add_subplot(gs[1, 1])

        robot_radius = self.sim_config["robot"]["radius"]
        target_size = self.sim_config["target"]["size"][0]

        ax_main.set_aspect("equal")
        ax_main.grid(True)
        ax_main.set_xlim(min(self.robots_x.min(), self.target_pos_x.min() - target_size) - 0.5,
                        max(self.robots_x.max(), self.target_pos_x.max() + target_size) + 0.5)
        ax_main.set_ylim(min(self.robots_y.min(), self.target_pos_y.min() - target_size) - 0.5,
                        max(self.robots_y.max(), self.target_pos_y.max() + target_size) + 0.5)
        ax_main.set_xlabel(r"$x$ [m]", fontsize=16)
        ax_main.set_ylabel(r"$y$ [m]", fontsize=16)
        ax_main.set_title("Robot and Object Trajectories", fontsize=16)

        # 로봇 객체
        robot_circles = [plt.Circle((0, 0), robot_radius, fc="none", ec="k", lw=2.0) for _ in range(num_robots)]
        robot_headings = [ax_main.plot([], [], "k-")[0] for _ in range(num_robots)]
        robot_paths = [ax_main.plot([], [], "--", lw=1.2, alpha=0.5)[0] for _ in range(num_robots)]
        for c in robot_circles:
            ax_main.add_patch(c)

        # 타깃 (현재)
        target_rect = plt.Rectangle((-target_size, -target_size), 2 * target_size, 2 * target_size,
                                    fc="none", ec="k", lw=2.0)
        ax_main.add_patch(target_rect)
        target_x_axis, = ax_main.plot([], [], "b-", lw=2.5)
        target_y_axis, = ax_main.plot([], [], "r-", lw=2.5)

        # desired 타깃
        desired_circle = plt.Circle((0, 0), robot_radius, fc="none", ec="r", lw=1.5, linestyle="-", alpha=0.7)
        ax_main.add_patch(desired_circle)
        desired_x_axis, = ax_main.plot([], [], "b-", lw=8.0, alpha=0.3)
        desired_y_axis, = ax_main.plot([], [], "r-", lw=8.0, alpha=0.3)

        # ext_trajs (라인, 시작점, 목표점)
        traj_lines = [ax_main.plot([], [], linestyle="--", color='k')[0]
                    for i in range(self.num_robots)]
        traj_starts = [ax_main.scatter([], [], color='k', marker="o")
                    for i in range(self.num_robots)]
        traj_goals = [ax_main.scatter([], [], color='k', marker="x")
                    for i in range(self.num_robots)]

        # ----------------- Lyapunov -----------------
        ax_v.set_title("Lyapunov Function V(t)", fontsize=16)
        # ax_v.set_xlabel("Time [s]")
        ax_v.grid(True)
        line_v, = ax_v.plot([], [], "k-", lw=2)

        # NAV 구간 shading
        nav_patches_v = []
        in_nav = False
        start_t = 0
        for i in range(len(time)):
            if ctrl_modes[i] == 1 and not in_nav:
                start_t = time[i]; in_nav = True
            elif ctrl_modes[i] != 1 and in_nav:
                nav_patches_v.append(ax_v.axvspan(start_t, time[i], color="gray", alpha=0.2))
                in_nav = False
        if in_nav:
            nav_patches_v.append(ax_v.axvspan(start_t, time[-1], color="gray", alpha=0.2))

        # ----------------- 제어 입력 -----------------
        colors = plt.cm.tab10(np.linspace(0, 1, num_robots))
        for i in range(num_robots):
            ax_u.plot([], [], color=colors[i], lw=2, label=f"u_{i}")
        ax_u.axhline(0, color="k", linestyle="--", lw=1.2)
        ax_u.axhline(10, color="k", linestyle="--", lw=1.2)
        ax_u.set_title("Control Inputs u(t)", fontsize=16)
        ax_u.set_xlabel("Time [s]", fontsize=16)
        # ax_u.set_ylabel("u value")
        ax_u.grid(True)
        ax_u.legend(loc="upper right")
        nav_patches_u = []
        in_nav = False
        start_t = 0
        for i in range(len(time)):
            if ctrl_modes[i] == 1 and not in_nav:
                start_t = time[i]; in_nav = True
            elif ctrl_modes[i] != 1 and in_nav:
                nav_patches_u.append(ax_u.axvspan(start_t, time[i], color="gray", alpha=0.2))
                in_nav = False
        if in_nav:
            nav_patches_u.append(ax_u.axvspan(start_t, time[-1], color="gray", alpha=0.2))

        lines_u = [ax_u.lines[i] for i in range(num_robots)]
        time_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, fontsize=16)

        # ----------------- 애니메이션 -----------------
        def init():
            for c in robot_circles: c.set_center((0, 0))
            for h in robot_headings: h.set_data([], [])
            # for p in robot_paths: p.set_data([], [])
            target_rect.set_xy((-target_size, -target_size))
            target_rect.angle = 0
            line_v.set_data([], [])
            for lu in lines_u: lu.set_data([], [])
            time_text.set_text("")
            return robot_circles + robot_headings + [target_rect, target_x_axis, target_y_axis, desired_circle,
                                                    desired_x_axis, desired_y_axis, line_v, time_text] + lines_u + \
                                                    traj_lines + traj_starts + traj_goals

        def update(frame):
            # 시간 범위
            t_end = time[frame]
            t_start = max(0, t_end - window_sec)
            mask = (time >= t_start) & (time <= t_end)

            # (1) 로봇 업데이트
            for i, (c, h, p) in enumerate(zip(robot_circles, robot_headings, robot_paths)):
                x = self.robots_x[frame, i]; y = self.robots_y[frame, i]; yaw = self.robots_yaw[frame, i]
                c.set_center((x, y))
                hx = [x, x + robot_radius * np.cos(yaw)]
                hy = [y, y + robot_radius * np.sin(yaw)]
                h.set_data(hx, hy)
                # p.set_data(self.robots_x[:frame, i], self.robots_y[:frame, i])

            # (2) 타깃 (현재 + desired)
            tx, ty, tyaw = self.target_pos_x[frame], self.target_pos_y[frame], self.target_yaw[frame]
            dx, dy, dyaw = self.obj_d_pos_x[frame], self.obj_d_pos_y[frame], self.obj_d_yaw[frame]

            # 현재 타깃
            transf = transforms.Affine2D().rotate(tyaw).translate(tx, ty) + ax_main.transData
            target_rect.set_transform(transf)
            axis_len = target_size * 0.8
            R = np.array([[np.cos(tyaw), -np.sin(tyaw)], [np.sin(tyaw), np.cos(tyaw)]])
            x_axis = np.array([tx, ty]) + R @ np.array([axis_len / 2, 0])
            y_axis = np.array([tx, ty]) + R @ np.array([0, axis_len / 2])
            target_x_axis.set_data([tx, x_axis[0]], [ty, x_axis[1]])
            target_y_axis.set_data([tx, y_axis[0]], [ty, y_axis[1]])

            # desired 타깃
            desired_circle.set_center((dx, dy))
            R_d = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
            dx_axis = np.array([dx, dy]) + R_d @ np.array([axis_len, 0])
            dy_axis = np.array([dx, dy]) + R_d @ np.array([0, axis_len])
            desired_x_axis.set_data([dx, dx_axis[0]], [dy, dx_axis[1]])
            desired_y_axis.set_data([dx, dy_axis[0]], [dy, dy_axis[1]])

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

            # (3) Lyapunov window 업데이트
            line_v.set_data(time[mask], V[mask])
            ax_v.set_xlim(t_start, t_end)
            y_min, y_max = V[mask].min(), V[mask].max()
            if y_min == y_max: y_min -= 0.1; y_max += 0.1
            ax_v.set_ylim(y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max))

            # (4) u window 업데이트
            for i in range(num_robots):
                lines_u[i].set_data(time[mask], u_robot_mat[mask, i])
            ax_u.set_xlim(t_start, t_end)

            time_text.set_text(f"t = {t_end:.2f}s")
            return robot_circles + robot_headings + [target_rect, target_x_axis, target_y_axis,
                                                    desired_circle, desired_x_axis, desired_y_axis,
                                                    line_v, time_text] + lines_u + \
                                                    traj_lines + traj_starts + traj_goals

        # ----------------- 애니메이션 실행 -----------------
        frame_step = 10
        ani = animation.FuncAnimation(
            fig, update, frames=range(0, len(time), frame_step),
            init_func=init, blit=False, interval=interval
        )

        fps = int(1.0 / (dt * frame_step))
        if save_path is not None:
            ani.save(save_path, writer="ffmpeg", fps=fps, dpi=500)
            print(f"Animation saved to {save_path}")
        
        # fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.15, hspace=0.25)
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.show()