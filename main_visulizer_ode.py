from visualizer.logvisualizer_ode import LogVisualizer

def main():
    # 시뮬레이션 로그 파일 경로 지정 (이미 저장된 로그 사용)
    # log_path = "logs/paper_2robots_simul_ode_log_20251014_155709.pkl"
    log_path = "logs/paper_3robots_simul_ode_log_20251014_154925.pkl"
    sim_config = "configs/sim_config.yaml"

    visualizer = LogVisualizer(log_path=log_path, sim_config=sim_config)

    # 필요한 그래프/애니메이션 호출
    dt = 1/240  # 시뮬레이션 timestep (로그랑 맞춰야 함)

    # visualizer.plot_animation(dt=dt, save_path="results/simulation.mp4")
    # visualizer.plot_animation(dt=dt, save_path=None)
    # visualizer.plot_combined_animation(dt=dt, interval=100, save_path="results/simulation_integrated_2robots.mp4")
    # visualizer.plot_combined_animation(dt=dt, interval=100, save_path=None)
    # visualizer.plot_target_box()
    # visualizer.plot_u_with_mode()
    # visualizer.plot_switching_data()
    # visualizer.plot_snapshot_overlay([1.0, 2.0, 8.0, 18.0])
    # visualizer.plot_snapshot_overlay([1.0, 2.0, 4.0, 18.0])
    visualizer.plot_paper()
    visualizer.show()

if __name__ == "__main__":
    main()

