from visualizer.logvisualizer_ode import LogVisualizer

def main():
    # 시뮬레이션 로그 파일 경로 지정 (이미 저장된 로그 사용)
    log_path = "logs/simul_ode_log_20250930_154219.pkl"   # 실제 로그 파일 경로 넣으세요
    sim_config = "configs/sim_config.yaml"

    visualizer = LogVisualizer(log_path=log_path, sim_config=sim_config)

    # 필요한 그래프/애니메이션 호출
    dt = 1/240  # 시뮬레이션 timestep (로그랑 맞춰야 함)

    # visualizer.plot_animation(dt=dt, save_path="results/simulation.mp4")
    visualizer.plot_target_box()
    visualizer.plot_u_with_mode()
    visualizer.plot_switching_data()
    visualizer.show()

if __name__ == "__main__":
    main()
