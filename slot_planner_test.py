import numpy as np
import yaml
import matplotlib.pyplot as plt
from planner.robot_slot_planner import RobotSlotPlanner

def main():
    # -------------------------------
    # 1. 환경 파라미터 설정
    # -------------------------------
    sim_config_path = "configs/sim_config.yaml"
    with open(sim_config_path, "r") as f:
        sim_config = yaml.safe_load(f)
        L = (sim_config["target"]["size"][0])/2 # target half-length
        r = sim_config["robot"]["radius"] # robot radius
    
    planner = RobotSlotPlanner(sim_config_path)

    # -------------------------------
    # 2. 가짜 state 생성
    # -------------------------------
    # target object (world frame)
    obj_state = {
        "position": np.array([0.0, 0.0, 0.0]),      # 중심
        "rotation_matrix": np.eye(3).flatten(),     # 회전 없음
    }

    obj_pos = np.array(obj_state["position"][:2])
    psi = np.deg2rad(0)
    R = np.array([[np.cos(psi), -np.sin(psi)],
              [np.sin(psi),  np.cos(psi)]])
    R3 = np.eye(3)
    R3[:2,:2] = R
    obj_state["rotation_matrix"] = R3.flatten()

    # -------------------------------
    # 3. 로봇 초기 위치 = 특정 슬롯
    # -------------------------------
    previous_delta_indicator = [7,6]   # 로봇들이 처음 붙어 있는 슬롯 번호
    robots = []
    for s in previous_delta_indicator:
        local_pos, _ = planner._slot_local(s)
        world_pos = obj_pos + R @ local_pos
        robots.append({"position": np.array([world_pos[0], world_pos[1], 0.0]),
                       "rotation_matrix": np.eye(3).flatten()
                       })

    # -------------------------------
    # 4. delta 설정 (새로운 목표 슬롯 집합)
    # -------------------------------
    # 예: slot 1, 3, 5번 활성화
    delta = [0 for _ in range(8)]
    delta[4] = 1
    delta[5] = 1
    # delta[2] = 1

    # -------------------------------
    # 3. Planner 실행
    # -------------------------------
    new_delta_indicator, trajectories = planner.compute(robots, obj_state, delta, previous_delta_indicator)

    print("Initial slots:", previous_delta_indicator)
    print("New delta_indicator (goal slots):", new_delta_indicator)

    # -------------------------------
    # 4. Visualization
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    # target object 사각형 그리기
    obj_size = 4*L
    square = plt.Rectangle((-2*L, -2*L), obj_size, obj_size,
                           fill=True, color="gray", alpha=0.3)
    t = plt.matplotlib.transforms.Affine2D().rotate(psi) + ax.transData
    square.set_transform(t)
    ax.add_patch(square)

    # 로봇 궤적
    colors = ["r", "g", "b", "c", "m", "y"]
    for i, traj in trajectories["positions"].items():
        traj = np.array(traj)
        ax.plot(traj[:,0], traj[:,1], color=colors[i%len(colors)], label=f"Robot {i}")
        ax.scatter(traj[0,0], traj[0,1], color=colors[i%len(colors)], marker="o")  # start
        ax.scatter(traj[-1,0], traj[-1,1], color=colors[i%len(colors)], marker="x") # goal

    ax.set_aspect("equal")
    ax.set_title("RobotSlotPlanner Trajectories")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
