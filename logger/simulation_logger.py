import os
import pickle
from datetime import datetime

class SimulationLogger:
    def __init__(self, log_dir="logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_log_{timestamp}.pkl"
        self.log_path = os.path.join(log_dir, filename)
        self.data = {
            "environment": {},
            "steps": []
        }

    def log_environment(self, env):
        self.data["environment"] = {
            "gravity": env.gravity,
            "friction": env.friction,
            "target_mass": env.get_mass(env.target_box),
            "target_inertia": env.get_inertia(env.target_box),
            "robot_masses": [env.get_mass(r) for r in env.robots],
            "robot_inertias": [env.get_inertia(r) for r in env.robots]
        }

    def log_step(self, time, state, actions):
        u = actions.get("u", None)
        ctrl_mode = actions.get("ctrl_mode", None)
        
        step_data = {
            "time": time,
            "robots": state["robots"],
            "target": state["target"],
            "actions": [
                {
                    "force_x": actions["forces_x"][i],
                    "force_y": actions["forces_y"][i],
                    "torque": actions["torques"][i],
                    "u": float(u[i]) if u is not None else None
                }
                for i in range(len(state["robots"]))
            ],
            "ctrl_mode": ctrl_mode  # Can be str or int
        }
        self.data["steps"].append(step_data)

    def save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "wb") as f:
            pickle.dump(self.data, f)
