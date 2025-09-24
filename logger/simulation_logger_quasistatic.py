import os
import pickle
from datetime import datetime

class SimulationLogger_QS:
    def __init__(self, log_dir="logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_log_QS_{timestamp}.pkl"
        self.log_path = os.path.join(log_dir, filename)
        self.data = {
            "environment": {},
            "steps": []
        }

    def log_environment(self, env):
        self.data["environment"] = {
            "gravity": env.gravity,
            "friction": env.friction,
            "robot_masses": env.get_mass(),
            "robot_inertias": env.get_inertia()
        }

    def log_step(self, time, robot_state, reference, actions):
        step_data = {
            "time": time,
            "robot": robot_state,
            "reference": reference,
            "actions": [
                {
                    "force_x": actions["forces_x"][i],
                    "force_y": actions["forces_y"][i],
                    "torque": actions["torques"][i]
                }
                for i in range(len(robot_state))
            ]
        }
        self.data["steps"].append(step_data)



    def save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "wb") as f:
            pickle.dump(self.data, f)
