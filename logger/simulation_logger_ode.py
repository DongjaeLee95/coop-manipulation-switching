import os
import pickle
from datetime import datetime

class SimulationLogger:
    def __init__(self, log_dir="logs", prefix="simulation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_log_{timestamp}.pkl"
        self.log_path = os.path.join(log_dir, filename)
        self.data = {
            "environment": {},
            "steps": []
        }

    def log_environment(self, env):
        self.data["environment"] = {
            "gravity": env.gravity,
            "friction": env.mu,
            "target_mass": env.mass_target,
            "target_inertia": env.inertia_target
        }

    def log_step(self, time, state, actions, switching_data, ext_trajs):
        u = actions.get("u", None)
        ctrl_mode = actions.get("ctrl_mode", None)
        
        step_data = {
            "time": time,
            "robots": state["robots"],
            "target": state["target"],
            "actions": [
                {
                    "vx": actions["vx"][i],
                    "vy": actions["vy"][i],
                    "omega": actions["omega"][i],
                    "u": float(actions["u"][i]) if u is not None else None
                }
                for i in range(len(state["robots"]))
            ],
            "ctrl_mode": ctrl_mode,
            "switching": {
                "V_lyap": switching_data["V_lyap"],
                "delta": switching_data["delta"],
                "trigger": switching_data["trigger"],
                "MILP_compt_time": switching_data["MILP_compt_time"],
                "MILP_rho": switching_data["MILP_rho"]
            },
            "ext_trajs": ext_trajs
        }
        self.data["steps"].append(step_data)

    def save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "wb") as f:
            pickle.dump(self.data, f)
