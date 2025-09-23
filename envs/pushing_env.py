import pybullet as p
import pybullet_data
import yaml
import os
import time


class PushingEnv:
    def __init__(self, gui=True, config_path="configs/sim_config.yaml"):
        """Initialize simulation environment"""

        # Load YAML configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Connect to PyBullet
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load plane
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=self.config["friction"]["plane"])

        # Load target box
        target_size = self.config["target"]["size"]
        target_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=target_size)
        target_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=target_size, rgbaColor=[1, 0, 0, 1])

        self.target_box = p.createMultiBody(
            baseMass=self.config["mass"]["target"],
            baseCollisionShapeIndex=target_collision,
            baseVisualShapeIndex=target_visual,
            basePosition=[1.5, 0, target_size[2]]
        )

        p.changeDynamics(self.target_box, -1, lateralFriction=self.config["friction"]["target"])

        # Initialize robots
        self.robots = []
        self.sliders = []

        robot_radius = self.config["robot"]["radius"]
        robot_height = self.config["robot"]["height"]

        robot_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=robot_radius, height=robot_height)
        robot_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=robot_radius, length=robot_height, rgbaColor=[0, 0, 1, 1])
        robot_orientation = p.getQuaternionFromEuler([3.14, 0, 0])  # Cylinder stands upright

        # Robot positions — can be expanded
        positions = [
            [0, 0, robot_height/2]  # Single robot at center
        ]

        for i, pos in enumerate(positions):
            robot_id = p.createMultiBody(
                baseMass=self.config["mass"]["robot"],
                baseCollisionShapeIndex=robot_collision,
                baseVisualShapeIndex=robot_visual,
                basePosition=pos,
                baseOrientation=robot_orientation
            )
            self.robots.append(robot_id)

            # Slider for manual force input (X direction)
            slider = p.addUserDebugParameter(f"robot{i+1}_force", -100, 100, 0)
            self.sliders.append(slider)

            # Set robot friction
            p.changeDynamics(robot_id, -1, lateralFriction=self.config["friction"]["robot"])

    def step_manual(self):
        """Read slider values and apply forces manually (for GUI control)"""
        for robot, slider in zip(self.robots, self.sliders):
            force = p.readUserDebugParameter(slider)
            p.applyExternalForce(
                robot, -1,
                [force, 0, 0],  # force in X-direction
                [0, 0, 0],
                p.LINK_FRAME
            )

        p.stepSimulation()
        time.sleep(1 / 240)

    def step(self):
        """Advance simulation one step (for controller use)"""
        p.stepSimulation()
        time.sleep(1 / 240)

    def apply_actions(self, forces):
        """Apply list of external forces to robots (controller input)"""
        for robot, force in zip(self.robots, forces):
            p.applyExternalForce(
                robot, -1,
                [force, 0, 0],  # force in X-direction
                [0, 0, 0],
                p.LINK_FRAME
            )

    def get_state(self):
        """Return full simulation state for controller"""
        state = {
            "robots": [],
            "target": {}
        }

        # Get robot states
        for robot in self.robots:
            pos, orn = p.getBasePositionAndOrientation(robot)
            lin_vel, ang_vel = p.getBaseVelocity(robot)
            state["robots"].append({
                "position": pos,
                "orientation_quat": orn,
                "orientation_euler": p.getEulerFromQuaternion(orn),
                "linear_velocity": lin_vel,
                "angular_velocity": ang_vel
            })

        # Get target box state
        pos, orn = p.getBasePositionAndOrientation(self.target_box)
        lin_vel, ang_vel = p.getBaseVelocity(self.target_box)
        state["target"] = {
            "position": pos,
            "orientation_quat": orn,
            "orientation_euler": p.getEulerFromQuaternion(orn),
            "linear_velocity": lin_vel,
            "angular_velocity": ang_vel
        }

        return state

    def close(self):
        """Disconnect from simulation"""
        p.disconnect()
