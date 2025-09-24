import pybullet as p
import pybullet_data
import yaml
import os
import time
import numpy as np


class PushingEnv:
    def __init__(self, gui=True, sim_config_path="configs/sim_config.yaml", ctrl_config_path="configs/ctrl_config.yaml"):
        """Initialize simulation environment"""

        # Load YAML configuration
        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
        
        with open(ctrl_config_path, "r") as f:
            self.ctrl_config = yaml.safe_load(f)

        # Connect to PyBullet
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load plane
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, lateralFriction=self.sim_config["friction"]["plane"])

        # Load target box
        # self.target_box = p.loadURDF("urdf/target_box.urdf", basePosition=[1.5, 0, 0.3])
        target_size = self.sim_config["target"]["size"]
        target_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=target_size)
        target_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=target_size, rgbaColor=[1, 0, 0, 1])
        self.target_box = p.createMultiBody(
            baseMass=self.sim_config["mass"]["target"],
            baseCollisionShapeIndex=target_collision,
            baseVisualShapeIndex=target_visual,
            basePosition=[0, 0, target_size[2]]
        )

        p.changeDynamics(self.target_box, -1, lateralFriction=self.sim_config["friction"]["target"])

        # Initialize robots
        self.robots = []
        # self.sliders = []

        robot_radius = self.sim_config["robot"]["radius"]
        robot_height = self.sim_config["robot"]["height"]

        robot_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=robot_radius, height=robot_height)
        robot_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=robot_radius, length=robot_height, rgbaColor=[0, 0, 1, 1])
        # robot_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Cylinder stands upright

        # Robot positions — can be expanded
        sys_L = self.ctrl_config["system"]["L"]
        # positions = [
        #     [-sys_L, -3*sys_L, robot_height/2], 
        #     [sys_L, -3*sys_L, robot_height/2], 
        #     [3*sys_L, -sys_L, robot_height/2], 
        #     [3*sys_L, sys_L, robot_height/2],
        #     [sys_L, 3*sys_L, robot_height/2], 
        #     [-sys_L, 3*sys_L, robot_height/2], 
        #     [-3*sys_L, sys_L, robot_height/2], 
        #     [-3*sys_L, -sys_L, robot_height/2]
        # ]
        positions = [
            [-2*sys_L, -3*sys_L, robot_height/2], 
            [2*sys_L, -3*sys_L, robot_height/2], 
            [3*sys_L, -2*sys_L, robot_height/2], 
            [3*sys_L, 2*sys_L, robot_height/2],
            [2*sys_L, 3*sys_L, robot_height/2], 
            [-2*sys_L, 3*sys_L, robot_height/2], 
            [-3*sys_L, 2*sys_L, robot_height/2], 
            [-3*sys_L, -2*sys_L, robot_height/2]
        ]
        orientations = 1.2*np.array([
            np.pi/2, np.pi/2, 
            np.pi, np.pi, 
            3*np.pi/2, 3*np.pi/2, 
            0, 0
        ])
        
        for i, pos in enumerate(positions):
            robot_orientation = p.getQuaternionFromEuler([0, 0, orientations[i]])
            robot_id = p.createMultiBody(
                baseMass=self.sim_config["mass"]["robot"],
                baseCollisionShapeIndex=robot_collision,
                baseVisualShapeIndex=robot_visual,
                basePosition=pos,
                baseOrientation=robot_orientation
            )
            self.robots.append(robot_id)

            # Slider for manual force input (X direction)
            # slider = p.addUserDebugParameter(f"robot{i+1}_force", -100, 100, 0)
            # self.sliders.append(slider)

            # Set robot friction
            p.changeDynamics(robot_id, -1, 
                             lateralFriction=self.sim_config["friction"]["robot_t"],
                             rollingFriction=self.sim_config["friction"]["robot_r"])
        

    def print_dynamics_info(self, body_id, link_index=-1):
        info = p.getDynamicsInfo(body_id, link_index)

        print("Dynamics Info:")
        print(f"  Mass: {info[0]}")
        print(f"  Lateral Friction: {info[1]}")
        print(f"  Local Inertia Diagonal (Moment of Inertia): {info[2]}")
        print(f"  Local Inertial Position: {info[3]}")
        print(f"  Local Inertial Orientation (Quaternion): {info[4]}")
        print(f"  Rolling Friction: {info[5]}")
        print(f"  Spinning Friction: {info[6]}")
        print(f"  Restitution (Bounciness): {info[7]}")
        print(f"  Contact Damping: {info[8]}")
        print(f"  Contact Stiffness: {info[9]}")
        print(f"  Body Type (0=kinematic, 1=static, 2=rigid): {info[10]}")
        print(f"  Collision Margin: {info[11]}")


    def step_manual(self):
        """Read slider values and apply forces manually (for GUI control)"""
        for robot, slider in zip(self.robots, self.sliders):
            force = p.readUserDebugParameter(slider)
            p.applyExternalForce(
                robot, -1,print
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

    def apply_actions(self, forces_x, forces_y, torques):
        """Apply list of external forces to robots (controller input)"""
        for robot, force_x, force_y, torque in zip(self.robots, forces_x, forces_y, torques):
            p.applyExternalForce(
                robot, -1,
                [force_x, force_y, 0],  # force in X,Y-direction
                [0, 0, 0],
                p.LINK_FRAME
            )
            p.applyExternalTorque(
                robot, -1,
                [0, 0, torque],  # Rotate around Z-axis
                p.LINK_FRAME  # or p.WORLD_FRAME depending on your intent
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
            rot_matrix = p.getMatrixFromQuaternion(orn)
            lin_vel, ang_vel = p.getBaseVelocity(robot)
            state["robots"].append({
                "position": pos,
                "orientation_quat": orn,
                "orientation_euler": p.getEulerFromQuaternion(orn),
                "rotation_matrix": rot_matrix,
                "linear_velocity": lin_vel,
                "angular_velocity": ang_vel
            })

        # Get target box state
        pos, orn = p.getBasePositionAndOrientation(self.target_box)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.target_box)
        state["target"] = {
            "position": pos,
            "orientation_quat": orn,
            "orientation_euler": p.getEulerFromQuaternion(orn),
            "rotation_matrix": rot_matrix,
            "linear_velocity": lin_vel,
            "angular_velocity": ang_vel
        }

        return state
    
    def draw_body_frame(self, robot, length=0.2):
        pos, orn = p.getBasePositionAndOrientation(robot)
        rot_matrix = p.getMatrixFromQuaternion(orn)

        # Extract basis vectors from rotation matrix
        x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
        z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        # Compute end points
        x_end = [pos[i] + length * x_axis[i] for i in range(3)]
        y_end = [pos[i] + length * y_axis[i] for i in range(3)]
        z_end = [pos[i] + length * z_axis[i] for i in range(3)]

        # Draw lines
        p.addUserDebugLine(pos, x_end, [1, 0, 0], lineWidth=2, lifeTime=5/240)  # Red = X
        p.addUserDebugLine(pos, y_end, [0, 1, 0], lineWidth=2, lifeTime=5/240)  # Green = Y
        p.addUserDebugLine(pos, z_end, [0, 0, 1], lineWidth=2, lifeTime=5/240)  # Blue = Z


    def close(self):
        """Disconnect from simulation"""
        p.disconnect()
