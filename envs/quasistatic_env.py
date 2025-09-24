import pybullet as p
import pybullet_data
import yaml
import time
import numpy as np


class QuasiStaticEnv:
    def __init__(self, gui=True, sim_config_path="configs/sim_config_quasistatic.yaml", ctrl_config_path="configs/ctrl_config.yaml"):
        with open(sim_config_path, "r") as f:
            self.sim_config = yaml.safe_load(f)
        with open(ctrl_config_path, "r") as f:
            self.ctrl_config = yaml.safe_load(f)

        self.friction = {
            "plane": self.sim_config["friction"]["plane"],
            "robot": self.sim_config["friction"]["robot"]
        }
        self.gravity = [0, 0, -9.8]

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*self.gravity)

        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane, -1, 
                         lateralFriction=self.friction["plane"])

        # Create single robot
        self.robot = self._create_robot()

    def _create_robot(self):
        size = self.sim_config["robot"]["size"]  # Expecting [x, y, z] half extents
        mass = self.sim_config["robot"]["mass"]
        sys_L = self.ctrl_config["system"]["L"]

        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0, 0, 1, 1])
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        position = [0, -2 * sys_L, size[2]]  # Z = half height

        robot_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation
        )

        p.changeDynamics(robot_id, -1,
                        lateralFriction=self.friction["robot"])
        # p.changeDynamics(robot_id, -1,
        #                 lateralFriction=self.friction["robot"])
        return robot_id


    def apply_action(self, force_x, force_y, torque):
        p.applyExternalForce(self.robot, -1, [force_x, force_y, 0], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.robot, -1, [0, 0, torque], p.LINK_FRAME)

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
        
    def step(self):
        p.stepSimulation()
        time.sleep(1 / 240)

    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        return {
            "position": pos,
            "orientation_quat": orn,
            "orientation_euler": p.getEulerFromQuaternion(orn),
            "rotation_matrix": rot_matrix,
            "linear_velocity": lin_vel,
            "angular_velocity": ang_vel
        }

    def draw_body_frame(self, length=0.2):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        x_axis = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        y_axis = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
        z_axis = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        p.addUserDebugLine(pos, [pos[i] + length * x_axis[i] for i in range(3)], [1, 0, 0], 2, 5/240)
        p.addUserDebugLine(pos, [pos[i] + length * y_axis[i] for i in range(3)], [0, 1, 0], 2, 5/240)
        p.addUserDebugLine(pos, [pos[i] + length * z_axis[i] for i in range(3)], [0, 0, 1], 2, 5/240)

    def get_mass(self):
        return p.getDynamicsInfo(self.robot, -1)[0]

    def get_inertia(self):
        return p.getDynamicsInfo(self.robot, -1)[2]

    def close(self):
        p.disconnect()
