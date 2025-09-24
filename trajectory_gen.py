import numpy as np

class Traj_gen:
    def __init__(self, radius=0.5, omega=0.2, yaw=np.pi/2):
        self.radius = radius      # radius of the circle
        self.omega = omega        # angular speed (rad/s)
        self.yaw = yaw            # constant yaw angle

    def get_reference(self, t):
        # Position on the circle
        x = self.radius * np.cos(self.omega * t)
        y = self.radius * np.sin(self.omega * t)
        pos = np.array([[x], [y]])

        # Velocity (tangent to the circle)
        vx = -self.radius * self.omega * np.sin(self.omega * t)
        vy =  self.radius * self.omega * np.cos(self.omega * t)
        vel = np.array([[vx], [vy]])

        # Orientation and angular velocity
        yaw = self.yaw
        ang_vel = np.zeros((3, 1))  # or np.array([[0], [0], [self.omega]]) if rotating

        return {
            "position": pos,
            "velocity": vel,
            "yaw": yaw,
            "angular_velocity": ang_vel
        }
