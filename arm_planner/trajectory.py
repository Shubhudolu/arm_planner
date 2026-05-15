import numpy as np

class TrajectoryPlanner:
    def __init__(self, kinematics):
        self.kin = kinematics
    
    def cubic_interpolation(self, q_start, q_goal, t):
        """Cubic interpolation for smooth motion"""
        s = 3 * (t ** 2) - 2 * (t ** 3)
        return q_start + s * (q_goal - q_start)
    
    def plan_trajectory(self, q_start, q_goal, num_steps=50, method='cubic'):
        """Plan smooth trajectory from start to goal"""
        trajectory = []
        positions = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            if method == 'cubic':
                q_t = self.cubic_interpolation(q_start, q_goal, t)
            else:
                q_t = self.cubic_interpolation(q_start, q_goal, t)
            trajectory.append(q_t)
            pos, _, _ = self.kin.forward_kinematics(q_t)
            positions.append(pos)
        return np.array(trajectory), np.array(positions)