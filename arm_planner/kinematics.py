import numpy as np
from math import sin, cos

class Kinematics:
    def __init__(self, joints, link_lengths=None):
        self.joints = joints
        self.n_joints = len(joints)
        if link_lengths is None:
            self.link_lengths = [1.0] * self.n_joints
        else:
            self.link_lengths = link_lengths
    
    def rodrigues_rotation(self, axis, angle):
        """Create rotation matrix from axis-angle"""
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        R = I + sin(angle) * K + (1 - cos(angle)) * (K @ K)
        return R
    
    def forward_kinematics(self, q):
        """Compute end-effector position given joint angles q"""
        T = np.eye(4)
        positions = [np.array([0, 0, 0])]
        for i, joint in enumerate(self.joints):
            theta = q[i]
            R = self.rodrigues_rotation(joint.axis, theta)
            p = np.array([0, 0, self.link_lengths[i]])
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T_joint[:3, 3] = p
            T = T @ T_joint
            positions.append(T[:3, 3])
        return T[:3, 3], T[:3, :3], positions
    
    def compute_jacobian(self, q, delta=1e-6):
        """Compute Jacobian matrix numerically"""
        J = np.zeros((3, self.n_joints))
        pos0, _, _ = self.forward_kinematics(q)
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] += delta
            pos_plus, _, _ = self.forward_kinematics(q_plus)
            J[:, i] = (pos_plus - pos0) / delta
        return J
    
    def inverse_kinematics(self, target_position, q_init=None, max_iterations=1000, 
                           tolerance=1e-4, lambda_damp=0.01, step_size=0.1):
        """Find joint angles to reach target position"""
        if q_init is None:
            q_init = np.random.uniform(-1, 1, self.n_joints)
        q = q_init.copy().astype(float)
        error_history = []
        
        for iteration in range(max_iterations):
            current_pos, _, _ = self.forward_kinematics(q)
            error = target_position - current_pos
            error_magnitude = np.linalg.norm(error)
            error_history.append(error_magnitude)
            
            if error_magnitude < tolerance:
                return q, True, error_history
            
            J = self.compute_jacobian(q)
            JtJ = J.T @ J
            damping_matrix = (lambda_damp ** 2) * np.eye(self.n_joints)
            
            try:
                inv_term = np.linalg.inv(JtJ + damping_matrix)
                dq = inv_term @ J.T @ error
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J) @ error
            
            q = q + step_size * dq
            
            for i, joint in enumerate(self.joints):
                q[i] = np.clip(q[i], joint.q_min, joint.q_max)
        
        return q, False, error_history
    