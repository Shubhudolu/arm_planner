import xml.etree.ElementTree as ET
import numpy as np

class Joint:
    def __init__(self, name, axis, q_min, q_max):
        self.name = name
        self.axis = np.array(axis, dtype=float)
        self.q_min = q_min
        self.q_max = q_max
    
    def __repr__(self):
        return f"Joint({self.name}, axis={self.axis}, limits=[{self.q_min:.2f}, {self.q_max:.2f}])"

class SimpleRobot:
    """A simple 3-joint robot arm"""
    def __init__(self):
        self.joints = [
            Joint("joint1", [0, 0, 1], -np.pi, np.pi),
            Joint("joint2", [0, 1, 0], -np.pi/2, np.pi/2),
            Joint("joint3", [0, 1, 0], -np.pi/2, np.pi/2),
        ]
    
    def get_joints(self):
        return self.joints
    
    def num_joints(self):
        return len(self.joints)