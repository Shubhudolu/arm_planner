import numpy as np

class CollisionDetector:
    def __init__(self, kinematics, link_radii=None):
        self.kin = kinematics
        num_positions = len(kinematics.joints) + 1
        if link_radii is None:
            self.link_radii = [0.1] * num_positions
        else:
            self.link_radii = link_radii
    
    def sphere_collision(self, center1, radius1, center2, radius2):
        distance = np.linalg.norm(center1 - center2)
        return distance < (radius1 + radius2)
    
    def check_self_collision(self, q):
        _, _, positions = self.kin.forward_kinematics(q)
        positions = np.array(positions)
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                if self.sphere_collision(pos_i, self.link_radii[i], pos_j, self.link_radii[j]):
                    return True
        return False