# Robot Arm Planner

Overview

A complete robotics arm simulator that implements:
- Forward Kinematics (FK) - Calculate end-effector position from joint angles
- Inverse Kinematics (IK) - Find joint angles to reach target position
- Trajectory Planning - Plan smooth paths between configurations
- Collision Detection - Detect self-collisions using bounding spheres
- 3D Visualization - Animate the arm movement and plot joint angles

Features

 3-DOF robot arm simulation  
 Numerical Jacobian computation  
 Damped least-squares IK solver  
 Cubic trajectory interpolation  
 Self-collision detection  
 3D matplotlib animations  

Files

- `arm_planner/urdf.py` - Robot configuration and URDF parsing
- `arm_planner/kinematics.py` - FK and IK implementation (Rodrigues formula, Jacobian)
- `arm_planner/trajectory.py` - Trajectory planning with cubic interpolation
- `arm_planner/collision.py` - Self-collision detection using bounding spheres
- `arm_planner/visualize.py` - 3D visualization and animation
- `main.py` - Main integration script
- `tests/test_fk.py` - Forward kinematics tests

Output

The program generates:
- `output/joint_angles.png` - Plot of joint angles over time
- `output/trajectory_3d.png` - 3D end-effector trajectory
- `output/trajectory.gif` - 3D animation of arm movement
## How to Run
### Execute
-First open the root folder arm_planner in VS code.
-Then execute main.py using the following command in powershell.
```bash(Mine was done in Windows)
python3 main.py
```
-Then execute test_fk.py inside thet tests folder using the following command
```bash(Mine was done in Windows)
python3 test_fk.py
