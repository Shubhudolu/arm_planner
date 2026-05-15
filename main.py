import numpy as np
from arm_planner.urdf import SimpleRobot
from arm_planner.kinematics import Kinematics
from arm_planner.trajectory import TrajectoryPlanner
from arm_planner.collision import CollisionDetector
from arm_planner.visualize import Visualizer

def main():
    print("=" * 60)
    print("ROBOT ARM PLANNER")
    print("=" * 60)
    
    print("\n[1] Creating robot...")
    robot = SimpleRobot()
    print(f"    Joints: {robot.num_joints()}")
    
    print("\n[2] Setting up kinematics...")
    kin = Kinematics(robot.get_joints(), link_lengths=[1.0, 0.8, 0.6])
    
    print("\n[3] Forward kinematics test...")
    q_test = np.array([0.0, 0.0, 0.0])
    pos, rot, all_pos = kin.forward_kinematics(q_test)
    print(f"    End-effector position: {pos}")
    
    print("\n[4] Inverse kinematics...")
    target = np.array([1.5, 0.5, 0.8])
    q_sol, success, errors = kin.inverse_kinematics(target, q_init=np.zeros(3))
    print(f"    Success: {success}")
    print(f"    Final error: {errors[-1]:.6f}")
    
    print("\n[5] Planning trajectory...")
    planner = TrajectoryPlanner(kin)
    q_goal = q_sol if success else np.array([0.5, 0.3, -0.2])
    traj, pos_traj = planner.plan_trajectory(
        np.array([0, 0, 0]), 
        q_goal,
        num_steps=50
    )
    print(f"    Trajectory shape: {traj.shape}")
    
    print("\n[6] Collision checking...")
    collision = CollisionDetector(kin)
    collisions = sum(1 for q in traj if collision.check_self_collision(q))
    print(f"    Collisions detected: {collisions}/50")
    
    print("\n[7] Creating visualizations...")
    viz = Visualizer(kin)
    print("    - Joint angles plot...")
    viz.plot_joint_angles(traj)
    print("    - 3D trajectory plot...")
    viz.plot_trajectory_3d(pos_traj)
    print("    - 3D animation...")
    viz.animate_trajectory(traj)
    
    print("\n" + "=" * 60)
    print("✓ DONE! Check output/ folder for results")
    print("=" * 60)

if __name__ == "__main__":
    main()