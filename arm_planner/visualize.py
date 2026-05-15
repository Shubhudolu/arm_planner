import numpy as np
import matplotlib
matplotlib.use('Agg')  # CRITICAL: Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, kinematics):
        self.kin = kinematics
    
    def plot_joint_angles(self, trajectory, save_path='output/joint_angles.png'):
        """Plot joint angles over time"""
        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            num_steps = trajectory.shape[0]
            time = np.arange(num_steps)
            for i in range(trajectory.shape[1]):
                ax.plot(time, trajectory[:, i], label=f'Joint {i+1}', linewidth=2)
            ax.set_xlabel('Timestep', fontsize=12)
            ax.set_ylabel('Joint Angle (radians)', fontsize=12)
            ax.set_title('Joint Angles Over Time', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
            plt.close('all')
        except Exception as e:
            print(f"Error in plot_joint_angles: {e}")
    
    def animate_trajectory(self, trajectory, save_path='output/trajectory.gif', fps=20, interval=50):
        """Create 3D animation of arm movement"""
        try:
            print("  Creating animation (this may take 30-60 seconds)...")
            fig = plt.figure(figsize=(14, 6))
            ax_3d = fig.add_subplot(121, projection='3d')
            ax_angles = fig.add_subplot(122)
            
            # Pre-compute all positions to speed up animation
            all_positions = []
            for q in trajectory:
                _, _, positions = self.kin.forward_kinematics(q)
                all_positions.append(positions)
            all_positions = np.array(all_positions)
            
            def update(frame):
                ax_3d.clear()
                ax_angles.clear()
                
                positions = all_positions[frame]
                positions = np.array(positions)
                
                # 3D plot
                ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                          'b-o', linewidth=3, markersize=8)
                ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                             c='red', s=100)
                
                lim = 2.5
                ax_3d.set_xlim([-lim, lim])
                ax_3d.set_ylim([-lim, lim])
                ax_3d.set_zlim([0, lim])
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                ax_3d.set_title(f'Frame {frame+1}/{len(trajectory)}')
                
                # Joint angles plot
                time = np.arange(len(trajectory))
                for i in range(trajectory.shape[1]):
                    ax_angles.plot(time, trajectory[:, i], linewidth=2, label=f'J{i+1}')
                ax_angles.axvline(frame, color='red', linestyle='--', alpha=0.7)
                ax_angles.set_xlabel('Timestep')
                ax_angles.set_ylabel('Angle')
                ax_angles.legend(fontsize=8)
                ax_angles.grid(True, alpha=0.3)
            
            anim = FuncAnimation(fig, update, frames=len(trajectory), 
                                interval=interval, repeat=True)
            
            # Try saving as GIF
            try:
                print("  Saving as GIF...")
                anim.save(save_path, writer='pillow', fps=fps)
                print(f"✓ Saved to {save_path}")
            except Exception as gif_error:
                print(f"GIF save failed ({gif_error}), trying MP4...")
                # Try MP4 instead
                try:
                    mp4_path = save_path.replace('.gif', '.mp4')
                    anim.save(mp4_path, writer='ffmpeg', fps=fps)
                    print(f"✓ Saved to {mp4_path}")
                except Exception as mp4_error:
                    print(f"MP4 also failed ({mp4_error})")
                    print("Animation saving skipped")
            
            plt.close('all')
        
        except Exception as e:
            print(f"Error in animate_trajectory: {e}")
    
    def plot_trajectory_3d(self, positions, save_path='output/trajectory_3d.png'):
        """Plot end-effector trajectory in 3D"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            positions = np.array(positions)
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=2, label='Path')
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                      c='green', s=200, marker='o', label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                      c='red', s=200, marker='X', label='Goal')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('End-Effector Trajectory')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
            plt.close('all')
        
        except Exception as e:
            print(f"Error in plot_trajectory_3d: {e}")