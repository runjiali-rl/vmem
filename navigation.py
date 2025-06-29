import numpy as np
import torch
from PIL import Image
import argparse
import os
import json
from typing import List, Optional, Tuple
import scipy.spatial.transform as spt
from omegaconf import OmegaConf
import shutil

from modeling.pipeline import VMemPipeline
from utils import load_img_and_K, transform_img_and_K, get_default_intrinsics





class Navigator:
    """
    Navigator class for moving through a 3D scene with a virtual camera.
    Provides methods to move forward and turn left/right, generating new camera poses
    and rendering frames using VMemPipeline.
    """
    def __init__(self, pipeline: VMemPipeline, step_size: float = 0.1, num_interpolation_frames: int = 4):
        """
        Initialize the Navigator.
        
        Args:
            pipeline: The VMemPipeline used for rendering frames
            step_size: The distance to move forward with each step
            num_interpolation_frames: Number of frames to generate for each movement
        """
        self.pipeline = pipeline
        self.step_size = step_size
        self.current_pose = None
        self.current_K = None
        self.frames = []
        self.num_interpolation_frames = num_interpolation_frames
        self.pose_history = []  # Store history of camera poses
        
    def initialize(self, image, initial_pose, initial_K):
        """
        Initialize the navigator with an image and camera parameters.
        Uses the pipeline's initialize method to set up the state.
        
        Args:
            image: Initial image tensor
            initial_pose: Initial camera pose (4x4 camera-to-world matrix)
            initial_K: Initial camera intrinsics matrix
            
        Returns:
            The initial frame as PIL Image
        """
        self.current_pose = initial_pose
        self.current_K = initial_K
        
        # Use the pipeline's initialize method
        initial_frame = self.pipeline.initialize(image, initial_pose, initial_K)
        self.frames = [initial_frame]
        
        # Save the initial pose
        self.pose_history.append({
            "file_path": f"images/frame_001.png",
            "transform_matrix": initial_pose.tolist() if isinstance(initial_pose, np.ndarray) else initial_pose
        })
        
        # clean the visualization folder
        if os.path.exists("visualization"):
            shutil.rmtree("visualization")
        os.makedirs("visualization", exist_ok=True)
        
        return initial_frame
        
    def initialize_pipeline(self, image_tensor, initial_pose, initial_K):
        """
        Initialize the pipeline with the first image and camera pose.
        Deprecated: Use initialize() instead.
        
        Args:
            image_tensor: Initial image tensor [1, C, H, W]
            initial_pose: Initial camera pose (4x4 camera-to-world matrix)
            initial_K: Initial camera intrinsics matrix
            
        Returns:
            The generated frame as PIL Image
        """
        return self.initialize(image_tensor, initial_pose, initial_K)
        
    def initialize_with_image(self, image, initial_pose, initial_K):
        """
        Initialize the navigator with an image and camera parameters.
        Deprecated: Use initialize() instead.
        
        Args:
            image: Initial image tensor
            initial_pose: Initial camera pose (4x4 camera-to-world matrix)
            initial_K: Initial camera intrinsics matrix
            
        Returns:
            The generated frame as PIL Image
        """
        return self.initialize(image, initial_pose, initial_K)
    
    def _interpolate_poses(self, start_pose, end_pose, num_frames):
        """
        Interpolate between two camera poses.
        
        Args:
            start_pose: Starting camera pose (4x4 matrix)
            end_pose: Ending camera pose (4x4 matrix)
            num_frames: Number of interpolation frames to generate (including end pose)
            
        Returns:
            List of interpolated camera poses
        """
        # Extract rotation matrices
        start_R = start_pose[:3, :3]
        end_R = end_pose[:3, :3]
        
        # Extract translation vectors
        start_t = start_pose[:3, 3]
        end_t = end_pose[:3, 3]
        
        # Convert rotation matrices to quaternions for smooth interpolation
        start_quat = spt.Rotation.from_matrix(start_R).as_quat()
        end_quat = spt.Rotation.from_matrix(end_R).as_quat()
        
        # Generate interpolated poses
        interpolated_poses = []
        for i in range(num_frames):
            # Interpolation factor (0 to 1)
            t = (i + 1) / num_frames
            
            # Interpolate translation
            interp_t = (1 - t) * start_t + t * end_t
            
            # Interpolate rotation (SLERP)
            interp_quat = spt.Slerp(
                np.array([0, 1]), 
                spt.Rotation.from_quat([start_quat, end_quat])
            )(t).as_matrix()
            
            # Create interpolated pose matrix
            interp_pose = np.eye(4)
            interp_pose[:3, :3] = interp_quat
            interp_pose[:3, 3] = interp_t
            
            interpolated_poses.append(interp_pose)
            
        return interpolated_poses
        
    def move_backward(self, num_steps: int = 1) -> List[Image.Image]:
        """
        Move the camera backward along its viewing direction with smooth interpolation.
        
        Args:
            num_steps: Number of steps to move forward
            
        Returns:
            List of generated frames as PIL Images
        """
        if self.current_pose is None:
            print("Navigator not initialized. Call initialize first.")
            return None
        
        # Get the current forward direction from the camera pose
        forward_dir = self.current_pose[:3, 2]
        
        # Create the target pose
        target_pose = self.current_pose.copy()
        target_pose[:3, 3] += forward_dir * self.step_size * num_steps
        
        # Interpolate between current pose and target pose
        interpolated_poses = self._interpolate_poses(
            self.current_pose, 
            target_pose, 
            self.num_interpolation_frames
        )
        
        # Create list of intrinsics (same for all frames)
        interpolated_Ks = [self.current_K] * len(interpolated_poses)
        
        # Generate frames for interpolated poses
        new_frames = self.pipeline.generate_trajectory_frames(interpolated_poses, 
                                                              interpolated_Ks,
                                                              use_non_maximum_suppression=False)
        
        # Update the current pose to the final pose
        self.current_pose = interpolated_poses[-1]
        self.frames.extend(new_frames)
        
        # Save the final pose
        self.pose_history.append({
            "file_path": f"images/frame_{len(self.pose_history) + 1:03d}.png",
            "transform_matrix": self.current_pose.tolist() if isinstance(self.current_pose, np.ndarray) else self.current_pose
        })
        
        return new_frames
    
        
    def move_forward(self, num_steps: int = 1) -> List[Image.Image]:
        """
        Move the camera forward along its viewing direction with smooth interpolation.
        
        Args:
            num_steps: Number of steps to move forward
            
        Returns:
            List of generated frames as PIL Images
        """
        if self.current_pose is None:
            print("Navigator not initialized. Call initialize first.")
            return None
        
        # Get the current forward direction from the camera pose
        forward_dir = self.current_pose[:3, 2]
        
        # Create the target pose
        target_pose = self.current_pose.copy()
        target_pose[:3, 3] -= forward_dir * self.step_size * num_steps
        
        # Interpolate between current pose and target pose
        interpolated_poses = self._interpolate_poses(
            self.current_pose, 
            target_pose, 
            self.num_interpolation_frames
        )
        
        # Create list of intrinsics (same for all frames)
        interpolated_Ks = [self.current_K] * len(interpolated_poses)
        
        # Generate frames for interpolated poses
        new_frames = self.pipeline.generate_trajectory_frames(interpolated_poses, 
                                                              interpolated_Ks,
                                                              use_non_maximum_suppression=False)
        
        # Update the current pose to the final pose
        self.current_pose = interpolated_poses[-1]
        self.frames.extend(new_frames)
        
        # Save the final pose
        self.pose_history.append({
            "file_path": f"images/frame_{len(self.pose_history) + 1:03d}.png",
            "transform_matrix": self.current_pose.tolist() if isinstance(self.current_pose, np.ndarray) else self.current_pose
        })
        
        return new_frames
    
    def turn_left(self, degrees: float = 3) -> List[Image.Image]:
        """
        Rotate the camera left around the up vector with smooth interpolation.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            List of generated frames as PIL Images
        """
        return self._turn(degrees)
    
    def turn_right(self, degrees: float = 3) -> List[Image.Image]:
        """
        Rotate the camera right around the up vector with smooth interpolation.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            List of generated frames as PIL Images
        """
        return self._turn(-degrees)
    
    def _turn(self, degrees: float) -> List[Image.Image]:
        """
        Helper method to turn the camera by the specified angle with smooth interpolation.
        Positive angles turn left, negative angles turn right.
        
        Args:
            degrees: Rotation angle in degrees
            
        Returns:
            List of generated frames as PIL Images
        """
        if self.current_pose is None:
            print("Navigator not initialized. Call initialize first.")
            return None
        
        # Convert degrees to radians
        angle_rad = np.radians(degrees)
        
        # Create rotation matrix around the up axis (assuming Y is up)
        rotation = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ])
        
        # Apply rotation to the current pose
        position = self.current_pose[:3, 3].copy()
        rotation_matrix = self.current_pose[:3, :3].copy()
        
        # Create the target pose
        target_pose = np.eye(4)
        target_pose[:3, :3] = rotation[:3, :3] @ rotation_matrix
        target_pose[:3, 3] = position
        
        # Interpolate between current pose and target pose
        interpolated_poses = self._interpolate_poses(
            self.current_pose, 
            target_pose, 
            self.num_interpolation_frames
        )
        
        # Create list of intrinsics (same for all frames)
    
        interpolated_Ks = [self.current_K] * len(interpolated_poses)
        
        # Generate frames for interpolated poses
        new_frames = self.pipeline.generate_trajectory_frames(interpolated_poses, interpolated_Ks)
        
        # Update the current pose to the final pose
        self.current_pose = interpolated_poses[-1]
        self.frames.extend(new_frames)
        
        # Save the final pose
        self.pose_history.append({
            "file_path": f"images/frame_{len(self.pose_history) + 1:03d}.png",
            "transform_matrix": self.current_pose.tolist() if isinstance(self.current_pose, np.ndarray) else self.current_pose
        })
        
        return new_frames
    
 
    
    def navigate(self, commands: List[str]) -> List[List[Image.Image]]:
        """
        Execute a series of navigation commands and return the generated frames.
        
        Args:
            commands: List of commands ('w', 'a', 'd', 'q')
            
        Returns:
            List of lists of generated frames, one list per command
        """
        if self.current_pose is None:
            print("Navigator not initialized. Call initialize first.")
            return []
        
        all_generated_frames = []
        
        for idx, cmd in enumerate(commands):
        
            if cmd == 'w':
                frames = self.move_forward()
                if frames:
                    all_generated_frames.extend(frames)
            elif cmd == 's':
                # self.pipeline.temporal_only = True
                frames = self.move_backward()
                if frames:
                    all_generated_frames.extend(frames)
            elif cmd == 'a':
                frames = self.turn_left(4)
                if frames:
                    all_generated_frames.extend(frames)
            elif cmd == 'd':
                frames = self.turn_right(4)
                if frames:
                    all_generated_frames.extend(frames)


        
        return all_generated_frames

    def undo(self) -> bool:
        """
        Undo the last navigation step by removing the most recent frames and poses.
        Uses the pipeline's undo_latest_move method to handle the frame removal.
        
        Returns:
            bool: True if undo was successful, False otherwise
        """
        # Check if we have enough poses to undo
        if len(self.pose_history) <= 1:
            print("Cannot undo: at initial position")
            return False
            
        # Use pipeline's undo function to remove the last batch of frames
        success = self.pipeline.undo_latest_move()
        
        if success:
            # Remove the last pose from history
            self.pose_history.pop()
            
            # Set current pose to the previous pose
            prev_pose_data = self.pose_history[-1]
            self.current_pose = np.array(prev_pose_data["transform_matrix"])
            
            # Remove frames from the frames list
            frames_to_remove = min(self.pipeline.config.model.target_num_frames, len(self.frames) - 1)
            for _ in range(frames_to_remove):
                if len(self.frames) > 1:  # Keep at least the initial frame
                    self.frames.pop()
            
            print(f"Successfully undid last movement. Now at position {len(self.pose_history)}")
            return True
        
        return False

    def save_camera_poses(self, output_path):
        """
        Save the camera pose history to a JSON file in the format
        required for NeRF training.
        
        Args:
            output_path: Path to save the JSON file
        """
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Format the data as required
        transforms_data = {
            "frames": self.pose_history
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(transforms_data, f, indent=4)
        
        print(f"Camera poses saved to {output_path}")
    
   

