import os
from typing import List, Union
from copy import deepcopy

import math


import PIL
import numpy as np
from einops import repeat

import torch
import torch.nn.functional as F
import torchvision.transforms as tvf


from diffusers.utils import export_to_gif

import sys
sys.path.append("./extern/CUT3R")
from extern.CUT3R.surfel_inference import run_inference_from_pil
from extern.CUT3R.add_ckpt_path import add_path_to_dust3r
from extern.CUT3R.src.dust3r.model import ARCroco3DStereo

from modeling import VMemWrapper, VMemModel, VMemModelParams
from modeling.modules.autoencoder import AutoEncoder
from modeling.sampling import DDPMDiscretization, DiscreteDenoiser, create_samplers
from modeling.modules.conditioner import CLIPConditioner
from utils import (encode_vae_image, 
                   encode_image, 
                   visualize_depth, 
                   visualize_surfels, 
                   tensor_to_pil,
                   Octree, 
                   Surfel,
                   get_plucker_coordinates,
                   do_sample,
                   average_camera_pose)



ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



class VMemPipeline:
    def __init__(self, config, device="cpu", dtype=torch.float32):
        self.config = config
        
        model_path = self.config.model.get("model_path", None)

        self.model = VMemModel(VMemModelParams()).to(device, dtype)
        # load from huggingface
        from huggingface_hub import hf_hub_download
        state_dict = torch.load(hf_hub_download(repo_id=model_path, filename="vmem_weights.pth"), map_location='cpu')
        state_dict = {k.replace("module.", "") if "module." in k else k: v for k, v in state_dict.items()}
                
            
        self.model.load_state_dict(state_dict, strict=True)
     
        
        self.model_wrapper = VMemWrapper(self.model)
        self.model_wrapper.eval()

        
        self.vae = AutoEncoder(chunk_size=1).to(device, dtype)
        self.vae.eval()
        self.image_encoder = CLIPConditioner().to(device, dtype)
        self.image_encoder.eval()
        
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(discretization=self.discretization, num_idx=1000, device=device)
        self.sampler = create_samplers(guider_types=config.model.guider_types,
                                discretization=self.discretization,
                                num_frames=config.model.num_frames,
                                num_steps=config.model.inference_num_steps,
                                cfg_min=config.model.cfg_min,
                                device=device)

        
                
        self.dtype = dtype
        self.device = device
        

        surfel_model_path = hf_hub_download(repo_id=self.config.surfel.model_path, filename="cut3r_512_dpt_4_64.pth")
        print(f"Loading model from {surfel_model_path}...")
        add_path_to_dust3r(surfel_model_path)
        self.surfel_model = ARCroco3DStereo.from_pretrained(surfel_model_path).to(device)
        self.surfel_model.eval()
        
        
        # Import CUT3R scene alignment module
        from extern.CUT3R.cloud_opt.dust3r_opt import global_aligner, GlobalAlignerMode
        self.GlobalAlignerMode = GlobalAlignerMode
        self.global_aligner = global_aligner
            
            

  

            
        self.use_non_maximum_suppression = self.config.model.use_non_maximum_suppression
        
        self.context_num_frames = self.config.model.context_num_frames
        self.target_num_frames = self.config.model.target_num_frames
        
        self.original_height = self.config.model.original_height
        self.original_width = self.config.model.original_width
        self.height = self.config.model.height
        self.width = self.config.model.width
        
        self.w_ratio = self.width / self.original_width
        self.h_ratio = self.height / self.original_height
        
        self.camera_scale = self.config.model.camera_scale
        
        self.latents = []
        self.encoder_embeddings = []
        self.poses = []
        self.Ks = []
        self.surfel_Ks = []
        self.surfels = []

        self.surfel_depths = []
        self.surfel_to_timestep = {}
        self.pil_frames = []
        self.visualize_dir = self.config.model.samples_dir
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
        
        self.global_step = 0
       

    def reset(self):
        self.rgb_vae_latents = []
        self.rgb_encoder_embeddings = []
        self.poses = []
        self.focal_lengths = []
        self.surfels = []
        self.surfel_Ks = []
        self.surfel_depths = []
        self.Ks = []
        self.surfel_to_timestep = {}
        self.all_pil_frames = []
        self.global_step = 0

    
    def initialize(self, image, c2w, K):
        """
        Initialize the pipeline with a single image and camera parameters.
        This method sets up internal state without generating additional frames.
        
        Args:
            image: Tensor of input image [1, C, H, W]
            c2w: Camera-to-world matrix (4x4)
            K: Camera intrinsic matrix
            
        Returns:
            PIL image of the initial frame
        """
        # Reset internal state
        self.reset()
        
        # Process the image
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            # Convert to tensor if it's not already (fallback)
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device, self.dtype)
        
        # Encode the image to VAE latents
        self.latents = [encode_vae_image(image_tensor, self.vae, self.device, self.dtype).detach().cpu().numpy()[0]]
        
        # Encode the image embeddings for the image_encoder
        self.encoder_embeddings = [encode_image(image_tensor, self.image_encoder, self.device, self.dtype).detach().cpu().numpy()[0]]
        
        # Store camera pose and intrinsics
        self.c2ws = [c2w]
        self.Ks = [K]
        
        # Convert to PIL and store
        pil_frame = tensor_to_pil(image_tensor)
        self.pil_frames = [pil_frame]
        

        
        return pil_frame
    
    def geodesic_distance(self,
                        camera_pose1,
                        camera_pose2,
                        weight_translation=1,):
        """
        Computes the geodesic distance between two camera poses in SE(3).
        
        Parameters:
            extrinsic1 (torch.Tensor): 4x4 extrinsic matrix of the first pose.
            extrinsic2 (torch.Tensor): 4x4 extrinsic matrix of the second pose.

        Returns:
            float: Geodesic distance between the two poses.
        """
        # Extract the rotation and translation components
        R1 = camera_pose1[:3, :3]
        t1 = camera_pose1[:3, 3]
        R2 = camera_pose2[:3, :3]
        t2 = camera_pose2[:3, 3]

        # Compute the translation distance (Euclidean distance)
        translation_distance = torch.norm(t1 - t2)
        
        # Compute the relative rotation matrix
        R_relative = torch.matmul(R1.T, R2)
        
        # Compute the angular distance from the trace of the relative rotation matrix
        trace_value = torch.trace(R_relative)
        # Clamp the trace value to avoid numerical issues
        trace_value = torch.clamp(trace_value, -1.0, 3.0)
        angular_distance = torch.acos((trace_value - 1) / 2)
        
        # Combine the two distances
        geodesic_dist = translation_distance*weight_translation + angular_distance
            
        return geodesic_dist
    
    def render_surfels_to_image(
        self,
        surfels,
        poses,
        focal_lengths,
        principal_points,
        image_width,
        image_height,
        disk_resolution=16
    ):
        """
        Renders oriented surfels into a 2D RGB image with a simple z-buffer.
        Each surfel is treated as a 2D disk in 3D, oriented by its normal.
        The disk is approximated by a polygon of 'disk_resolution' segments.

        Args:
            surfels (list): List of Surfel objects, each having:
                - position: (x, y, z) in world coords
                - normal:   (nx, ny, nz)
                - radius:   float, radius in world units
            poses (torch.Tensor): Tensor of poses, shape [4, 4]
            focal_lengths (torch.Tensor): Tensor of focal lengths, shape [2]
            principal_points (torch.Tensor): Tensor of principal points, shape [2]
            image_width, image_height (int): output image size
            disk_resolution (int): number of segments for approximating each disk

        Returns:
            Dictionary containing:
            - depth: depth map
            - surfel_index_map: map of surfel indices
            - cos_value_map: map of cosine values between view and normal directions
        """
        if isinstance(focal_lengths, torch.Tensor):
            focal_lengths = focal_lengths.detach().cpu().numpy()
        if isinstance(principal_points, torch.Tensor):
            principal_points = principal_points.detach().cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.detach().cpu().numpy()

        # Initialize buffers
        surfel_index_map = np.full((image_height, image_width), -1, dtype=np.int32)
        z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)
        cos_buffer = np.zeros((image_height, image_width), dtype=np.float32)

        # Unpack camera parameters
        fx, fy, cx, cy = focal_lengths[0], focal_lengths[1], principal_points[0], principal_points[1]
        R = poses[0:3, 0:3]
        t = poses[0:3, 3]
        
        # Compute view frustum planes in world space
        # We'll use 6 planes: near, far, left, right, top, bottom
        near_z = 0.1  # Near plane distance
        far_z = 1000.0  # Far plane distance
        
        # Convert all surfel positions to camera space at once for efficient culling
        positions = np.array([s.position for s in surfels])
        positions_h = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        
        # Compute camera matrix
        extrinsics = np.zeros((4, 4))
        extrinsics[0:3, 0:3] = np.linalg.inv(R)
        extrinsics[0:3, 3] = -np.linalg.inv(R) @ t
        extrinsics[3, 3] = 1
        
        # Transform all points to camera space at once
        cam_points = (extrinsics @ positions_h.T).T
        cam_points = cam_points[:, :3] / cam_points[:, 3:]
        
        # Compute view frustum culling mask
        in_front = cam_points[:, 2] > near_z
        behind_far = cam_points[:, 2] < far_z
        
        # Project points to get screen coordinates
        screen_x = fx * (cam_points[:, 0] / cam_points[:, 2]) + cx
        screen_y = fy * (cam_points[:, 1] / cam_points[:, 2]) + cy
        
        # Check which points are within screen bounds (with some margin for surfel radius)
        margin = 50  # Margin in pixels to account for surfel radius
        in_screen_x = (screen_x >= -margin) & (screen_x < image_width + margin)
        in_screen_y = (screen_y >= -margin) & (screen_y < image_height + margin)
        
        # Combine all culling masks
        visible_mask = in_front & behind_far & in_screen_x & in_screen_y
        visible_indices = np.where(visible_mask)[0]

        def point_in_polygon_2d(px, py, polygon):
            """Fast point-in-polygon test using ray casting"""
            inside = False
            n = len(polygon)
            j = n - 1
            for i in range(n):
                if (((polygon[i][1] > py) != (polygon[j][1] > py)) and
                    (px < (polygon[j][0] - polygon[i][0]) * (py - polygon[i][1]) /
                     (polygon[j][1] - polygon[i][1] + 1e-15) + polygon[i][0])):
                    inside = not inside
                j = i
            return inside

        # Pre-compute angle samples for circle approximation
        angles = np.linspace(0, 2*math.pi, disk_resolution, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Process only visible surfels
        for idx in visible_indices:
            surfel = surfels[idx]
            px, py, pz = surfel.position
            nx, ny, nz = surfel.normal
            radius = surfel.radius

            # Skip degenerate normals
            normal = np.array([nx, ny, nz], dtype=float)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal /= norm_len

            # Compute view direction and cosine value
            point_direction = (px, py, pz) - t
            point_direction = point_direction / np.linalg.norm(point_direction)
            cos_value = np.dot(point_direction, normal)

            # Skip backfaces
            if cos_value < 0:
                continue

            # Build local coordinate frame
            up = np.array([0, 0, 1], dtype=float)
            if abs(np.dot(normal, up)) > 0.9:
                up = np.array([0, 1, 0], dtype=float)
            xAxis = np.cross(normal, up)
            xAxis /= np.linalg.norm(xAxis)
            yAxis = np.cross(normal, xAxis)
            yAxis /= np.linalg.norm(yAxis)

            # Generate circle points efficiently
            offsets = radius * (cos_angles[:, None] * xAxis + sin_angles[:, None] * yAxis)
            circle_points = positions[idx] + offsets

            # Project all circle points at once
            circle_points_h = np.concatenate([circle_points, np.ones((len(circle_points), 1))], axis=1)
            cam_circle = (extrinsics @ circle_points_h.T).T
            depths = cam_circle[:, 2]
            valid_mask = depths > 0
            if not np.any(valid_mask):
                continue

            screen_points = np.zeros((len(circle_points), 2))
            screen_points[:, 0] = fx * (cam_circle[:, 0] / depths) + cx
            screen_points[:, 1] = fy * (cam_circle[:, 1] / depths) + cy
            
            # Get bounding box
            valid_points = screen_points[valid_mask]
            if len(valid_points) < 3:
                continue

            min_x = max(0, int(np.floor(np.min(valid_points[:, 0]))))
            max_x = min(image_width - 1, int(np.ceil(np.max(valid_points[:, 0]))))
            min_y = max(0, int(np.floor(np.min(valid_points[:, 1]))))
            max_y = min(image_height - 1, int(np.ceil(np.max(valid_points[:, 1]))))

            # Average depth for z-buffer
            avg_depth = float(np.mean(depths[valid_mask]))

            # Rasterize polygon
            for py_ in range(min_y, max_y + 1):
                for px_ in range(min_x, max_x + 1):
                    if point_in_polygon_2d(px_, py_, valid_points):
                        if avg_depth < z_buffer[py_, px_]:
                            z_buffer[py_, px_] = avg_depth
                            surfel_index_map[py_, px_] = idx
                            cos_buffer[py_, px_] = cos_value

        # Clean up depth buffer
        depth = z_buffer
        depth[depth == np.inf] = 0

        return {
            "depth": depth,
            "surfel_index_map": surfel_index_map,
            "cos_value_map": cos_buffer
        }
    
    def get_frame_distribution(self,
                               n, 
                               ratios):
        """
        Given:
        - an integer n,
        - a list of k ratios whose sum is 1 (k <= n),
        return a list of k integers [x1, x2, ..., xk],
        such that each xi >= 1, sum(xi) = n, and
        the xi are as proportional to ratios as possible.
        """
        k = len(ratios)
        if k > n:
            # set the top n ratios to 1
            result = [0] * k
            sort_indices = np.argsort(ratios)[::-1]
            for sort_index in sort_indices[:n]:
                result[sort_index] = 1
            return result

        # 1. Reserve 1 for each ratio
        result = [1] * k

        # 2. Distribute the leftover among the k ratios proportionally
        leftover = n - k
        if leftover == 0:
            # If n == k, each ratio just gets 1
            return result

        # Compute products for leftover distribution
        products = [r * leftover for r in ratios]
        floored = [int(p // 1) for p in products]  # floor of each product

        sum_floors = sum(floored)
        leftover2 = leftover - sum_floors  # how many units still to distribute

        # Add the floored part to the result
        for i in range(k):
            result[i] += floored[i]

        # Sort by the fractional remainder, descending
        remainders = [(p - f, i) for i, (p, f) in enumerate(zip(products, floored))]
        remainders.sort(key=lambda x: x[0], reverse=True)

        # Distribute the leftover2 among the largest fractional remainders
        for j in range(leftover2):
            _, idx = remainders[j]
            result[idx] = 1

        return result
    
    def process_retrieved_spatial_information(self, retrieved_spatial_information):
        
        timestep_count = {} 
  
        surfel_index_map = retrieved_spatial_information["surfel_index_map"]
        cos_value_map = retrieved_spatial_information["cos_value_map"]
        depth_map = retrieved_spatial_information["depth"]
        filtered_cos_value = cos_value_map[surfel_index_map >= 0]
        filtered_surfel_index = surfel_index_map[surfel_index_map >= 0]
        filtered_depth = depth_map[surfel_index_map >= 0]
        assert len(filtered_cos_value) == len(filtered_surfel_index), "filtered_cos_value and filtered_surfel_index should have the same length"
        for j in range(len(filtered_surfel_index)):
            cos_value = filtered_cos_value[j]
            depth_value = filtered_depth[j]
            if cos_value < 0:
                continue
            surfel_index = filtered_surfel_index[j]
            timesteps = self.surfel_to_timestep[surfel_index]
   
            for timestep in timesteps:
                
                if timestep not in timestep_count:
                    timestep_count[timestep] = cos_value/(1+depth_value)
                timestep_count[timestep] += cos_value/(1+depth_value)
            


        timestep_count_values = np.array(list(timestep_count.values()))
        timestep_count_ratios = timestep_count_values / np.sum(timestep_count_values)
        timestep_weights = {k: timestep_count_ratios[i] for i, k in enumerate(timestep_count)}
        num_retrieved_frames = min(self.config.model.context_num_frames+10, len(timestep_weights))
        frame_count = self.get_frame_distribution(num_retrieved_frames, list(timestep_weights.values())) # hard code
        frame_count = {k: int(v) for k, v in zip(timestep_count.keys(), frame_count)}
        
        # sort timestep_weights and frame_distribution by timestep without 
        timestep_weights = sorted(timestep_weights.items(), key=lambda x: x[0])
        frame_count = sorted(frame_count.items(), key=lambda x: x[0])
        
        

        return timestep_weights, frame_count
    
    
    def get_context_info(self, target_c2ws, use_non_maximum_suppression=None):
        """Get context information for novel view synthesis.
        
        Args:
            target_c2ws: Target camera-to-world matrices
            Ks: Camera intrinsic matrices
            current_timestep: Current timestep (used in temporal mode)
            
        Returns:
            Dictionary containing context information for the target view
        """
        # Function to prepare context tensors from indices
        def prepare_context_data(indices):
            c2ws = [self.c2ws[i] for i in indices]
            latents = [torch.from_numpy(self.latents[i]).to(self.device, self.dtype) for i in indices]
            embeddings = [torch.from_numpy(self.encoder_embeddings[i]).to(self.device, self.dtype) for i in indices]
            intrinsics = [self.Ks[i] for i in indices]
            return c2ws, latents, embeddings, intrinsics, indices
        
        # if self.temporal_only:
        #     # Select frames based on timesteps (temporal mode)
        #     context_time_indices = [len(self.c2ws) - 1 - i for i in range(self.config.model.context_num_frames) if len(self.c2ws) - 1 - i >= 0]
        #     context_data = prepare_context_data(context_time_indices)
        
        # elif not self.use_surfel:
        #     # Select frames based on camera pose distance with NMS
        #     average_c2w = average_camera_pose(target_c2ws)
        #     distances = torch.stack([self.geodesic_distance(torch.from_numpy(average_c2w).to(self.device, self.dtype), torch.from_numpy(np.array(c2w)).to(self.device, self.dtype), weight_translation=self.config.model.translation_distance_weight) 
        #                  for c2w in self.c2ws])
            
        #     # Sort frames by distance (closest to target first)
        #     sorted_indices = torch.argsort(distances)
        #     max_frames = min(self.config.model.context_num_frames, len(distances), len(self.latents))
            
        #     # Apply non-maximum suppression to select diverse frames
        #     is_first_step = len(self.pil_frames) <= 1
        #     is_second_step = len(self.pil_frames) == 5
        #     min_required_frames = 1 if is_first_step else max_frames
            
        #     # Adaptively determine initial threshold based on camera pose distribution
        #     if use_non_maximum_suppression is None:
        #         use_non_maximum_suppression = self.use_non_maximum_suppression
                
        #     if use_non_maximum_suppression:
  
        #         if is_second_step:
        #             # Calculate pairwise distances between existing frames
        #             pairwise_distances = []
        #             for i in range(len(self.c2ws)):
        #                 for j in range(i+1, len(self.c2ws)):
        #                     sim = self.geodesic_distance(
        #                         torch.from_numpy(np.array(self.c2ws[i])).to(self.device, self.dtype),
        #                         torch.from_numpy(np.array(self.c2ws[j])).to(self.device, self.dtype),
        #                         weight_translation=self.config.model.translation_distance_weight
        #                     )
        #                     pairwise_distances.append(sim.item())
                    
        #             if pairwise_distances:
        #                 # Sort distances and take percentile as threshold
        #                 pairwise_distances.sort()
        #                 percentile_idx = int(len(pairwise_distances) * 0.5)  # 25th percentile
        #                 self.initial_threshold = pairwise_distances[percentile_idx]
                        
        #                 # Ensure threshold is within reasonable bounds
        #                 # initial_threshold = max(0.00, min(0.001, initial_threshold))
        #             else:
        #                 self.initial_threshold = 0.001
        #         elif is_first_step:
        #             # Default threshold for first frame
        #             self.initial_threshold = 1e8
        #     else:
        #         self.initial_threshold = 1e8
                
        
            
        #     selected_indices = []
            
        #     # Try with increasingly relaxed thresholds until we get enough frames
        #     current_threshold = self.initial_threshold
        #     while len(selected_indices) < min_required_frames and current_threshold <= 1.0:
        #         # Reset selection with new threshold
        #         selected_indices = []
                
        #         # Always start with the closest pose
        #         selected_indices.append(sorted_indices[0])
                
        #         # Try to add each subsequent pose in order of distance
        #         for idx in sorted_indices[1:]:
        #             if len(selected_indices) >= max_frames:
        #                 break
                        
        #             # Check if this candidate is sufficiently different from all selected frames
        #             is_too_similar = False
        #             for selected_idx in selected_indices:
        #                 similarity = self.geodesic_distance(
        #                     torch.from_numpy(np.array(self.c2ws[idx])).to(self.device, self.dtype),
        #                     torch.from_numpy(np.array(self.c2ws[selected_idx])).to(self.device, self.dtype),
        #                     weight_translation=self.config.model.translation_distance_weight
        #                 )
        #                 if similarity < current_threshold:
        #                     is_too_similar = True
        #                     break
                            
        #             # Add to selected frames if not too similar to any existing selection
        #             if not is_too_similar:
        #                 selected_indices.append(idx)
                
        #         # If we still don't have enough frames, relax the threshold and try again
        #         if len(selected_indices) < min_required_frames:
        #             current_threshold *= 1.2
        #         else:
        #             break
            
        #     # If we still don't have enough frames, just take the top frames by distance
        #     if len(selected_indices) < min_required_frames:
        #         available_indices = []
        #         for idx in sorted_indices:
        #             if idx not in selected_indices:
        #                 available_indices.append(idx)
        #         selected_indices.extend(available_indices[:min_required_frames-len(selected_indices)])
            
        #     # Convert to tensor and maintain original order (don't reverse)
        #     context_time_indices = torch.tensor(selected_indices, device=distances.device)
        #     context_data = prepare_context_data(context_time_indices)
        
        # else:
        if len(self.pil_frames) == 1:
            context_time_indices = [0]
        else:
            # get the average camera pose
            average_c2w = average_camera_pose(target_c2ws[-self.config.model.context_num_frames//4:])
            transformed_average_c2w = self.get_transformed_c2ws(average_c2w)
            target_K = np.mean(self.surfel_Ks, axis=0)
            # Select frames using surfel-based relevance
            retrieved_info = self.render_surfels_to_image(
                self.surfels,
                transformed_average_c2w,
                [target_K*0.65] * 2,
                principal_points=(int(self.config.surfel.width/2), int(self.config.surfel.height/2)),
                image_width=int(self.config.surfel.width),
                image_height=int(self.config.surfel.height)
            )
            _, frame_count = self.process_retrieved_spatial_information(retrieved_info)
            if self.config.inference.visualize:
                visualize_depth(retrieved_info["depth"],
                                visualization_dir=self.visualize_dir, 
                                file_name=f"retrieved_depth_surfels.png",
                                size=(self.width, self.height))
            
            
            # Build candidate frames based on relevance count
            candidates = []
            for frame, count in frame_count:
                candidates.extend([frame] * count)
                indices_to_frame = {
                    i: frame for i, frame in enumerate(candidates)
                }
                
            # Sort candidates by distance to target view
            distances = [self.geodesic_distance(torch.from_numpy(average_c2w).to(self.device, self.dtype), 
                                                torch.from_numpy(self.c2ws[frame]).to(self.device, self.dtype), 
                                                weight_translation=self.config.model.translation_distance_weight).item() 
                        for frame in candidates]
            
            sorted_indices = torch.argsort(torch.tensor(distances))
            sorted_frames = [indices_to_frame[int(i.item())] for i in sorted_indices]
            max_frames = min(self.config.model.context_num_frames, len(candidates), len(self.latents))
            

            is_second_step = len(self.pil_frames) == 5
    
            
            # Adaptively determine initial threshold based on camera pose distribution
            if use_non_maximum_suppression is None:
                use_non_maximum_suppression = self.use_non_maximum_suppression
                
            if use_non_maximum_suppression:
                if is_second_step:
                    # Calculate pairwise distances between existing frames
                    pairwise_distances = []
                    for i in range(len(self.c2ws)):
                        for j in range(i+1, len(self.c2ws)):
                            sim = self.geodesic_distance(
                                torch.from_numpy(np.array(self.c2ws[i])).to(self.device, self.dtype),
                                torch.from_numpy(np.array(self.c2ws[j])).to(self.device, self.dtype),
                                weight_translation=self.config.model.translation_distance_weight
                            )
                            pairwise_distances.append(sim.item())
                    
                    if pairwise_distances:
                        # Sort distances and take percentile as threshold
                        pairwise_distances.sort()
                        percentile_idx = int(len(pairwise_distances) * 0.5)  # 25th percentile
                        self.initial_threshold = pairwise_distances[percentile_idx]
                    else:
                        self.initial_threshold = 1

            
                
            else:
                self.initial_threshold = 1e8
            
            selected_indices = []
            current_threshold = self.initial_threshold
            
            # Always start with the closest pose
            selected_indices.append(sorted_frames[0])
            if not use_non_maximum_suppression:
                selected_indices.append(len(self.c2ws) - 1)
            
            # Try with increasingly relaxed thresholds until we get enough frames
            while len(selected_indices) < max_frames and current_threshold >= 1e-5 and use_non_maximum_suppression:
                # Try to add each subsequent pose in order of distance
                for idx in sorted_frames[1:]:
                    if len(selected_indices) >= max_frames:
                        break
                        
                    # Check if this candidate is sufficiently different from all selected frames
                    is_too_similar = False
                    for selected_idx in selected_indices:
                        similarity = self.geodesic_distance(
                            torch.from_numpy(np.array(self.c2ws[idx])).to(self.device, self.dtype),
                            torch.from_numpy(np.array(self.c2ws[selected_idx])).to(self.device, self.dtype),
                            weight_translation=self.config.model.translation_distance_weight
                        )
                        if similarity < current_threshold:
                            is_too_similar = True
                            break
                            
                    # Add to selected frames if not too similar to any existing selection
                    if not is_too_similar:
                        selected_indices.append(idx)
                
                # If we still don't have enough frames, relax the threshold and try again
                if len(selected_indices) < max_frames:
                    current_threshold /= 1.2
                else:
                    break
            
            # If we still don't have enough frames, just take the top frames by distance
            if len(selected_indices) < max_frames:
                available_indices = []
                for idx in sorted_frames:
                    if idx not in selected_indices:
                        available_indices.append(idx)
                selected_indices.extend(available_indices[:max_frames-len(selected_indices)])
            
            # Convert to tensor and maintain original order (don't reverse)
            context_time_indices = torch.from_numpy(np.array(selected_indices))
        context_data = prepare_context_data(context_time_indices)
            
        (context_c2ws, context_latents, context_encoder_embeddings, context_Ks, context_time_indices) = context_data

            
        return {
            "context_c2ws": torch.from_numpy(np.array(context_c2ws)).to(self.device, self.dtype),
            "context_latents": torch.stack(context_latents).to(self.device, self.dtype),
            "context_encoder_embeddings": torch.stack(context_encoder_embeddings).to(self.device, self.dtype),
            "context_Ks": torch.from_numpy(np.array(context_Ks)).to(self.device, self.dtype),
            "context_time_indices": context_time_indices,
        }


        

    def merge_surfels(
        self,
        new_surfels: list,
        current_timestep: str,
        existing_surfels: list,
        existing_surfel_to_timestep: dict,
        position_threshold: Union[float, None] = None,  # Now optional
        normal_threshold: float = 0.7,
        max_points_per_node: int = 10 
    ):

        assert len(existing_surfels) == len(existing_surfel_to_timestep), (
            "existing_surfels and existing_surfel_to_timestep should have the same length"
        )
        
        # Automatically calculate position threshold if not provided
        if position_threshold is None:
            # Calculate average radius from both new and existing surfels
            all_radii = np.array([s.radius for s in existing_surfels + new_surfels])
            if len(all_radii) > 0:
                # Use mean radius as base threshold with a scaling factor
                mean_radius = np.mean(all_radii)
                std_radius = np.std(all_radii)
                # Position threshold = mean + 0.5 * std to account for variance
                position_threshold = mean_radius + 0.5 * std_radius
            else:
                # Fallback to default if no surfels available
                position_threshold = 0.025

        positions = np.array([s.position for s in existing_surfels])  # Shape: (N, 3)
        normals = np.array([s.normal for s in existing_surfels])      # Shape: (N, 3)

        if len(positions) > 0:
            octree = Octree(positions, max_points=max_points_per_node)
        else:
            octree = None
        

        filtered_surfels = []
        
        merge_count = 0
        for new_surfel in new_surfels:
            is_merged = False
            if octree is not None:
                neighbor_indices = octree.query_ball_point(new_surfel.position, position_threshold)
            else:
                neighbor_indices = []
            
            for idx in neighbor_indices:
                if np.dot(normals[idx], new_surfel.normal) > normal_threshold:
                    if current_timestep not in existing_surfel_to_timestep[idx]:
                        existing_surfel_to_timestep[idx].append(current_timestep)
                    is_merged = True
                    merge_count += 1
                    break
            
            if not is_merged:
                filtered_surfels.append(new_surfel)
        
        print(f"merge_count: {merge_count}")
        return filtered_surfels, existing_surfel_to_timestep
    
    def pointmap_to_surfels(self,
                            pointmap: torch.Tensor,
                            focal_lengths: torch.Tensor,
                            depths: torch.Tensor,
                            confs: torch.Tensor,
                            poses: torch.Tensor, # shape: (4, 4)
                            radius_scale: float = 0.5,
                            estimate_normals: bool = True):
        """
        Vectorized version of pointmap to surfels conversion.
        All operations are performed on the specified device (self.device) until final numpy conversion.
        """
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).to(self.device)
        if isinstance(focal_lengths, np.ndarray):
            focal_lengths = torch.from_numpy(focal_lengths).to(self.device)
        if isinstance(depths, np.ndarray):
            depths = torch.from_numpy(depths).to(self.device)
        if isinstance(confs, np.ndarray):
            confs = torch.from_numpy(confs).to(self.device)
            
        # Ensure all inputs are on the correct device
        pointmap = pointmap.to(self.device)
        focal_lengths = focal_lengths.to(self.device)
        depths = depths.to(self.device)
        confs = confs.to(self.device)
        poses = poses.to(self.device)
            
        if len(focal_lengths) == 2:
            focal_lengths = torch.mean(focal_lengths, dim=0)
            
        # 1) Estimate normals
        if estimate_normals:
            normal_map = self.estimate_normal_from_pointmap(pointmap)
        else:
            normal_map = torch.zeros_like(pointmap)
            
        # Create mask for valid points
        # depth threshold is the 95 percentile of the depth map
        depth_threshold = torch.quantile(depths, 0.999)
        valid_mask = (depths <= depth_threshold) & (confs >= self.config.surfel.conf_thresh)
        
        # Get positions, normals and depths for valid points
        positions = pointmap[valid_mask]  # [N, 3]
        normals = normal_map[valid_mask]  # [N, 3]
        valid_depths = depths[valid_mask]  # [N]
        
        # Calculate view directions for all valid points at once
        camera_pos = poses[0:3, 3]
        view_directions = positions - camera_pos.unsqueeze(0)  # [N, 3]
        view_directions = F.normalize(view_directions, dim=1)  # [N, 3]
        
        # Calculate dot products between view directions and normals
        dot_products = torch.sum(view_directions * normals, dim=1)  # [N]
        
        # Flip normals where needed
        flip_mask = dot_products < 0
        normals[flip_mask] = -normals[flip_mask]
        
        # Recalculate dot products with potentially flipped normals
        dot_products = torch.abs(torch.sum(view_directions * normals, dim=1))  # [N]
        
        # Calculate adjustment values and radii
        adjustment_values = 0.2 + 0.8 * dot_products  # [N]
        radii = (radius_scale * valid_depths / focal_lengths / adjustment_values)  # [N]
        
        # Convert to numpy only at the end
        positions = positions.detach().cpu().numpy()
        normals = normals.detach().cpu().numpy()
        radii = radii.detach().cpu().numpy()
        
        # Create surfels list using list comprehension
        surfels = [Surfel(pos, norm, rad) for pos, norm, rad in zip(positions, normals, radii)]

            
        
        return surfels

    def estimate_normal_from_pointmap(self,pointmap: torch.Tensor) -> torch.Tensor:
        h, w = pointmap.shape[:2]
        device = pointmap.device  # Keep the device (CPU/GPU) consistent
        dtype = pointmap.dtype
        
        # Initialize the normal map
        normal_map = torch.zeros((h, w, 3), device=device, dtype=dtype)
        
        for y in range(h):
            for x in range(w):
                # Check if neighbors are within bounds
                if x+1 >= w or y+1 >= h:
                    continue
                
                p_center = pointmap[y, x]
                p_right  = pointmap[y, x+1]
                p_down   = pointmap[y+1, x]
                
                # Compute vectors
                v1 = p_right - p_center
                v2 = p_down - p_center
                
                v1 = v1 / torch.linalg.norm(v1)
                v2 = v2 / torch.linalg.norm(v2)
                
                # Cross product in camera coordinates
                n_c = torch.cross(v1, v2)
                # n_c *= 1e10
                
                # Compute norm of the normal vector
                norm_len = torch.linalg.norm(n_c)
                
                if norm_len < 1e-8:
                    continue
                
                # Normalize and store
                normal_map[y, x] = n_c / norm_len
        
        return normal_map

    def get_transformed_c2ws(self, c2ws=None):
        if c2ws is None:
            c2ws = self.c2ws
        c2ws_transformed = deepcopy(np.array(c2ws))
        c2ws_transformed[..., :, [1, 2]] *= -1
        return c2ws_transformed

    def construct_and_store_scene(self, 
            input_images: List[PIL.Image.Image],
            time_indices,
            niter = 1000,
            lr = 0.01,
            device = 'cuda',
            ):
        """
        Constructs a scene from input images and stores the resulting surfels.

        Args:
            input_images: List of PIL images to process
            time_indices: The time indices for each image
            niter: Number of iterations for optimization
            lr: Learning rate for optimization
            device: Device to run inference on
            only_last_frame: Whether to only process the last frame
        """
        # Flip Y and Z components of camera poses to match dataset convention
        c2ws_transformed = self.get_transformed_c2ws()
        

        scene = run_inference_from_pil(
            input_images,
            self.surfel_model,
            poses=c2ws_transformed,
            depths=torch.from_numpy(np.array(self.surfel_depths)) if len(self.surfel_depths) > 0 else None,
            lr = lr,
            niter = niter,
            visualize=self.config.inference.visualize_pointcloud,
            device=device,
        )

        # Extract outputs
        pointcloud = torch.cat(scene['point_clouds'], dim=0)
        confs = torch.cat(scene['confidences'], dim=0)
        depths = torch.cat(scene['depths'], dim=0)
        focal_lengths = scene['camera_info']['focal']
        self.surfel_Ks.extend([focal_lengths[i] for i in range(len(focal_lengths))])
        self.surfel_depths = [depths[i].detach().cpu().numpy() for i in range(len(depths))]
        # Resize pointcloud
        pointcloud = pointcloud.permute(0, 3, 1, 2)
        pointcloud = F.interpolate(
            pointcloud, 
            scale_factor=self.config.surfel.shrink_factor, 
            mode='bilinear'
        )
        pointcloud = pointcloud.permute(0, 2, 3, 1)



        depths = depths.unsqueeze(1)
        depths = F.interpolate(
            depths, 
            scale_factor=self.config.surfel.shrink_factor, 
            mode='bilinear'
        )
        depths = depths.squeeze(1)

        confs = confs.unsqueeze(1)
        confs = F.interpolate(
            confs, 
            scale_factor=self.config.surfel.shrink_factor, 
            mode='bilinear'
        )
        confs = confs.squeeze(1)
        
        # self.surfels = []
        # self.surfel_to_timestep = {}
        start_idx = 0 if len(self.surfels) == 0 else len(pointcloud) - self.config.model.target_num_frames
        end_idx = len(pointcloud)
        # for frame_idx in range(len(pointcloud)):
        # Create surfels for the current frame
        for frame_idx in range(start_idx, end_idx):
            surfels = self.pointmap_to_surfels(
                pointmap=pointcloud[frame_idx],
                focal_lengths=focal_lengths[frame_idx] * self.config.surfel.shrink_factor,
                depths=depths[frame_idx],
                confs=confs[frame_idx],
                poses=c2ws_transformed[frame_idx],
                estimate_normals=True,
                radius_scale=self.config.surfel.radius_scale,
            )

            if len(self.surfels) > 0:
                surfels, self.surfel_to_timestep = self.merge_surfels(
                    new_surfels=surfels,
                    current_timestep=frame_idx,
                    existing_surfels=self.surfels,
                    existing_surfel_to_timestep=self.surfel_to_timestep,
                    # position_threshold=self.config.surfel.merge_position_threshold,
                    normal_threshold=self.config.surfel.merge_normal_threshold
                )


            # Update timestep mapping
            num_surfels = len(surfels)
            surfel_start_index = len(self.surfels)
            for surfel_index in range(num_surfels):
                self.surfel_to_timestep[surfel_start_index + surfel_index] = [frame_idx]

            # Save surfels if configured
            # if self.config.inference.save_surfels and len(self.surfels) > 0:
            #     positions = np.array([s.position for s in surfels], dtype=np.float32)
            #     normals   = np.array([s.normal   for s in surfels], dtype=np.float32)
            #     radii     = np.array([s.radius   for s in surfels], dtype=np.float32)
            #     colors    = np.array([s.color    for s in surfels], dtype=np.float32)

            #     np.savez(f"{self.config.visualization_dir}/surfels_added.npz",
            #             positions=positions,
            #             normals=normals,
            #             radii=radii,
            #             colors=colors)
                
            #     positions = np.array([s.position for s in self.surfels], dtype=np.float32)
            #     normals   = np.array([s.normal   for s in self.surfels], dtype=np.float32)
            #     radii     = np.array([s.radius   for s in self.surfels], dtype=np.float32)
            #     colors    = np.array([s.color    for s in self.surfels], dtype=np.float32)

            #     np.savez(f"{self.config.visualization_dir}/surfels_original.npz",
            #             positions=positions,
            #             normals=normals,
            #             radii=radii,
            #             colors=colors)
            
            self.surfels.extend(surfels)
        
        if self.config.inference.visualize_surfel:
            visualize_surfels(self.surfels, draw_normals=True, normal_scale=0.0003)

 
    
    def get_translation_scaling_factor(self, c2ws):
        # camera centering
        """
        Args:
            c2ws: camera-to-world matrices, shape: (N, 4, 4)

        Returns:
            translation_scaling_factor: translation scaling factor
        """
        ref_c2ws = c2ws
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2ws[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)

        # camera normalization
        camera_dists = c2ws[:, :3, 3].clone()
        translation_scaling_factor = (
            self.camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]),
                torch.zeros(1).to(self.device, self.dtype),
                atol=1e-5,
            ).any()
            else (self.camera_scale / torch.norm(camera_dists[0]) + 0.01)
        )
        return translation_scaling_factor, c2ws
    
    
    def get_cond(self, context_latents, all_c2ws, all_Ks, translation_scaling_factor, encoder_embeddings, input_masks):
        context_encoder_embeddings = torch.mean(encoder_embeddings, dim=0)
        input_masks = input_masks.bool()
        
        # batch_size = context_latents.shape[0]
        all_c2ws[:, :, [1, 2]] *= -1
        all_w2cs = torch.linalg.inv(all_c2ws)
        all_c2ws[:, :3, 3] *= translation_scaling_factor
        all_w2cs[:, :3, 3] *= translation_scaling_factor
        num_cameras = all_w2cs.shape[0]

        
        pluckers = get_plucker_coordinates(
            extrinsics_src=all_w2cs[:1],
            extrinsics=all_w2cs,
            intrinsics=all_Ks.float().clone(),
            target_size=(context_latents.shape[-2], context_latents.shape[-1]),
        ) # [B, 3, 6, H, W]
        
        target_latents = torch.nn.functional.pad(
            torch.zeros(self.config.model.num_frames - context_latents.shape[0], *context_latents.shape[1:]), (0, 0, 0, 0, 0, 1), value=0
        ).to(self.device, self.dtype)
        context_latents = torch.nn.functional.pad(
            context_latents, (0, 0, 0, 0, 0, 1), value=1.0
        )

        c_crossattn = repeat(context_encoder_embeddings, "d -> n 1 d", n=num_cameras)
        # c_crossattn = repeat(context_encoder_embeddings, "b 1 d -> b n 1 d", n=num_cameras)
 
        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = torch.zeros((num_cameras, *context_latents.shape[1:])).to(self.device)
        c_replace[input_masks] = context_latents
        c_replace[~input_masks] = target_latents
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                repeat(
                    input_masks,
                    "n ->n 1 h w",
                    h=pluckers.shape[-2],
                    w=pluckers.shape[-1],
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [torch.zeros((num_cameras, 1, *pluckers.shape[-2:])).to(self.device), pluckers], 1
        )
        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector
        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
        }
    
        return {"c": c, 
                "uc": uc, 
                "all_c2ws": all_c2ws, 
                "all_Ks": all_Ks, 
                "input_masks": input_masks,
                "num_cameras": num_cameras}
     
    
    def _generate_frames_for_trajectory(self, c2ws_tensor, Ks_tensor, use_non_maximum_suppression=None):
        """
        Internal helper method to generate frames for a trajectory.
        
        Args:
            c2ws: List of camera-to-world matrices
            Ks: List of camera intrinsic matrices

        
        Returns:
            List of all generated PIL frames
        """

        padding_size = 0
        # Determine generation steps based on trajectory length
        generation_steps = (len(c2ws_tensor) + 1 - self.config.model.num_frames) // self.config.model.target_num_frames + 2
        
        # Generate frames in steps
        cur_start_idx = 0
        for i in range(generation_steps):
            # Calculate frame indices for this step
            if i > 0:
                cur_start_idx = cur_end_idx
            if len(self.pil_frames) == 1: # first frame
                cur_end_idx = min(cur_start_idx + self.config.model.num_frames - 1, len(c2ws_tensor))
            else:
                cur_end_idx = min(cur_start_idx + self.config.model.target_num_frames, len(c2ws_tensor))
            
            target_length = cur_end_idx - cur_start_idx
            if target_length <= 0:
                break
                
            # Handle padding for target frames if needed
            if target_length < self.config.model.target_num_frames or (len(self.pil_frames) == 1 and target_length < self.config.model.num_frames - 1):
                # Pad target_c2ws and target_Ks with the last frame
                if len(self.pil_frames) == 1: # first frame
                    padding_size = self.config.model.num_frames - 1 - target_length
                else:
                    padding_size = self.config.model.target_num_frames - target_length
                padding = torch.tile(c2ws_tensor[cur_end_idx-1:cur_end_idx], (padding_size, 1, 1))
                c2ws_tensor = torch.cat([c2ws_tensor, padding], dim=0)
                
                padding_K = torch.tile(Ks_tensor[cur_end_idx-1:cur_end_idx], (padding_size, 1, 1))
                Ks_tensor = torch.cat([Ks_tensor, padding_K], dim=0)
                
                if len(self.pil_frames) == 1:
                    cur_end_idx = cur_start_idx + self.config.model.num_frames - 1
                else:
                    cur_end_idx = cur_start_idx + self.config.model.target_num_frames
            
            target_c2ws = c2ws_tensor[cur_start_idx:cur_end_idx]
            target_Ks = Ks_tensor[cur_start_idx:cur_end_idx]
            
 
            context_info = self.get_context_info(target_c2ws, use_non_maximum_suppression)
            
            (context_c2ws, 
             context_latents, 
             context_encoder_embeddings, 
             context_Ks,
             context_time_indices) \
                 = (context_info["context_c2ws"], 
                    context_info["context_latents"], 
                    context_info["context_encoder_embeddings"], 
                    context_info["context_Ks"], 
                    context_info["context_time_indices"])
            
            # Prepare conditioning
            all_c2ws = torch.cat([context_c2ws, target_c2ws], dim=0)
            all_Ks = torch.cat([context_Ks, target_Ks], dim=0)
            translation_scaling_factor, all_c2ws = self.get_translation_scaling_factor(all_c2ws)
            input_masks = torch.cat([torch.ones(len(context_c2ws)), torch.zeros(len(target_c2ws))], dim=0).bool().to(self.device)
            cond = self.get_cond(context_latents, all_c2ws, all_Ks, translation_scaling_factor, context_encoder_embeddings, input_masks)

            # Generate samples
            samples, samples_z = do_sample(self.model_wrapper, 
                                         self.vae, 
                                         self.denoiser, 
                                         self.sampler[0],
                                         cond["c"],
                                         cond["uc"],
                                         cond["all_c2ws"],
                                         cond["all_Ks"],
                                         input_masks,
                                         H=576, W=576, C=4, F=8, T=8, 
                                         cfg=self.config.model.cfg,  
                                         verbose=True, 
                                         global_pbar=None, 
                                         return_latents=True,
                                         device=self.device)

            # Process and store generated frames
            target_num = torch.sum(~input_masks)
            target_samples = samples[~input_masks]
            target_pil_frames = [tensor_to_pil(target_samples[j]) for j in range(target_num)]
            target_encoder_embeddings = encode_image(target_samples, self.image_encoder, self.device, self.dtype)
            target_latents = samples_z[~input_masks]
            
            for j in range(target_num - padding_size if padding_size > 0 else target_num):
                self.latents.append(target_latents[j].detach().cpu().numpy())
                self.encoder_embeddings.append(target_encoder_embeddings[j].detach().cpu().numpy())
                self.Ks.append(target_Ks[j].detach().cpu().numpy())
                self.c2ws.append(target_c2ws[j].detach().cpu().numpy())
                self.pil_frames.append(target_pil_frames[j])
                
                if self.config.inference.visualize:
                    self.pil_frames[-1].save(f"{self.config.visualization_dir}/final_{len(self.pil_frames):07d}.png")
            
            # Update scene reconstruction if needed
     
            self.construct_and_store_scene(self.pil_frames, 
                                        time_indices=context_time_indices,
                                        niter=self.config.surfel.niter, 
                                        lr=self.config.surfel.lr, 
                                        device=self.device)
            self.global_step += 1
                        
            if self.config.inference.visualize:
                export_to_gif(self.pil_frames, f"{self.config.visualization_dir}/inference_all.gif")
            
        # Return all frames or just the new ones
        return self.pil_frames[-self.config.model.target_num_frames:] if len(self.pil_frames) > self.config.model.target_num_frames + 1 else self.pil_frames
    
    def generate_trajectory_frames(self, c2ws: List[np.ndarray], Ks: List[np.ndarray], use_non_maximum_suppression=None):
        """
        Generate frames for a new trajectory segment while maintaining the pipeline state.
        This allows for interactive navigation through a scene.
        
        Args:
            c2ws: List of camera-to-world matrices for the new trajectory segment
            Ks: List of camera intrinsic matrices for the new trajectory segment
            
        Returns:
            List of PIL images for the newly generated frames
        """
        c2ws_tensor = torch.from_numpy(np.array(c2ws)).to(self.device, self.dtype)
        Ks_tensor = torch.from_numpy(np.array(Ks)).to(self.device, self.dtype)
        # translation_scaling_factor, c2ws_tensor = self.get_translation_scaling_factor(c2ws_tensor)
        
        return self._generate_frames_for_trajectory(c2ws_tensor, Ks_tensor, use_non_maximum_suppression)
    
    def undo_latest_move(self):
        """
        Undo the latest move by deleting the most recent batch of camera poses, embeddings, and pil images.
        This allows stepping back in the trajectory if navigation went in an undesired direction.
        
        The method removes the last generated batch of frames (up to target_num_frames) since the pipeline
        generates multiple frames at once during each generation step.
        
        Returns:
            bool: True if successfully removed the latest frames, False if there's nothing to remove
                 (e.g., only one frame in the pipeline)
        """
        # Ensure we have more than one frame to avoid removing the initial frame
        if len(self.pil_frames) <= 1:
            print("Cannot undo: only one frame in the pipeline")
            return False
        
        # Determine how many frames to remove - up to target_num_frames
        frames_to_remove = min(self.config.model.target_num_frames, len(self.pil_frames) - 1)
        
        # Remove the latest entries from all state lists
        for _ in range(frames_to_remove):
            self.latents.pop()
            self.encoder_embeddings.pop()
            self.c2ws.pop()
            self.Ks.pop()
            self.pil_frames.pop()
            
        
        # Handle surfels if using reconstructor
        self.global_step -= frames_to_remove

        for _ in range(frames_to_remove):
            self.surfel_depths.pop()

                
            # Find surfels that belong only to the removed timesteps
            current_frame_count = len(self.pil_frames)
            removed_timesteps = list(range(current_frame_count, current_frame_count + frames_to_remove))
            surfels_to_remove = []
            
            # Loop through surfel_to_timestep and update
            updated_surfel_to_timestep = {}
            for i, timesteps in self.surfel_to_timestep.items():
                # Check if this surfel only belongs to removed frames
                if all(ts in removed_timesteps for ts in timesteps):
                    surfels_to_remove.append(i)
                else:
                    # Keep this surfel but remove the timesteps of removed frames
                    updated_timesteps = [ts for ts in timesteps if ts not in removed_timesteps]
                    updated_surfel_to_timestep[i] = updated_timesteps
            
            # Now create new surfel list without the removed ones
            updated_surfels = []
            updated_final_surfel_to_timestep = {}
            new_idx = 0
            
            for i, surfel in enumerate(self.surfels):
                if i not in surfels_to_remove:
                    updated_surfels.append(surfel)
                    updated_final_surfel_to_timestep[new_idx] = updated_surfel_to_timestep[i]
                    new_idx += 1
            
            # Update surfel data
            self.surfels = updated_surfels
            self.surfel_to_timestep = updated_final_surfel_to_timestep
            
        print(f"Successfully removed the latest {frames_to_remove} frames. {len(self.pil_frames)} frames remaining.")
        return True
        

    
    def __call__(self, image:torch.Tensor, c2ws: List[np.ndarray], Ks: List[np.ndarray]):
        """
        Process an initial image and generate frames for a trajectory.
        
        Args:
            image: Initial image tensor
            c2ws: Camera-to-world matrices for the trajectory
            Ks: Camera intrinsic matrices for the trajectory
            
        Returns:
            List of PIL images for all generated frames
        """
        # Initialize with the first frame
        c2ws_tensor = torch.from_numpy(np.array(c2ws)).to(self.device, self.dtype)
        
        Ks_tensor = torch.from_numpy(np.array(Ks)).to(self.device, self.dtype)
        
        # translation_scaling_factor, c2ws_tensor = self.get_translation_scaling_factor(c2ws_tensor)
        
        self.initialize(image, c2ws_tensor[0].detach().cpu().numpy(), Ks_tensor[0].detach().cpu().numpy())
        
        return self._generate_frames_for_trajectory(c2ws_tensor[1:], Ks_tensor[1:])
    

 