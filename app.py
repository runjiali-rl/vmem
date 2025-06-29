from typing import List, Literal
from pathlib import Path
from functools import partial
import spaces
import gradio as gr
import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from einops import repeat
from omegaconf import OmegaConf
from modeling.pipeline import VMemPipeline
from diffusers.utils import export_to_video
from navigation import Navigator
from utils import tensor_to_pil, get_default_intrinsics, load_img_and_K, transform_img_and_K
import os
import shutil


CONFIG_PATH = "configs/inference/inference.yaml"
CONFIG = OmegaConf.load(CONFIG_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = VMemPipeline(CONFIG, DEVICE)
NAVIGATORS = []


NAVIGATION_FPS = 13
WIDTH = 576
HEIGHT = 576


def clear_visualization_directory():
    """
    Clear all contents from the visualization directory to prevent users from seeing
    each other's generated images.
    """
    viz_dir = "./visualization"
    try:
        if os.path.exists(viz_dir):
            shutil.rmtree(viz_dir)
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Cleared visualization directory: {viz_dir}")
    except Exception as e:
        print(f"Warning: Could not clear visualization directory: {e}")


IMAGE_PATHS = ['test_samples/oxford.jpg', 
               'test_samples/open_door.jpg', 
               'test_samples/living_room.jpg', 
               'test_samples/arc_de_tromphe.jpeg',
               'test_samples/changi.jpg', 
               'test_samples/jesus.jpg',]

# If no images found, create placeholders
if not IMAGE_PATHS:
    def create_placeholder_images(num_samples=5, height=HEIGHT, width=WIDTH):
        """Create placeholder images for the demo"""
        images = []
        for i in range(num_samples):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for h in range(height):
                for w in range(width):
                    img[h, w, 0] = int(255 * h / height)  # Red gradient
                    img[h, w, 1] = int(255 * w / width)   # Green gradient
                    img[h, w, 2] = int(255 * (i+1) / num_samples)  # Blue varies by image
            images.append(img)
        return images

    # Create placeholder video frames and poses
    def create_placeholder_video_and_poses(num_samples=5, num_frames=1, height=HEIGHT, width=WIDTH):
        """Create placeholder videos and poses for the demo"""
        videos = []
        poses = []
        
        for i in range(num_samples):
            # Create a simple video (just one frame initially for each sample)
            frames = []
            for j in range(num_frames):
                # Create a gradient frame
                img = np.zeros((height, width, 3), dtype=np.uint8)
                for h in range(height):
                    for w in range(width):
                        img[h, w, 0] = int(255 * h / height)  # Red gradient
                        img[h, w, 1] = int(255 * w / width)   # Green gradient
                        img[h, w, 2] = int(255 * (i+1) / num_samples)  # Blue varies by video
                
                # Convert to torch tensor [C, H, W] with normalized values
                frame = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                frames.append(frame)
            
            video = torch.stack(frames)
            videos.append(video)
            
            # Create placeholder poses (identity matrices flattened)
            # This creates a 4x4 identity matrix flattened to match expected format
            # pose = torch.eye(4).flatten()[:-4]  # Remove last row of 4x4 matrix
            poses.append(torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1))
        
        return videos, poses

    first_frame_list = create_placeholder_images(num_samples=5)
    video_list, poses_list = create_placeholder_video_and_poses(num_samples=5)

# Function to load image from path
def load_image_for_navigation(image_path):
    """Load image from path and prepare for navigation"""
    # Load image and get default intrinsics
    image, _ = load_img_and_K(image_path, None, K=None, device=DEVICE)
    
    # Transform image to the target size
    config = OmegaConf.load(CONFIG_PATH)
    image, _ = transform_img_and_K(image, (config.model.height, config.model.width), mode="crop", K=None)
    
    # Create initial video with single frame and pose
    video = image
    pose = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
    
    return {
        "image": tensor_to_pil(image),
        "video": video,
        "pose": pose
    }


class CustomProgressBar:
    def __init__(self, pbar):
        self.pbar = pbar

    def set_postfix(self, **kwargs):
        pass

    def __getattr__(self, attr):
        return getattr(self.pbar, attr)

def get_duration_navigate_video(video: torch.Tensor,
    poses: torch.Tensor,
    x_angle: float,
    y_angle: float,
    distance: float
):
    # Estimate processing time based on navigation complexity and number of frames
    base_duration = 15  # Base duration in seconds
    
    # Add time for more complex navigation operations
    if abs(x_angle) > 20 or abs(y_angle) > 30:
        base_duration += 10  # More time for sharp turns
    
    if distance > 100:
        base_duration += 10  # More time for longer distances
    
    # Add time proportional to existing video length (more frames = more processing)
    base_duration += min(10, len(video))
    
    return base_duration

@spaces.GPU(duration=get_duration_navigate_video)
@torch.autocast("cuda")
@torch.no_grad()
def navigate_video(
    video: torch.Tensor,
    poses: torch.Tensor,
    x_angle: float,
    y_angle: float,
    distance: float,
):
    """
    Generate new video frames by navigating in the 3D scene.
    This function uses the Navigator class from navigation.py to handle movement:
    - y_angle parameter controls left/right turning (turn_left/turn_right methods)
    - distance parameter controls forward movement (move_forward method)
    - x_angle parameter controls vertical angle (not directly implemented in Navigator)
    
    Each Navigator instance is stored based on the video session to maintain state.
    """
    try:
        # Convert first frame to PIL Image for navigator
        initial_frame = tensor_to_pil(video[0])
        
        # Initialize the navigator for this session if not already done
        if len(NAVIGATORS) == 0:
            # Create a new navigator instance
            NAVIGATORS.append(Navigator(MODEL, step_size=0.1, num_interpolation_frames=4))
            
            # Get the initial pose and convert to numpy
            initial_pose = poses[0].cpu().numpy().reshape(4, 4)
            
            # Default camera intrinsics if not available
            initial_K = np.array(get_default_intrinsics()[0])
            
            # Initialize the navigator
            NAVIGATORS[0].initialize(initial_frame, initial_pose, initial_K)

        navigator = NAVIGATORS[0]
        
        # Generate new frames based on navigation commands
        new_frames = []
        
        # First handle any x-angle (vertical angle) adjustments
        # Note: This is approximated as Navigator doesn't directly support this
        if abs(x_angle) > 0:
            # Implementation for x-angle could be added here
            # For now, we'll skip this as it's not directly supported
            pass
        
        # Next handle y-angle (turning left/right)
        if abs(y_angle) > 0:
            # Use Navigator's turn methods
            if y_angle > 0:
                new_frames = navigator.turn_left(abs(y_angle//2))
            else:
                new_frames = navigator.turn_right(abs(y_angle//2))
        # Finally handle distance (moving forward)
        elif distance > 0:
            # Calculate number of steps based on distance
            steps = max(1, int(distance / 10))
            new_frames = navigator.move_forward(steps)
        elif distance < 0:
            # Handle moving backward if needed
            steps = max(1, int(abs(distance) / 10))
            new_frames = navigator.move_backward(steps)
        
        if not new_frames:
            # If no new frames were generated, return the current state
            return video, poses, tensor_to_pil(video[-1]), export_to_video([tensor_to_pil(video[i]) for i in range(len(video))], fps=NAVIGATION_FPS), [(tensor_to_pil(video[i]), f"t={i}") for i in range(len(video))]
        
        # Convert PIL images to tensors
        new_frame_tensors = []
        for frame in new_frames:
            # Convert PIL Image to tensor [C, H, W]
            frame_np = np.array(frame) / 255.0
            # Convert to [-1, 1] range to match the expected format
            frame_tensor = torch.from_numpy(frame_np.transpose(2, 0, 1)).float() * 2.0 - 1.0
            new_frame_tensors.append(frame_tensor)
        
        new_frames_tensor = torch.stack(new_frame_tensors)
        
        # Get the updated camera poses from the navigator
        current_pose = navigator.current_pose
        new_poses = torch.from_numpy(current_pose).float().unsqueeze(0).repeat(len(new_frames), 1, 1)
        
        # Reshape the poses to match the expected format
        new_poses = new_poses.view(len(new_frames), 4, 4)
        
        # Concatenate new frames and poses with existing ones
        updated_video = torch.cat([video.cpu(), new_frames_tensor], dim=0)
        updated_poses = torch.cat([poses.cpu(), new_poses], dim=0)
        
        # Create output images for gallery
        all_images = [(tensor_to_pil(updated_video[i]), f"t={i}") for i in range(len(updated_video))]
        updated_video_pil = [tensor_to_pil(updated_video[i]) for i in range(len(updated_video))]
        
        return (
            updated_video,
            updated_poses,
            tensor_to_pil(updated_video[-1]),  # Current view
            export_to_video(updated_video_pil, fps=NAVIGATION_FPS),  # Video
            all_images,  # Gallery
        )
    except Exception as e:
        print(f"Error in navigate_video: {e}")
        gr.Warning(f"Navigation error: {e}")
        # Return the original inputs to avoid crashes
        current_frame = tensor_to_pil(video[-1]) if len(video) > 0 else None
        all_frames = [(tensor_to_pil(video[i]), f"t={i}") for i in range(len(video))]
        video_frames = [tensor_to_pil(video[i]) for i in range(len(video))]
        video_output = export_to_video(video_frames, fps=NAVIGATION_FPS) if video_frames else None
        return video, poses, current_frame, video_output, all_frames


def undo_navigation(
    video: torch.Tensor,
    poses: torch.Tensor,
):
    """
    Undo the last navigation step by removing the last set of frames.
    Uses the Navigator's undo method which in turn uses the pipeline's undo_latest_move
    to properly handle surfels and state management.
    """
    if len(NAVIGATORS) > 0:
        navigator = NAVIGATORS[0]
        
        # Call the Navigator's undo method to handle the operation
        success = navigator.undo()
        
        if success:
            # Since the navigator has handled the frame removal internally,
            # we need to update our video and poses tensors to match
            updated_video = video[:len(navigator.frames)]
            updated_poses = poses[:len(navigator.frames)]
            
            # Create gallery images
            all_images = [(tensor_to_pil(updated_video[i]), f"t={i}") for i in range(len(updated_video))]
            
            return (
                updated_video,
                updated_poses,
                tensor_to_pil(updated_video[-1]),
                export_to_video([tensor_to_pil(updated_video[i]) for i in range(len(updated_video))], fps=NAVIGATION_FPS),
                all_images,
            )
        else:
            gr.Warning("You have no moves left to undo!")
    else:
        gr.Warning("No navigation session available!")
    
    # If undo wasn't successful or no navigator exists, return original state
    all_images = [(tensor_to_pil(video[i]), f"t={i}") for i in range(len(video))]
    
    return (
        video,
        poses,
        tensor_to_pil(video[-1]),
        export_to_video([tensor_to_pil(video[i]) for i in range(len(video))], fps=NAVIGATION_FPS),
        all_images,
    )





def render_demonstrate(
    s: Literal["Selection", "Generation"],
    idx: int,
    demonstrate_stage: gr.State,
    demonstrate_selected_index: gr.State,
    demonstrate_current_video: gr.State,
    demonstrate_current_poses: gr.State
):
    gr.Markdown(
        """
        ## Single Image → Consistent Scene Navigation
        > #### _Select an image and navigate through the scene by controlling camera movements._
    """,
    elem_classes=["task-title"]
    )
    match s:
        case "Selection":
            with gr.Group():
                # Add upload functionality
                with gr.Group(elem_classes=["gradio-box"]):
                    gr.Markdown("### Upload Your Own Image")
                    gr.Markdown("_Upload an image to navigate through its 3D scene_")
                    with gr.Row():
                        with gr.Column(scale=3):
                            upload_image = gr.Image(
                                label="Upload an image",
                                type="filepath",
                                height=300,
                                elem_id="upload-image"
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("#### Instructions:")
                            gr.Markdown("1. Upload a clear, high-quality image")
                            gr.Markdown("2. Images with distinct visual features work best")
                            gr.Markdown("3. Landscape or architectural scenes are ideal")
                            upload_btn = gr.Button("Start Navigation", variant="primary", size="lg")
                    
                    def process_uploaded_image(image_path):
                        if image_path is None:
                            gr.Warning("Please upload an image first")
                            return "Selection", None, None, None
                        try:
                            # Clear visualization directory to prevent users from seeing each other's generated images
                            clear_visualization_directory()
                            
                            # Load image and prepare for navigation
                            result = load_image_for_navigation(image_path)
                            
                            # Clear any existing navigators
                            global NAVIGATORS
                            NAVIGATORS = []
                            
                            return (
                                "Generation",
                                None,  # No predefined index for uploaded images
                                result["video"],
                                result["pose"],
                            )
                        except Exception as e:
                            print(f"Error in process_uploaded_image: {e}")
                            gr.Warning(f"Error processing uploaded image: {e}")
                            return "Selection", None, None, None
                    
                    upload_btn.click(
                        fn=process_uploaded_image,
                        inputs=[upload_image],
                        outputs=[demonstrate_stage, demonstrate_selected_index, demonstrate_current_video, demonstrate_current_poses]
                    )
                
                gr.Markdown("### Or Choose From Our Examples")
                # Define image captions
                image_captions = {
          
                    'test_samples/oxford.jpg': 'Oxford University',
                    'test_samples/open_door.jpg': 'Bedroom Interior',
                    'test_samples/living_room.jpg': 'Living Room',
                    'test_samples/arc_de_tromphe.jpeg': 'Arc de Triomphe',
                    'test_samples/jesus.jpg': 'Jesus College',
                    'test_samples/changi.jpg': 'Changi Airport',
                }
                
                # Load all images for the gallery with captions
                gallery_images = []
                for img_path in IMAGE_PATHS:
                    try:
                        # Get caption or default to basename
                        caption = image_captions.get(img_path, os.path.basename(img_path))
                        gallery_images.append((img_path, caption))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                
                # Show image gallery for selection
                demonstrate_image_gallery = gr.Gallery(
                    value=gallery_images,
                    label="Select an Image to Start Navigation",
                    columns=len(gallery_images),
                    height=400,
                    allow_preview=True,
                    preview=False,
                    elem_id="navigation-gallery"
                )
                
                gr.Markdown("_Click on an image to begin navigation_")
                
                def start_navigation(evt: gr.SelectData):
                    try:
                        # Clear visualization directory to prevent users from seeing each other's generated images
                        clear_visualization_directory()
                        
                        # Get the selected image path
                        selected_path = IMAGE_PATHS[evt.index]
                        
                        # Load image and prepare for navigation
                        result = load_image_for_navigation(selected_path)
                        
                        # Clear any existing navigators
                        global NAVIGATORS
                        NAVIGATORS = []
                        
                        return (
                            "Generation",
                            evt.index,
                            result["video"],
                            result["pose"],
                        )
                    except Exception as e:
                        print(f"Error in start_navigation: {e}")
                        gr.Warning(f"Error starting navigation: {e}")
                        return "Selection", None, None, None
                
                demonstrate_image_gallery.select(
                    fn=start_navigation,
                    inputs=None, 
                    outputs=[demonstrate_stage, demonstrate_selected_index, demonstrate_current_video, demonstrate_current_poses]
                )

        case "Generation":
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        demonstrate_current_view = gr.Image(
                            label="Current View",
                            width=256,
                            height=256,
                        )
                        demonstrate_video = gr.Video(
                            label="Generated Video",
                            width=256,
                            height=256,
                            autoplay=True,
                            loop=True,
                            show_share_button=True,
                            show_download_button=True,
                        )

                    demonstrate_generated_gallery = gr.Gallery(
                        value=[],
                        label="Generated Frames",
                        columns=[6],
                    )
                    
                    # Initialize the current view with the selected image if available
                    if idx is not None:
                        try:
                            selected_path = IMAGE_PATHS[idx]
                            result = load_image_for_navigation(selected_path)
                            demonstrate_current_view.value = result["image"]
                        except Exception as e:
                            print(f"Error initializing current view: {e}")

                with gr.Column():
                    gr.Markdown("### Navigation Controls ↓")
                    with gr.Accordion("Instructions", open=False):
                        gr.Markdown("""
                            - **The model will predict the next few frames based on your camera movements. Repeat the process to continue navigating through the scene.**
                            - **Use the navigation controls to move forward/backward and turn left/right.**
                            - **At the end of your navigation, you can save your camera path for later use.**
                           
                        """)
                    # with gr.Tab("Basic", elem_id="basic-controls-tab"):
                    with gr.Group():
                        gr.Markdown("_**Select a direction to move:**_")
                        # First row: Turn left/right
                        with gr.Row(elem_id="basic-controls"):
                            gr.Button(
                                "↰20°\nVeer",
                                size="sm",
                                min_width=0,
                                variant="primary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=20,
                                    distance=0,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )

                            gr.Button(
                                "↖10°\nTurn",
                                size="sm",
                                min_width=0,
                                variant="primary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=10,
                                    distance=0,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )

                            gr.Button(
                                "↗10°\nTurn",
                                size="sm",
                                min_width=0,
                                variant="primary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=-10,
                                    distance=0,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )
                            gr.Button(
                                "↱\n20° Veer",
                                size="sm",
                                min_width=0,
                                variant="primary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=-20,
                                    distance=0,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )
                        
                        # Second row: Forward/Backward movement
                        with gr.Row(elem_id="forward-backward-controls"):
                            gr.Button(
                                "↓\nBackward",
                                size="sm",
                                min_width=0,
                                variant="secondary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=0,
                                    distance=-10,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )
                            
                            gr.Button(
                                "↑\nForward",
                                size="sm",
                                min_width=0,
                                variant="secondary",
                            ).click(
                                fn=partial(
                                    navigate_video,
                                    x_angle=0,
                                    y_angle=0,
                                    distance=10,
                                ),
                                inputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                ],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )
                    gr.Markdown("---")
                    with gr.Group():
                        gr.Markdown("_**Navigation controls:**_")
                        with gr.Row():
                            gr.Button("Undo Last Move", variant="huggingface").click(
                                fn=undo_navigation,
                                inputs=[demonstrate_current_video, demonstrate_current_poses],
                                outputs=[
                                    demonstrate_current_video,
                                    demonstrate_current_poses,
                                    demonstrate_current_view,
                                    demonstrate_video,
                                    demonstrate_generated_gallery,
                                ],
                            )
                            
                            # Add a function to save camera poses
                            def save_camera_poses(video, poses):
                                if len(NAVIGATORS) > 0:
                                    navigator = NAVIGATORS[0]
                                    # Create a directory for saved poses
                                    os.makedirs("./visualization", exist_ok=True)
                                    save_path = f"./visualization/transforms_{len(navigator.frames)}_frames.json"
                                    navigator.save_camera_poses(save_path)
                                    return gr.Info(f"Camera poses saved to {save_path}")
                                return gr.Warning("No navigation instance found")
                            
                            gr.Button("Save Camera", variant="huggingface").click(
                                fn=save_camera_poses,
                                inputs=[demonstrate_current_video, demonstrate_current_poses],
                                outputs=[]
                            )
                            
                            # Add a button to return to image selection
                            def reset_navigation():
                                # Clear current navigator
                                global NAVIGATORS
                                NAVIGATORS = []
                                return "Selection", None, None, None
                            
                            gr.Button("Choose New Image", variant="secondary").click(
                                fn=reset_navigation,
                                inputs=[],
                                outputs=[demonstrate_stage, demonstrate_selected_index, demonstrate_current_video, demonstrate_current_poses]
                            )


# Create the Gradio Blocks
with gr.Blocks(theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.HTML(
        """
    <style>
    [data-tab-id="task-1"], [data-tab-id="task-2"], [data-tab-id="task-3"] {
        font-size: 16px !important;
        font-weight: bold;
    }
    #page-title h1 {
        color: #002147 !important;
    }
    .task-title h2 {
        color: #004080 !important;
    }
    .header-button-row {
        gap: 4px !important;
    }
    .header-button-row div {
        width: 131.0px !important;
    }
    .header-button-column {
        width: 131.0px !important;
        gap: 5px !important;
    }
    .header-button a {
        border: 1px solid #002147;
    }
    .header-button .button-icon {
        margin-right: 8px;
    }
    .demo-button-column .gap {
        gap: 5px !important;
    }
    #basic-controls {
        column-gap: 0px;
    }
    #basic-controls-tab {
        padding: 0px;
    }
    #advanced-controls-tab {
        padding: 0px;
    }
    #forward-backward-controls {
        column-gap: 0px;
        justify-content: center;
        margin-top: 8px;
    }
    #selected-demo-button {
        color: #004080;
        text-decoration: underline;
    }
    .demo-button {
        text-align: left !important;
        display: block !important;
    }
    #navigation-gallery {
        margin-bottom: 15px;
    }
    #navigation-gallery .gallery-item {
        cursor: pointer;
        border-radius: 6px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    #navigation-gallery .gallery-item:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    #navigation-gallery .gallery-item.selected {
        border: 3px solid #002147;
    }
    /* Upload image styling */
    #upload-image {
        border-radius: 8px;
        border: 2px dashed #002147;
        padding: 10px;
        transition: all 0.3s ease;
    }
    #upload-image:hover {
        border-color: #002147;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    /* Box styling */
    .gradio-box {
        border-radius: 10px;
        margin-bottom: 20px;
        padding: 15px;
        background-color: #002147;
        border: 1px solid #002147;
    }
    /* Start Navigation button styling */
    button[data-testid="Start Navigation"] {
        background-color: #004080 !important;
        border-color: #004080 !important;
        color: white !important;
    }
    button[data-testid="Start Navigation"]:hover {
        background-color: #002147 !important;
        border-color: #002147 !important;
    }
    /* Override Gradio's primary button color */
    .gradio-button.primary {
        background-color: #004080 !important;
        border-color: #004080 !important;
        color: white !important;
    }
    .gradio-button.primary:hover {
        background-color: #002147 !important;
        border-color: #002147 !important;
    }
    </style>
    """
    )

    demo_idx = gr.State(value=3)

    with gr.Sidebar():
        gr.Image("assets/title_logo.png", width=60, height=60, show_label=False, show_download_button=False, container=False, interactive=False, show_fullscreen_button=False)
        gr.Markdown("# Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory", elem_id="page-title")
        gr.Markdown(
            "### Interactive Demo for [_VMem_](http://arxiv.org/abs/2506.18903) that enables interactive consistent video scene generation."
        )
        gr.Markdown("---")
        gr.Markdown("#### Links ↓")
        with gr.Row(elem_classes=["header-button-row"]):
            with gr.Column(elem_classes=["header-button-column"], min_width=0):
                gr.Button(
                    value="Website",
                    link="https://v-mem.github.io/",
                    icon="https://simpleicons.org/icons/googlechrome.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
                gr.Button(
                    value="Paper",
                    link="http://arxiv.org/abs/2506.18903",
                    icon="https://simpleicons.org/icons/arxiv.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
            with gr.Column(elem_classes=["header-button-column"], min_width=0):
                gr.Button(
                    value="Code",
                    link="https://github.com/runjiali-rl/vmem",
                    icon="https://simpleicons.org/icons/github.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
                gr.Button(
                    value="Weights",
                    link="https://huggingface.co/liguang0115/vmem",
                    icon="https://simpleicons.org/icons/huggingface.svg",
                    elem_classes=["header-button"],
                    size="md",
                    min_width=0,
                )
        gr.Markdown("---")
        gr.Markdown("This demo interface is adapted from the History-Guided Video Diffusion demo template. We thank the authors for their work.")



    demonstrate_stage = gr.State(value="Selection")
    demonstrate_selected_index = gr.State(value=None)
    demonstrate_current_video = gr.State(value=None)
    demonstrate_current_poses = gr.State(value=None)

    @gr.render(inputs=[demo_idx, demonstrate_stage, demonstrate_selected_index])
    def render_demo(
        _demo_idx, _demonstrate_stage, _demonstrate_selected_index
    ):
        match _demo_idx:
            case 3:
                render_demonstrate(_demonstrate_stage, _demonstrate_selected_index, demonstrate_stage, demonstrate_selected_index, demonstrate_current_video, demonstrate_current_poses)
                

if __name__ == "__main__":
    demo.launch(debug=False,
                share=True,
                max_threads=1,
                show_error=False,
                )
