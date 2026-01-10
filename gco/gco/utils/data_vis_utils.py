# General imports.
import torch
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import numpy as np

# Project imports.
from gco.utils.transform_utils import Transform2, meters2pixel_dist
from gco.utils.model_utils import tokens_to_pixels
from gco.utils.transform_utils import apply_transform_to_mask
from gco.config import Config as cfg

####################
# Data visualization.
#################### 
def visualize_mask_with_contacts(mask: torch.Tensor,  # Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
                                    contacts: torch.Tensor,  # Contact points (num_contacts, 2).
                                    save_path: str = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mask = mask.cpu().numpy()
    contacts = contacts.cpu().numpy()
    # Draw contacts on the mask.
    for contact in contacts:
        mask[contact[0], contact[1]] = 0.5
    plt.scatter(contacts[:, 1], contacts[:, 0], color="red", marker="x")
    ax.imshow(mask, cmap="gray")
    plt.savefig(save_path)
    plt.close()

def visualize_transformed_mask(mask: torch.Tensor, transform: Transform2, save_path: str = None):
    """
    Visualize a relative transform in the world.
    :param world: The world.
    :param relative_transform: The relative transform.
    :param save_path: The path to save the figure to.
    """
    # Create a mask goal by transforming the mask.
    (dx, dy), theta = transform.get_t(), transform.get_theta()
    transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
    mask_goal = apply_transform_to_mask(mask, transform)
    # Create a figure.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Draw the mask.
    ax.imshow(mask.cpu().numpy(), cmap="gray")
    # Draw the goal mask.
    ax.imshow(mask_goal.cpu().numpy(), cmap="gray", alpha=0.1)
    # Draw the transform.
    dx, dy = transform.get_t().cpu().numpy().squeeze()
    theta = transform.get_theta().cpu().numpy().squeeze()
    ax.text(0.05, 0.95, f"dx: {dx:.2f}, dy: {dy:.2f}, theta: {theta:.2f}", 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment="top",
            color='white',
            fontfamily='sans-serif',
            fontweight='bold')
    # Add RGB XY    axes lines on the top right corner.
    ax.text(20, 45, "X", fontsize=12, color='red', fontfamily='sans-serif', fontweight='bold')
    ax.text(5, 65, "Y", fontsize=12, color='green', fontfamily='sans-serif', fontweight='bold')
    ax.arrow(25, 65, 0, -10, head_width=2, head_length=4, fc='red', ec='red')
    ax.arrow(25, 65, -10, 0, head_width=2, head_length=4, fc='green', ec='green')

    # Add a title.
    ax.set_title("Transformed Mask")

    # Save the figure, if can, otherwise return the figure.
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        return ax

def visualize_push_trajectory(  mask: torch.Tensor,  # Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
                                trajectory: torch.Tensor,  # Trajectory (num_contacts,num_steps, 2).
                                contacts: torch.Tensor,  # Contact points (num_steps, num_contacts, 2). In pixels.
                                transform: Transform2,  # Transform (num_steps, 2).
                                mask_goal: torch.Tensor | None = None,  # Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
                                save_path: str = None,
                                ax: plt.Axes = None,
                                gt_contacts: torch.Tensor | None = None,  # Ground truth contact points (N, 2). In pixels.
                                gt_trajectory: torch.Tensor | None = None):  # Ground truth trajectory (N, H, 2).
    """
    Visualize a push trajectory.
    :param mask: Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
    :param mask_goal: Binary mask (HIGH_RES, HIGH_RES), 1 is object, 0 is background.
    :param trajectory: Trajectory (num_contacts, num_steps, 2).
    :param contacts: Contact points (num_contacts, 2). In pixels.
    :param transform: Transform2.
    :param save_path: The path to save the figure to.
    """
    
    if mask_goal is None:
        mask_goal = torch.zeros_like(mask)
        B = mask.shape[0]
        for b in range(B):  # For each batch element.
            dx, dy, theta = transform.get_t().cpu().numpy().squeeze(), transform.get_theta().cpu().numpy().squeeze()
            transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
            mask_goal_b = apply_transform_to_mask(mask[b], transform)
            mask_goal[b] = mask_goal_b


    # Translate the trajectories to start at their corresponding contact points.
    trajectory = trajectory + contacts.unsqueeze(1)

    # Define scaling factor for higher resolution matplotlib artists
    scale_factor = 4  # Scale up coordinates by 4x for smoother rendering
    
    # Create a figure.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15 * scale_factor, 15 * scale_factor))  # Scale figure size too
    
    # Set background to white
    ax.set_facecolor('white')
    
    # Add faint grid for pixel boundaries (scaled)
    # ax.grid(True, alpha=0.7, linewidth=0.5, color='gray')
    # ax.set_xticks(range(0, mask.shape[1] * scale_factor, 10 * scale_factor))  # Every 10 pixels (scaled)
    # ax.set_yticks(range(0, mask.shape[0] * scale_factor, 10 * scale_factor))   # Every 10 pixels (scaled)
    
    # Remove axis boundaries
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Create custom colormap for orange masks with noise
    mask_np = mask.cpu().numpy()
    # Add noise to the mask for texture
    noise = np.random.normal(0, 0.0, mask_np.shape)
    mask_with_noise = np.clip(mask_np + noise, 0, 1)
    
    # Scale up the mask image to match coordinate system
    from scipy.ndimage import zoom
    mask_scaled = zoom(mask_with_noise, scale_factor, order=0)  # order=0 for nearest neighbor (keeps pixelated look)
    
    # Create orange colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', '#FFB366']  # White to light orange
    orange_cmap = LinearSegmentedColormap.from_list('orange', colors)
    
    # Draw the mask with orange color and noise (now scaled to match coordinates)
    ax.imshow(mask_scaled, cmap=orange_cmap, alpha=0.95)
    # Draw the trajectory with improved color (scaled coordinates)
    for i in range(trajectory.shape[0]):
        ax.plot(trajectory[i, :, 1] * scale_factor, trajectory[i, :, 0] * scale_factor, color="#8E44AD", linewidth=2.5, alpha=0.9)
    # Draw the contacts with correct robot radius size (scaled coordinates)
    robot_radius_pixels = meters2pixel_dist(cfg.robot_radius)
    # ax.scatter(contacts[:, 1], contacts[:, 0], color="yellow", marker="o", s=robot_radius_pixels**2 * np.pi, alpha=0.7)
    contacts_np = contacts.cpu().numpy()
    for i, contact in enumerate(contacts_np):
        circle = plt.Circle((contact[1] * scale_factor, contact[0] * scale_factor), robot_radius_pixels * scale_factor, color="#F39C12", alpha=0.8, 
                           edgecolor='#D68910', linewidth=2)
        ax.add_artist(circle)
    
    # Draw fixed square at mask token position (outside image bounds) - scaled coordinates
    mask_position = cfg.pixel_value_mask * scale_factor  # Scale the mask position
    square_size = 16 * scale_factor  # Scale the square size
    square = plt.Rectangle((mask_position - square_size//2, mask_position - square_size//2), 
                         square_size, square_size, 
                         fill=False, edgecolor='#E74C3C', linewidth=2)
    ax.add_artist(square)
    
    # Add [M] label above the square (no boundary) - scaled coordinates
    ax.text(mask_position, mask_position - 2 * square_size//3, "[M]", 
           fontsize=12, color='#E74C3C', fontfamily='serif', fontweight='bold',
           ha='center', va='bottom')
    
    # Set axis limits to always include mask token position - scaled coordinates
    ax.set_xlim(-10 * scale_factor, mask_position + 20 * scale_factor)  # Extend x-axis to include mask position
    ax.set_ylim(mask_position + 20 * scale_factor, -10 * scale_factor)  # Extend y-axis to include mask position (inverted for matplotlib)
    
    # Draw ground truth data if provided
    if gt_contacts is not None:
        gt_contacts = gt_contacts.unsqueeze(1)
        # Draw ground truth contacts with different color (scaled coordinates)
        gt_contacts_np = gt_contacts.cpu().numpy()
        for contact in gt_contacts_np:
            contact = contact[0]
            circle = plt.Circle((contact[1] * scale_factor, contact[0] * scale_factor), robot_radius_pixels * scale_factor, color="#27AE60", alpha=0.95,
                               edgecolor='#1E8449', linewidth=2)
            ax.add_artist(circle)
    
    if gt_trajectory is not None:
        # Translate ground truth trajectories to start at their corresponding contact points
        gt_trajectory_translated = gt_trajectory + gt_contacts if gt_contacts is not None else gt_trajectory
        # Draw ground truth trajectories with different color (scaled coordinates)
        for i in range(gt_trajectory_translated.shape[0]):
            ax.plot(gt_trajectory_translated[i, :, 1] * scale_factor, gt_trajectory_translated[i, :, 0] * scale_factor, 
                   color="#27AE60", linestyle="--", alpha=0.8, linewidth=2)

    # Draw the goal mask with brighter orange (scale up to match coordinates)
    mask_goal_np = mask_goal.cpu().numpy()
    # Add noise to the goal mask for texture
    noise_goal = np.random.normal(0, 0.0, mask_goal_np.shape)
    mask_goal_with_noise = np.clip(mask_goal_np + noise_goal, 0, 1)
    
    # Scale up the goal mask image to match coordinate system
    mask_goal_scaled = zoom(mask_goal_with_noise, scale_factor, order=0)  # order=0 for nearest neighbor (keeps pixelated look)
    
    # Create brighter orange colormap for goal mask
    colors_goal = ['white', '#FF8C00']  # White to brighter orange
    orange_goal_cmap = LinearSegmentedColormap.from_list('orange_goal', colors_goal)
    
    ax.imshow(mask_goal_scaled, cmap=orange_goal_cmap, alpha=0.2)

    # Write the transform (dx, dy, theta) to the figure with improved font
    dx, dy = transform.get_t().cpu().numpy().squeeze()
    theta = transform.get_theta().cpu().numpy().squeeze()
    ax.text(0.05, 0.95, f"dx: {dx:.2f}, dy: {dy:.2f}, dθ: {theta:.2f}", 
            transform=ax.transAxes, 
            fontsize=14,  # Don't scale text that uses transAxes
            verticalalignment="top",
            color='#2C3E50',  # Dark blue-gray for better contrast
            fontfamily='sans-serif',
            fontweight='normal',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none'))

    # Add RGB XY axes lines on the top right corner with improved styling (scaled coordinates)
    ax.text(20 * scale_factor, 45 * scale_factor, "X", fontsize=12, color='#E74C3C', fontfamily='serif', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    ax.text(5 * scale_factor, 65 * scale_factor, "Y", fontsize=12, color='#27AE60', fontfamily='serif', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    ax.arrow(25 * scale_factor, 65 * scale_factor, 0, -10 * scale_factor, head_width=2 * scale_factor, head_length=4 * scale_factor, fc='#E74C3C', ec='#E74C3C', linewidth=2)
    ax.arrow(25 * scale_factor, 65 * scale_factor, -10 * scale_factor, 0, head_width=2 * scale_factor, head_length=4 * scale_factor, fc='#27AE60', ec='#27AE60', linewidth=2)


    # Save the figure, if can, otherwise return the figure.
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        return ax

def visualize_batch(batch: dict, 
                   pred_contacts: torch.Tensor | None = None,
                   pred_trajectory: torch.Tensor | None = None,
                   save_path: str = None):
    """
    Visualize a batch of push trajectories.
    :param batch: A dict of keys: mask, mask_goal, trajectory, start_contacts, transform_object.
    :param pred_contacts: Optional predicted contact points (B, N, 2) to compare with ground truth.
    :param pred_trajectory: Optional predicted trajectory (B, N, H, 2) to compare with ground truth.
    :param save_path: The path to save the figure to.
    """
    if pred_contacts is not None and pred_trajectory is not None:
        print(pred_contacts.shape, pred_trajectory.shape)
    # Create a figure with subplots.
    B = batch["mask"].shape[0]
    if "mask_goal" not in batch:
        batch["mask_goal"] = torch.zeros_like(batch["mask"])
        for b in range(B):
            dx, dy, theta = batch["transform_object"][b,0], batch["transform_object"][b,1], batch["transform_object"][b,2]
            transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
            mask_goal = apply_transform_to_mask(batch["mask"][b], transform)
            batch["mask_goal"][b] = mask_goal

    if "tokens" in batch:
        batch["start_contacts"] = tokens_to_pixels(batch["tokens"])
        # Shape (B, N, 2) to (B, N, 1, 2).
        batch["start_contacts"] = batch["start_contacts"]
    
    # Calculate optimal grid layout
    import numpy as np
    rows = min(2, B) if B <= 4 else int(np.ceil(np.sqrt(B)))
    cols = int(np.ceil(B / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 10))
    
    # Ensure axs is always 2D for consistent indexing
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1 and cols > 1:
        # axs is 1D array, reshape to 2D
        axs = axs.reshape(1, -1)
    elif rows > 1 and cols == 1:
        # axs is 1D array, reshape to 2D
        axs = axs.reshape(-1, 1)
    # If rows > 1 and cols > 1, axs is already 2D

    for b in range(B): 
        # To cpu.
        mask = batch["mask"][b].cpu()
        mask_goal = batch["mask_goal"][b].cpu()
        trajectory = batch["trajectory"][b].cpu()
        start_contacts = batch["start_contacts"][b].cpu()
        transform_object = batch["transform_object"][b].cpu()
        transform_object = Transform2(transform_object[:2], transform_object[2])
        
        # Determine what to show as main data vs ground truth
        if pred_contacts is not None and pred_trajectory is not None:
            # print("WITH PREDICTIONS =================================")
            # Show predictions as main data, ground truth as comparison
            contacts_to_show = pred_contacts[b].cpu()
            trajectory_to_show = pred_trajectory[b].cpu()
            gt_contacts = start_contacts
            gt_trajectory = trajectory

        else:
            # print("WITHOUT PREDICTIONS =================================")
            # Show ground truth as main data, no comparison
            contacts_to_show = start_contacts
            trajectory_to_show = trajectory
            gt_contacts = None
            gt_trajectory = None 

        visualize_push_trajectory(
            mask, 
            trajectory_to_show, 
            contacts_to_show, 
            transform_object, 
            mask_goal=mask_goal, 
            save_path=None, 
            ax=axs[b//cols, b%cols],
            gt_contacts=gt_contacts,
            gt_trajectory=gt_trajectory
        )
    return fig, axs


def visualize_batch_denoising(
    batch_gt: dict,
    tokens_t: torch.Tensor,
    trajectory_t: torch.Tensor | None = None,
    gt_d: torch.Tensor | None = None,  # (B, N, 2)
    gt_c: torch.Tensor | None = None,  # (B, N, H, 2)
    interval: int = 100,          # ms between frames
    repeat: bool = True,
    num_freeze_frames: int = 0,
    save_path: str | None = None,  # e.g. "denoise.mp4" or "denoise.gif"
    only_last_frame: bool = False,
):
    """
    Show / save an animation of the contact-point denoising process.

    Args
    ----
    batch_gt      : dict with keys "mask", "mask_goal", "trajectory",
                    "start_contacts", "transform_object".
    tokens_t      : (K, B, N) tensor – contact tokens at each generation step.
    trajectory_t  : (K, B, N, H, 2) or None – trajectories at each step.
                    If None, zeros of shape (K, B, N, H, 2) are used.
    gt_d          : (K, B, N, V) tensor – ground truth contact tokens at each generation step.
    gt_c          : (K, B, N, H, 2) tensor – ground truth trajectories at each step.
    interval      : delay between frames in milliseconds (default 500 ms).
    repeat        : whether to loop the animation (default True).
    save_path     : if given, the animation is saved to this file.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    assert "mask" in batch_gt, "mask must be in batch_gt"
    assert "transform_object" in batch_gt, "transform_object must be in batch_gt"

    # Create a mask goal in the batch if not exists.
    if "mask_goal" not in batch_gt:
        batch_gt["mask_goal"] = torch.zeros_like(batch_gt["mask"])
        B = batch_gt["mask"].shape[0]
        for b in range(B):
            dx, dy, theta = batch_gt["transform_object"][b,0], batch_gt["transform_object"][b,1], batch_gt["transform_object"][b,2]
            transform = Transform2(t=torch.tensor([-dx, -dy], device=cfg.device), theta=theta)
            mask_goal = apply_transform_to_mask(batch_gt["mask"][b], transform)
            batch_gt["mask_goal"][b] = mask_goal

    K, B = tokens_t.shape[0], batch_gt["mask"].shape[0]

    # If no trajectory_t supplied, build a zero tensor with correct shape
    if trajectory_t is None:
        trajectory_t = torch.zeros_like(batch_gt["trajectory"])           # (B, N, H, 2)
        trajectory_t = trajectory_t.unsqueeze(0).repeat(K, 1, 1, 1, 1)    # (K, B, …)

    rows = min(3, B)                        # keep the grid moderately tall
    cols = math.ceil(B / rows)

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),  # Increased resolution
                            squeeze=False)  # always returns 2-D array
    
    # Set figure background to white
    fig.patch.set_facecolor('white')

    # Turn off unused axes (if any) and improve styling
    for i, ax in enumerate(axs.flat):
        if i >= B:
            ax.axis("off")
        else:
            # Set background to white
            ax.set_facecolor('white')
            # Remove all axis boundaries
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            # Remove tick marks
            ax.set_xticks([])
            ax.set_yticks([])

    def draw_frame(t: int):
        """Redraw all subplots for animation frame t."""
        if t >= K:
            t = K - 1

        for b in range(B):
            ax = axs[b // cols, b % cols]
            ax.clear()                              # wipe previous artists

            mask           = batch_gt["mask"][b].cpu()
            mask_goal      = batch_gt["mask_goal"][b].cpu()
            traj           = trajectory_t[t, b].cpu()
            contacts_px    = tokens_to_pixels(tokens_t[t, b]).cpu()      # (N, 2)
            transform_obj  = batch_gt["transform_object"][b]
            transform_obj  = Transform2(transform_obj[:2], transform_obj[2])

            # Prepare ground truth data if available
            gt_contacts_px = None
            gt_traj = None
            if gt_d is not None:
                gt_contacts_px = tokens_to_pixels(gt_d[b]).cpu()  # (N, 1, 2)
                gt_contacts_px = gt_contacts_px.squeeze(1)
            if gt_c is not None:
                gt_traj = gt_c[b].cpu()  # (N, H, 2)
            
            # Re-draw this subplot
            visualize_push_trajectory(
                mask,
                traj,
                contacts_px,
                transform_obj,
                mask_goal=mask_goal,
                ax=ax,                        # reuse the same axis
                save_path=None,               # don't save individual frames
                gt_contacts=gt_contacts_px,
                gt_trajectory=gt_traj
            )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Generation Step {t}", fontsize=16, fontfamily='sans-serif', fontweight='normal', 
                        color='#2C3E50', pad=20)

        return axs.flat[:B]   # returned artists (blit=False, so not critical)

    ani = FuncAnimation(
        fig,
        draw_frame,
        frames=range(K + 5) if not only_last_frame else [K - 1],  # Add 5 freeze frames at the end
        interval=interval,
        blit=False,
        repeat=repeat
    )

    if save_path is not None:
        # Pick the backend automatically from the file extension
        ani.save(save_path, dpi=300)  # Increased DPI for higher resolution
        print(f"animation saved to: {save_path}")
    plt.close(fig)  # <-- stops Jupyter from emitting the static PNG

    return ani
