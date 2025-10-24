import torch
import numpy as np
import cv2
import os
from collections import deque
from .ggcnn2 import GGCNN2

class Grasp_Predictor:
  def __init__(self):
    self.grasp_prediction_model = self.setup()

  def _find_best_grasp_center(self, quality_map, threshold_ratio=0.7):
    """
    Finds the best grasp center by growing a region from the best pixel using BFS.

    Args:
        quality_map (np.ndarray): 2D array of grasp quality scores.
        threshold_ratio (float): Fraction of max value to include in region (e.g., 0.9 means include neighbors >= 90% of max).

    Returns:
        (best_row, best_col): Tuple of center pixel coordinates of the best grasp region.
    """

    # Step 1. Find the starting pixel (maximum quality)
    max_val = np.max(quality_map)
    start_pos = np.unravel_index(np.argmax(quality_map), quality_map.shape)
    threshold = threshold_ratio * max_val

    h, w = quality_map.shape
    visited = np.zeros_like(quality_map, dtype=bool)
    region_pixels = []

    # Step 2. BFS (4-connected)
    q = deque([start_pos])
    visited[start_pos] = True

    while q:
        y, x = q.popleft()
        region_pixels.append((y, x))

        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # neighbors
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if quality_map[ny, nx] >= threshold:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    # Step 3. Compute region centroid
    if not region_pixels:
        return start_pos  # fallback to max pixel

    ys, xs = zip(*region_pixels)
    y_center = int(np.mean(ys))
    x_center = int(np.mean(xs))

    return y_center, x_center

  def setup(self):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grasp_prediction_model = GGCNN2().to(device)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "epoch_50_cornell_statedict.pt")
    try:
      print(f"Loading pretrained weights from {MODEL_PATH}...")
      state_dict = torch.load(MODEL_PATH, map_location=device)
      
      if 'model_state_dict' in state_dict:
          state_dict = state_dict['model_state_dict']
      elif 'state_dict' in state_dict:
          state_dict = state_dict['state_dict']
          
      grasp_prediction_model.load_state_dict(state_dict)
    except FileNotFoundError:
      print(f"Warning: Model file '{MODEL_PATH}' not found. Using untrained model.")
      print("Please provide the path to your pretrained GGCNN weights.")
    except Exception as e:
      print(f"Error loading weights: {e}")
      print("Proceeding with untrained model.")
    return grasp_prediction_model
  
  def predict_grasp(self, depth_normalized, bbox_mask = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.grasp_prediction_model.eval()
    # change (h,w) to (1,1,h,w) where channel and batch is 1
    depth_tensor = torch.from_numpy(depth_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
      output = self.grasp_prediction_model(depth_tensor)
      #print("Output: ", output)
      
      # GGCNN outputs a tuple of 4 tensors
      # Q, cos, sin, width
      # squeeze removes all dimensions of 1 so the batch and channel is removed
      quality, cos_map, sin_map, width = output
      quality_map = quality.squeeze().cpu().numpy()
      # recommended to add some gaussian noise by GGCNN2
      #quality_map = cv2.GaussianBlur(quality_map, (5,5), 0)

      #quality_map = mask_edges(quality_map)
      cos_angle = cos_map.squeeze().cpu().numpy()
      sin_angle = sin_map.squeeze().cpu().numpy()
      width_map = width.squeeze().cpu().numpy()
      # Convert cos/sin to angle
      angle_map = np.arctan2(sin_angle, cos_angle)
    
    if bbox_mask is not None:
      print("masking...")
      quality_map = quality_map * bbox_mask
      np.savetxt('grasp_quality.txt', quality_map, fmt='%.6f')
    
    # Find best grasp (highest quality)
    best_idx = np.unravel_index(np.argmax(quality_map), quality_map.shape)
    y_center, x_center = best_idx
    '''
    top_mask = quality_map > 0.9 * np.max(quality_map)
    y_center, x_center = np.mean(np.argwhere(top_mask), axis=0)
    '''
    #y_center, x_center = self._find_best_grasp_center(quality_map)

    print("y_center: ", y_center)
    print("x_center: ", x_center)
    best_row = int(y_center)
    best_col = int(x_center)
    
    # Getting the corresponding Q, angles and widths
    grasp_quality = quality_map[best_row, best_col]
    grasp_angle = angle_map[best_row, best_col]
    grasp_width = width_map[best_row, best_col]
    
    return best_row, best_col, grasp_angle, grasp_width, grasp_quality, quality_map, angle_map, width_map

  def visualize_grasp_predictions(self, depth_image, quality_map, angle_map, width_map, 
                                best_row, best_col, save_path='grasp_visualization.png'):
    """Visualize grasp predictions overlaid on depth image."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Depth image
    axes[0, 0].imshow(depth_image, cmap='gray')
    axes[0, 0].plot(best_col, best_row, 'r*', markersize=20)
    axes[0, 0].set_title('Depth Image + Best Grasp')
    axes[0, 0].axis('off')
    
    # Quality map
    im1 = axes[0, 1].imshow(quality_map, cmap='jet')
    axes[0, 1].plot(best_col, best_row, 'r*', markersize=20)
    axes[0, 1].set_title('Grasp Quality')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Angle map
    im2 = axes[1, 0].imshow(angle_map, cmap='hsv')
    axes[1, 0].plot(best_col, best_row, 'r*', markersize=20)
    axes[1, 0].set_title('Grasp Angle')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Width map
    im3 = axes[1, 1].imshow(width_map, cmap='viridis')
    axes[1, 1].plot(best_col, best_row, 'r*', markersize=20)
    axes[1, 1].set_title('Gripper Width')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grasp visualization saved to {save_path}")
    plt.close()