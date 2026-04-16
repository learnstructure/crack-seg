import torch
from PIL import Image
from crack_seg.data_handlers.transforms import pred_transform
import cv2
import numpy as np
# from scipy import ndimage

def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    # Use the prediction-specific transform
    input_tensor = pred_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()
    return pred


def analyze_cracks(prob_mask, threshold=0.5, min_pixels=10):
    """
    Extract crack instances and orientations from a probability mask.
    
    Args:
        prob_mask (np.ndarray): 2D array of float, values in [0,1].
        threshold (float): Pixels > threshold become crack.
        min_pixels (int): Minimum number of pixels to consider a valid crack.
        
    Returns:
        list of dict: Each dict contains 'id', 'pixel_count', 'orientation', 'angle_deg'.
    """
    # 1. Binarize using threshold
    binary_mask = (prob_mask > threshold).astype(np.uint8)
    
    # 2. Morphological cleaning (remove small holes, close small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)        # Close small gaps
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)         # Remove small noise
    
    # 3. Connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    # label 0 is background; cracks are labels 1..num_labels-1
    
    crack_info = []
    for label_id in range(1, num_labels):
        # Coordinates of pixels belonging to this crack
        pts = np.column_stack(np.where(labels == label_id))
        if len(pts) < min_pixels:
            continue
        
        # 4. Compute orientation using PCA (principal axis)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        # Covariance matrix
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Eigenvector with largest eigenvalue = principal direction
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle_rad = np.arctan2(principal_axis[0], principal_axis[1])
        angle_deg = angle_rad * 180 / np.pi
        
        # Normalize angle to [0, 180) to avoid sign ambiguity
        angle_deg = angle_deg % 180
        
        # Classify orientation
        if angle_deg <= 20 or angle_deg >= 160:
            orientation = "horizontal"
        elif 70 <= angle_deg <= 110:
            orientation = "vertical"
        else:
            orientation = "inclined"
        
        crack_info.append({
            "id": label_id,
            "pixel_count": len(pts),
            "orientation": orientation,
            "angle_deg": round(angle_deg, 1).item()
        })
    
    return crack_info