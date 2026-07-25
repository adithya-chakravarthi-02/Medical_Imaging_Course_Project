"""
CT Image Preprocessing & Nodule Mask Processing Module
Handles DICOM / CT slice loading, Hounsfield Unit (HU) windowing,
intensity normalization, and nodule Region of Interest (ROI) cropping.
"""

import numpy as np
import cv2

def apply_lung_window(ct_scan, window_center=-600, window_width=1500):
    """
    Applies standard Clinical Lung Windowing (-1350 HU to +150 HU).
    """
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    
    windowed = np.clip(ct_scan, min_hu, max_hu)
    normalized = ((windowed - min_hu) / (max_hu - min_hu) * 255.0).astype(np.uint8)
    return normalized

def preprocess_ct_slice(img, percentile_clip=(1.0, 99.0)):
    """
    Normalizes grayscale pixel intensities and applies percentile contrast clipping.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    p_low, p_high = np.percentile(img, percentile_clip)
    clipped = np.clip(img, p_low, p_high)
    
    norm = cv2.normalize(clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm

def isolate_nodule_roi(image, mask):
    """
    Applies segmentation mask to isolate nodule region of interest.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    nodule_roi = cv2.bitwise_and(image, image, mask=binary_mask)
    
    # Extract bounding box coordinates around the nodule mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped_roi = nodule_roi[y:y+h, x:x+w]
        cropped_mask = binary_mask[y:y+h, x:x+w]
        return cropped_roi, cropped_mask, (x, y, w, h)
        
    return nodule_roi, binary_mask, (0, 0, image.shape[1], image.shape[0])
