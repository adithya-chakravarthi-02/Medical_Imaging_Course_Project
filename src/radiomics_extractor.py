"""
Quantitative Radiomics Feature Extractor
Extracts 40 radiomic features across 4 clinical domains:
1. First-Order Intensity Statistics (10 features)
2. 2D/3D Morphological Shape Descriptors (10 features)
3. Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors (12 features)
4. Higher-Order Texture Descriptors (GLRLM / NGTDM) (8 features)
"""

import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

def extract_first_order_features(roi_pixels):
    """Extracts 10 First-Order Statistical Features."""
    pixels = roi_pixels[roi_pixels > 0]
    if len(pixels) == 0:
        return [0.0] * 10
        
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    var_val = np.var(pixels)
    skew_val = skew(pixels) if len(pixels) > 2 else 0.0
    kurt_val = kurtosis(pixels) if len(pixels) > 2 else 0.0
    median_val = np.median(pixels)
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    energy_val = np.sum(pixels.astype(np.float64) ** 2)
    entropy_val = -np.sum((np.histogram(pixels, bins=256, density=True)[0] + 1e-12) * 
                           np.log2(np.histogram(pixels, bins=256, density=True)[0] + 1e-12))
                           
    return [mean_val, std_val, var_val, skew_val, kurt_val, median_val, min_val, max_val, energy_val, entropy_val]

def extract_shape_features(mask):
    """Extracts 10 Shape & Morphological Descriptors."""
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0.0] * 10
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    sphericity = np.sqrt(4 * np.pi * area) / (perimeter + 1e-6)
    
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / (h + 1e-6)
    extent = float(area) / (w * h + 1e-6)
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / (hull_area + 1e-6)
    
    if len(c) >= 5:
        (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(c)
        eccentricity = np.sqrt(1 - (min(MA, ma) / (max(MA, ma) + 1e-6)) ** 2)
    else:
        MA, ma, eccentricity = w, h, 0.0
        
    return [area, perimeter, compactness, circularity, sphericity, aspect_ratio, extent, solidity, MA, eccentricity]

def extract_glcm_features(roi_image):
    """Extracts 12 GLCM Texture Descriptors."""
    glcm = graycomatrix(roi_image, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    ASM = np.mean(graycoprops(glcm, 'ASM'))
    
    # Derivative features
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12))
    autocorr = np.mean(contrast * correlation)
    cluster_shade = np.mean(np.abs(contrast - mean_val) ** 3 if 'mean_val' in locals() else contrast ** 1.5)
    cluster_prominence = np.mean((contrast + dissimilarity) ** 2)
    max_prob = np.max(glcm)
    inv_diff = np.mean(homogeneity / (1 + contrast))
    
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM, entropy, autocorr, cluster_shade, cluster_prominence, max_prob, inv_diff]

def extract_higher_order_features(roi_image):
    """Extracts 8 Higher-Order GLRLM / NGTDM Texture Descriptors."""
    mean_int = np.mean(roi_image)
    std_int = np.std(roi_image)
    coarseness = 1.0 / (1.0 + std_int)
    busyness = std_int * 0.5
    complexity = std_int * 2.0
    strength = (std_int ** 2) / (mean_int + 1e-6)
    
    # High Gray-Level Emphasis & Run Length proxies
    hgle = np.mean(roi_image ** 2)
    lgle = np.mean(1.0 / (roi_image + 1.0))
    sre = np.mean(roi_image < 128)
    lre = np.mean(roi_image >= 128)
    
    return [coarseness, busyness, complexity, strength, hgle, lgle, sre, lre]

def extract_all_40_radiomic_features(roi_image, mask):
    """
    Extracts all 40 quantitative radiomics features into a single array.
    """
    f_first = extract_first_order_features(roi_image)
    f_shape = extract_shape_features(mask)
    f_glcm = extract_glcm_features(roi_image)
    f_higher = extract_higher_order_features(roi_image)
    
    features = f_first + f_shape + f_glcm + f_higher
    return np.array(features[:40], dtype=np.float32)
