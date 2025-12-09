from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import io
import base64
import numpy as np
import cv2
from pathlib import Path
import time
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

# Import improved modules
import sys
# Add ml directory to path to import modules
_current_file = Path(__file__).resolve()
_ml_dir = _current_file.parent.parent  # Go up from ml/server/app.py to ml/
if str(_ml_dir) not in sys.path:
    sys.path.insert(0, str(_ml_dir))

try:
    from modules.infection_analysis import AdaptiveInfectionAnalyzer, ThresholdMethod
    from modules.region_visualization import GradCAMVisualizer
    print("[DEBUG] Successfully imported improved modules")
except ImportError as e:
    print(f"[WARNING] Failed to import improved modules: {e}")
    print(f"[WARNING] Paths in sys.path: {sys.path[:3]}")
    print(f"[WARNING] Falling back to basic implementation")
    # Fallback: define basic versions if imports fail
    class ThresholdMethod:
        PERCENTILE = "percentile"
    
    class AdaptiveInfectionAnalyzer:
        def __init__(self, method=None, percentile=60.0):
            self.method = method
            self.percentile = percentile
        
        def compute_region_infection(self, heatmap, regions, lung_mask, adaptive=True):
            # Fallback to basic normalization and thresholding
            # Normalize heatmap
            h_min = heatmap.min()
            h_max = heatmap.max()
            if h_max > h_min:
                heatmap_norm = (heatmap - h_min) / (h_max - h_min + 1e-8)
            else:
                heatmap_norm = np.zeros_like(heatmap)
            heatmap_norm = np.clip(heatmap_norm, 0, 1).astype(np.float32)
            
            # Calculate adaptive threshold
            if adaptive and heatmap_norm.size > 0:
                mean_val = np.mean(heatmap_norm)
                std_val = np.std(heatmap_norm)
                threshold = np.clip(mean_val + 0.5 * std_val, 0.3, 0.7)
            else:
                threshold = 0.45
            infection_mask = (heatmap_norm > threshold).astype(np.float32)
            
            # Create a simple result structure
            class SimpleResult:
                def __init__(self, name, total, infected, pct):
                    self.region_name = name
                    self.total_pixels = total
                    self.infected_pixels = infected
                    self.infection_percentage = pct
            
            class SimpleMetrics:
                def __init__(self):
                    self.region_results = {}
                    self.total_infection_percentage = 0.0
                    self.threshold_used = threshold
                    self.method_used = "fallback"
                    self.heatmap_stats = {
                        'min': float(heatmap_norm.min()),
                        'max': float(heatmap_norm.max()),
                        'mean': float(heatmap_norm.mean()),
                        'std': float(heatmap_norm.std()),
                        'median': float(np.median(heatmap_norm))
                    }
            
            metrics = SimpleMetrics()
            total_lung = 0
            total_infected = 0
            
            for region_name, region_mask in regions.items():
                valid_region = (region_mask & lung_mask).astype(np.float32)
                total_pixels = int(np.sum(valid_region))
                infected_pixels = int(np.sum(infection_mask * valid_region))
                percentage = (infected_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
                percentage = min(float(percentage), 100.0)
                metrics.region_results[region_name] = SimpleResult(region_name, total_pixels, infected_pixels, percentage)
                total_lung += total_pixels
                total_infected += infected_pixels
            
            if total_lung > 0:
                metrics.total_infection_percentage = min((total_infected / total_lung * 100.0), 100.0)
            
            return metrics
    
    class GradCAMVisualizer:
        def __init__(self, colormap='jet', alpha=0.5):
            self.colormap = colormap
            self.alpha = alpha
        
        def create_heatmap_visualization(self, heatmap):
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_8bit = (heatmap_norm * 255).astype(np.uint8)
            return cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        def create_overlay(self, original, heatmap, alpha=None):
            blend_alpha = alpha if alpha is not None else self.alpha
            heatmap_colored = self.create_heatmap_visualization(heatmap)
            if original.shape[:2] != heatmap_colored.shape[:2]:
                heatmap_colored = cv2.resize(heatmap_colored, (original.shape[1], original.shape[0]))
            return cv2.addWeighted(original, 1 - blend_alpha, heatmap_colored, blend_alpha, 0).astype(np.uint8)
        
        def create_multiregion_overlay_enhanced(self, original, heatmap, regions, infection_results=None):
            # Simple fallback implementation
            return self.create_overlay(original, heatmap, alpha=0.5)

# For testing: if TESTING in env, use CPU and mock model
if os.environ.get("TESTING"):
    DEVICE = torch.device("cpu")
    MODEL_PATH = Path(__file__).parent / "test_model.pt"
    
    # Create a dummy model for testing
    dummy_model = models.resnet18(weights=None)
    dummy_model.fc = nn.Linear(dummy_model.fc.in_features, 2)
    torch.save(dummy_model.state_dict(), MODEL_PATH)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = Path(__file__).parent.parent / "runs" / "medmnist_pneumonia" / "best.pt"

# Global model state
MODEL = None
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Pneumonia"]

# ===== ANATOMICAL LUNG SEGMENTATION =====
class LungRegionSegmenter:
    """
    Segment lungs into 4 anatomical regions:
    - Left Upper Lobe (LUL)
    - Left Lower Lobe (LLL)
    - Right Upper Lobe (RUL)
    - Right Lower Lobe (RLL)
    """
    
    @staticmethod
    def segment_lungs(img_np: np.ndarray) -> np.ndarray:
        """
        Fast heuristic lung segmentation using threshold + morphology.
        Returns binary mask (0/1) of lungs.
        """
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_np
        
        # Histogram equalization
        img_eq = cv2.equalizeHist(img_gray)
        
        # Invert (lungs are dark in X-rays)
        img_inv = 255 - img_eq
        img_inv_float = img_inv.astype(np.float32)
        img_inv = cv2.normalize(img_inv_float, img_inv_float, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Otsu thresholding
        _, th = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(th, dtype=np.float32)
        
        # Keep 2 largest (left & right lungs)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Create mask
        mask = np.zeros_like(th, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return (mask > 127).astype(np.float32)
    
    @staticmethod
    def segment_regions(lung_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Divide lungs into 4 anatomical regions.
        Uses anatomical landmarks: horizontal midline divides upper/lower lobes,
        vertical midline divides left/right lungs.
        
        Returns dict with region masks:
        - left_upper: LUL
        - left_lower: LLL
        - right_upper: RUL
        - right_lower: RRL
        """
        h, w = lung_mask.shape
        
        # Find lung bounding box for accurate division
        y_indices, x_indices = np.where(lung_mask > 0.5)
        
        if len(y_indices) == 0:
            # Empty mask, return empty regions
            return {
                "left_upper": np.zeros_like(lung_mask),
                "left_lower": np.zeros_like(lung_mask),
                "right_upper": np.zeros_like(lung_mask),
                "right_lower": np.zeros_like(lung_mask),
            }
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Calculate division points
        # Horizontal division: typically at ~40% from top (upper lobe smaller)
        y_div = y_min + int((y_max - y_min) * 0.42)
        
        # Vertical division: at center
        x_div = (x_min + x_max) // 2
        
        # Create region masks
        regions = {
            "left_upper": np.zeros_like(lung_mask),
            "left_lower": np.zeros_like(lung_mask),
            "right_upper": np.zeros_like(lung_mask),
            "right_lower": np.zeros_like(lung_mask),
        }
        
        # Assign pixels to regions (vectorized for speed)
        lung_pixels = lung_mask > 0.5
        left_mask = np.zeros((h, w), dtype=bool)
        left_mask[:, :x_div] = True
        
        # Create region masks using vectorized operations
        regions["left_upper"] = (lung_pixels & left_mask & (np.arange(h)[:, None] < y_div)).astype(np.float32)
        regions["left_lower"] = (lung_pixels & left_mask & (np.arange(h)[:, None] >= y_div)).astype(np.float32)
        regions["right_upper"] = (lung_pixels & ~left_mask & (np.arange(h)[:, None] < y_div)).astype(np.float32)
        regions["right_lower"] = (lung_pixels & ~left_mask & (np.arange(h)[:, None] >= y_div)).astype(np.float32)
        
        return regions


# ===== INFECTION ANALYSIS =====
class InfectionAnalyzer:
    """Analyze infection patterns in lungs using adaptive thresholding."""
    
    INFECTION_THRESHOLD = 0.45  # Threshold for infection detection
    
    @staticmethod
    def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range."""
        h_min = heatmap.min()
        h_max = heatmap.max()
        
        if h_max > h_min:
            normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)
        else:
            normalized = np.zeros_like(heatmap)
        
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    @staticmethod
    def get_adaptive_threshold(heatmap: np.ndarray) -> float:
        """
        Calculate adaptive threshold based on heatmap statistics.
        Uses mean + 0.5*std for dynamic thresholding.
        """
        if heatmap.size == 0:
            return 0.45
        
        mean_val = np.mean(heatmap)
        std_val = np.std(heatmap)
        
        # Adaptive threshold: mean + 0.5*std, clamped to [0.3, 0.7]
        threshold = np.clip(mean_val + 0.5 * std_val, 0.3, 0.7)
        return float(threshold)
    
    @staticmethod
    def compute_region_infection(
        heatmap: np.ndarray,
        regions: Dict[str, np.ndarray],
        adaptive: bool = True
    ) -> Dict[str, float]:
        """
        Compute infection percentage in each region.
        
        Args:
            heatmap: Normalized CAM heatmap [0, 1]
            regions: Dict of region masks
            adaptive: Use adaptive threshold if True
            
        Returns:
            Dict with infection percentages for each region + total
        """
        # Normalize heatmap
        heatmap_norm = InfectionAnalyzer.normalize_heatmap(heatmap)
        
        # Get threshold
        threshold = InfectionAnalyzer.get_adaptive_threshold(heatmap_norm) if adaptive else 0.45
        
        # Create infection mask
        infection_mask = (heatmap_norm > threshold).astype(np.float32)
        
        results = {}
        total_lung_pixels = 0
        total_infected_pixels = 0
        
        for region_name, region_mask in regions.items():
            lung_pixels = np.sum(region_mask)
            infected_pixels = np.sum(infection_mask * region_mask)
            
            if lung_pixels > 0:
                percentage = (infected_pixels / lung_pixels) * 100.0
            else:
                percentage = 0.0
            
            # Ensure percentage doesn't exceed 100%
            percentage = min(float(percentage), 100.0)
            results[region_name] = round(percentage, 2)
            
            total_lung_pixels += lung_pixels
            total_infected_pixels += infected_pixels
        
        # Calculate total infection percentage
        if total_lung_pixels > 0:
            total_pct = (total_infected_pixels / total_lung_pixels) * 100.0
        else:
            total_pct = 0.0
        
        results["total_infection"] = round(min(float(total_pct), 100.0), 2)
        
        return results
    
    @staticmethod
    def create_colored_overlay(
        img_bgr: np.ndarray,
        heatmap: np.ndarray,
        regions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Create colored overlay highlighting each region differently.
        
        Colors:
        - Blue: Left Upper Lobe (LUL)
        - Green: Left Lower Lobe (LLL)
        - Orange: Right Upper Lobe (RUL)
        - Red: Right Lower Lobe (RLL)
        """
        # Normalize heatmap
        heatmap_norm = InfectionAnalyzer.normalize_heatmap(heatmap)
        
        # Create output image
        overlay = img_bgr.copy().astype(np.float32)
        h, w = heatmap_norm.shape
        
        # Define colors for each region (BGR format)
        region_colors = {
            "left_upper": (255, 0, 0),      # Blue
            "left_lower": (0, 255, 0),      # Green
            "right_upper": (0, 165, 255),   # Orange
            "right_lower": (0, 0, 255),     # Red
        }
        
        # Apply colored heatmap to each region
        for region_name, region_mask in regions.items():
            color = region_colors.get(region_name, (0, 255, 0))
            
            # Apply color with intensity based on heatmap value
            for c in range(3):
                intensity = heatmap_norm * region_mask
                # Blend: 30% color + 70% original
                overlay[:, :, c] = (
                    overlay[:, :, c] * (1 - 0.3 * region_mask) +
                    color[c] * 0.3 * intensity
                )
        
        # Ensure values are in valid range
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay


class ResNet18Pneumonia(nn.Module):
    """Custom ResNet18 for pneumonia detection"""
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        # If model_path is provided, load it directly
        if model_path and Path(model_path).exists():
            self.model = torch.load(model_path)
            if isinstance(self.model, dict):
                self.model = self.model.get('model', self.model)
        else:
            # Initialize new model
            self.model = models.resnet18(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 2)  # binary: normal/pneumonia
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    @property
    def layer4(self) -> nn.Module:
        return self.model.layer4

def load_model() -> ResNet18Pneumonia:
    """Load the PyTorch model from checkpoint."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at {MODEL_PATH}")
    
    # First try loading as full model
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(model, ResNet18Pneumonia):
            model = model.to(DEVICE)
            model.eval()
            return model
    except Exception:
        pass
        
    # If that fails, try loading as state dict
    model = ResNet18Pneumonia()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            # Remove "model." prefix if present
            checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            
    # Try direct load into inner model
    model.model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(DEVICE)
    model.eval()
    return model

def get_gradcam(
    model: ResNet18Pneumonia,
    image: Tensor,
    target_layer: Optional[nn.Module] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Generate proper GradCAM activation maps for the predicted class.
    Uses improved gradient computation and normalization.
    Returns tuple: (activation_map as 2D array, feature_importances)
    """
    if not isinstance(model, ResNet18Pneumonia):
        raise TypeError("Model must be ResNet18Pneumonia instance")
    
    # For testing mode with dummy model, generate synthetic but realistic CAM
    if os.environ.get("TESTING"):
        # Create a more realistic CAM with proper size
        cam_grid = np.zeros((7, 7), dtype=np.float32)
        
        # Create Gaussian blobs at specific positions
        for i in range(7):
            for j in range(7):
                # Center blob (pneumonia area)
                dist_center = np.sqrt((i - 3)**2 + (j - 3)**2)
                cam_grid[i, j] += 0.8 * np.exp(-dist_center**2 / 2)
                
                # Right side blob
                dist_right = np.sqrt((i - 2)**2 + (j - 5)**2)
                cam_grid[i, j] += 0.5 * np.exp(-dist_right**2 / 3)
                
                # Left side blob
                dist_left = np.sqrt((i - 4)**2 + (j - 1)**2)
                cam_grid[i, j] += 0.4 * np.exp(-dist_left**2 / 4)
        
        # Add some noise for realism
        cam_grid += np.random.randn(7, 7) * 0.05
        cam_grid = np.clip(cam_grid, 0, 1)
        
        # Upsample to image size
        cam_upsampled = cv2.resize(cam_grid, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        importances = [0.1] * 512  # dummy importance weights
        return cam_upsampled, importances
        
    if target_layer is None:
        # Get the last conv layer (typically layer4 in ResNet)
        for name, layer in reversed(list(model.model.named_modules())):
            if isinstance(layer, nn.Conv2d) and 'layer4' in name:
                target_layer = layer
                break
        
        # Fallback to any conv layer
        if target_layer is None:
            for name, layer in reversed(list(model.model.named_modules())):
                if isinstance(layer, nn.Conv2d):
                    target_layer = layer
                    break
        
        if not target_layer:
            raise ValueError("Could not find convolutional layer for GradCAM")
    
    # Register hooks for feature and gradient capture
    features: List[Tensor] = []
    grads: List[Tensor] = []
    
    def save_features(_module: nn.Module, _input: Tuple[Tensor, ...], output: Tensor) -> None:
        features.append(output.detach())
    
    def save_grads(_module: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        if isinstance(grad_out, tuple) and grad_out and isinstance(grad_out[0], torch.Tensor):
            grads.append(grad_out[0].detach())
        elif isinstance(grad_out, torch.Tensor):
            grads.append(grad_out.detach())
    
    handles = [
        target_layer.register_forward_hook(save_features),
        target_layer.register_full_backward_hook(save_grads)
    ]
    
    try:
        # Forward pass
        logits = model(image)
        pred_class = logits.argmax(dim=1)
        
        # Backward pass for predicted class
        model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, pred_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)
        
        if not features or not grads:
            raise RuntimeError("Failed to capture features or gradients for GradCAM")
        
        # Generate activation map using proper GradCAM formula
        feature_map = features[0].squeeze(0)  # Remove batch dim: NCHW -> CHW
        grad_map = grads[0].squeeze(0)  # Remove batch dim
        
        # Global average pooling of gradients (GradCAM formula)
        # weights = average of gradients over spatial dimensions
        weights = grad_map.mean(dim=(1, 2))  # [C] - channel-wise importance
        
        # Weighted combination of feature maps (vectorized for speed)
        # CAM = sum(weights[i] * feature_map[i] for i in channels)
        # Use einsum for faster computation
        cam = torch.einsum('c,cwh->wh', weights, feature_map)
        
        # Apply ReLU to keep only positive contributions
        cam = torch.relu(cam)
        
        # Normalize to [0, 1] with better handling
        if cam.numel() > 0:
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            else:
                cam = torch.zeros_like(cam)
        
        # Upsample CAM to image size using bicubic interpolation for better quality
        cam_np = cam.detach().cpu().numpy()
        cam_upsampled = cv2.resize(cam_np, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Ensure values are in [0, 1] range
        cam_upsampled = np.clip(cam_upsampled, 0, 1).astype(np.float32)
        
        return cam_upsampled, weights.detach().cpu().tolist()
    
    finally:
        # Cleanup hooks
        for handle in handles:
            handle.remove()


# API response models
class RegionInfection(BaseModel):
    left_upper: float
    left_lower: float
    right_upper: float
    right_lower: float
    total_infection: float

class InferenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    prediction: str
    confidence: float
    processing_time: float
    gradcam_weights: List[float]
    feature_importances: List[float]
    region_infection: RegionInfection
    gradcam_overlay: Optional[str] = None

# Startup/Shutdown events using lifespan context manager
@asynccontextmanager
async def lifespan(app_instance):
    # Startup
    global MODEL
    try:
        print("=" * 60)
        print("[STARTUP] Loading model...")
        print(f"[STARTUP] Model path: {MODEL_PATH}")
        print(f"[STARTUP] Model exists: {MODEL_PATH.exists()}")
        print(f"[STARTUP] Device: {DEVICE}")
        
        if not MODEL_PATH.exists():
            print(f"[WARNING] Model file not found at {MODEL_PATH}")
            # Attempt to download model if MODEL_URL is provided in env
            model_url = os.environ.get("MODEL_URL")
            if model_url:
                print(f"[STARTUP] MODEL_URL provided, attempting to download from: {model_url}")
                try:
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    from urllib.request import urlopen, Request
                    import shutil
                    req = Request(model_url, headers={"User-Agent": "pneumoai/1.0"})
                    with urlopen(req, timeout=120) as r, open(MODEL_PATH, "wb") as out_f:
                        shutil.copyfileobj(r, out_f)
                    print(f"[STARTUP] Model downloaded to {MODEL_PATH}")
                except Exception as e:
                    print(f"[ERROR] Failed to download model: {e}")
            if not MODEL_PATH.exists():
                print("[WARNING] Server will start but inference will fail until model is available")
                MODEL = None
            else:
                MODEL = load_model()
                print(f"[STARTUP] Model loaded successfully on {DEVICE}")
                print(f"[STARTUP] Model type: {type(MODEL)}")
        else:
            MODEL = load_model()
            print(f"[STARTUP] Model loaded successfully on {DEVICE}")
            print(f"[STARTUP] Model type: {type(MODEL)}")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        print("[WARNING] Server will start but inference will fail")
        MODEL = None
    
    yield
    
    # Shutdown
    print("[DEBUG] Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="PneumoAI Inference API - Region-wise Analysis",
    lifespan=lifespan
)

# CORS for local development
# Note: EchoCORSMiddleware below will handle dynamic origins
# This list is for explicit origins that should always be allowed
# Configure CORS origins from environment for production
allowed_env = os.environ.get("ALLOWED_ORIGINS", "")
allow_all = os.environ.get("CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes")
if allow_all:
    allow_origins = ["*"]
elif allowed_env:
    allow_origins = [o.strip() for o in allowed_env.split(",") if o.strip()]
else:
    # default development origins
    allow_origins = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://[::1]:5173",
        "http://[::1]:5174",
        "http://[::1]:5175",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class EchoCORSMiddleware(BaseHTTPMiddleware):
    """
    Additional CORS middleware to echo the origin for better compatibility.
    This allows any localhost origin dynamically, which is useful during development.
    """
    async def dispatch(self, request, call_next):
        response: StarletteResponse = await call_next(request)
        origin = request.headers.get("origin")
        
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            response = StarletteResponse()
            if origin and self._is_allowed_origin(origin):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Max-Age"] = "3600"
            response.status_code = 200
            return response
        
        # For regular requests, add CORS headers if origin is allowed
        if origin and self._is_allowed_origin(origin):
            response.headers.setdefault("Access-Control-Allow-Origin", origin)
            response.headers.setdefault("Access-Control-Allow-Credentials", "true")
            response.headers.setdefault("Vary", "Origin")
        
        return response
    
    @staticmethod
    def _is_allowed_origin(origin: str) -> bool:
        """Check if origin is allowed (localhost variants for development)"""
        allowed_patterns = [
            "http://localhost:",
            "http://127.0.0.1:",
            "http://[::1]:",
        ]
        return any(origin.startswith(pattern) for pattern in allowed_patterns)

app.add_middleware(EchoCORSMiddleware)

@app.get("/")
async def root():
    """Root endpoint to verify server is running"""
    return JSONResponse(
        status_code=200,
        content={
            "message": "PneumoAI Backend Server is running",
            "endpoints": {
                "health": "/api/health",
                "infer": "/api/infer"
            }
        }
    )

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "message": "PneumoAI API",
            "version": "1.0",
            "endpoints": {
                "health": "/api/health",
                "infer": "/api/infer"
            }
        }
    )

@app.post("/api/infer")
async def infer(request: Request, file: UploadFile = File(...)):
    """
    Analyze chest X-ray or CT scan for pneumonia.
    
    Returns:
    - prediction: "pneumonia" or "normal"
    - confidence: prediction confidence (0-1)
    - processing_time: inference time in seconds
    - gradcam_weights: flattened CAM heatmap
    - feature_importances: channel importance weights
    - region_infection: infection percentages for each lung region
    - gradcam_overlay: base64 encoded colored overlay image
    """
    try:
        print(f"[DEBUG] Received inference request from {request.client}")
        print(f"[DEBUG] File: {file.filename}, content_type: {file.content_type}")
        
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        
        if MODEL is None:
            raise HTTPException(503, "Model not loaded. Please check server logs and ensure model file exists.")
        if not isinstance(MODEL, ResNet18Pneumonia):
            raise HTTPException(500, f"Model not properly initialized. Got type: {type(MODEL)}")
        
        start_time = time.perf_counter()
        step_times = {}
        
        # Read and preprocess image
        step_start = time.perf_counter()
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(pil_image)
        step_times['image_load'] = time.perf_counter() - step_start
        print(f"[PERF] Image load: {step_times['image_load']:.3f}s")
        
        # PyTorch transform pipeline
        step_start = time.perf_counter()
        transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_image)
        tensor = img_tensor.unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)
        step_times['preprocessing'] = time.perf_counter() - step_start
        print(f"[PERF] Preprocessing: {step_times['preprocessing']:.3f}s")
        
        # Model inference (no grad needed for forward pass)
        step_start = time.perf_counter()
        with torch.no_grad():
            logits = MODEL(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0][pred_idx].item())
        step_times['inference'] = time.perf_counter() - step_start
        print(f"[PERF] Model inference: {step_times['inference']:.3f}s")
        print(f"[DEBUG] Prediction: {CLASS_NAMES[pred_idx]}, Confidence: {confidence:.4f}")
        
        # Generate GradCAM (needs gradients)
        step_start = time.perf_counter()
        tensor.requires_grad_(True)  # Re-enable for GradCAM
        cam_map, importances = get_gradcam(MODEL, tensor)
        step_times['gradcam'] = time.perf_counter() - step_start
        print(f"[PERF] GradCAM: {step_times['gradcam']:.3f}s")
        print(f"[DEBUG] GradCAM generated - shape: {cam_map.shape}")
        
        # Segment lungs into anatomical regions (optimize with smaller image)
        step_start = time.perf_counter()
        # Skip complex segmentation for speed - use simple heuristic instead
        # Simple approach: assume regions based on CAM distribution
        img_resized = cv2.resize(img_np, IMG_SIZE, interpolation=cv2.INTER_AREA)
        h, w = IMG_SIZE
        
        # Create simple region masks based on image quadrants
        regions = {
            "left_upper": np.ones((h, w), dtype=np.float32),
            "left_lower": np.ones((h, w), dtype=np.float32),
            "right_upper": np.ones((h, w), dtype=np.float32),
            "right_lower": np.ones((h, w), dtype=np.float32),
        }
        
        # Apply simple mask to quadrants
        regions["left_upper"][:, w//2:] = 0
        regions["left_upper"][h//2:, :] = 0
        
        regions["left_lower"][:, w//2:] = 0
        regions["left_lower"][:h//2, :] = 0
        
        regions["right_upper"][:, :w//2] = 0
        regions["right_upper"][h//2:, :] = 0
        
        regions["right_lower"][:, :w//2] = 0
        regions["right_lower"][:h//2, :] = 0
        
        step_times['segmentation'] = time.perf_counter() - step_start
        print(f"[PERF] Segmentation: {step_times['segmentation']:.3f}s")
        
        # Analyze infection in each region (simplified)
        step_start = time.perf_counter()
        threshold = 0.45
        infection_mask = (cam_map > threshold).astype(np.float32)
        
        infection_pcts = {}
        for region_name, region_mask in regions.items():
            region_pixels = np.sum(region_mask)
            if region_pixels > 0:
                infected_pixels = np.sum(infection_mask * region_mask)
                pct = min((infected_pixels / region_pixels) * 100.0, 100.0)
            else:
                pct = 0.0
            infection_pcts[region_name] = round(float(pct), 2)
        
        # Calculate total
        total_pixels = np.sum(np.ones_like(cam_map))
        total_infected = np.sum(infection_mask)
        total_pct = min((total_infected / total_pixels) * 100.0, 100.0) if total_pixels > 0 else 0.0
        infection_pcts["total_infection"] = round(float(total_pct), 2)
        
        step_times['infection_analysis'] = time.perf_counter() - step_start
        print(f"[PERF] Infection analysis: {step_times['infection_analysis']:.3f}s")
        
        # Create visualizations (optimize encoding)
        step_start = time.perf_counter()
        # Use turbo colormap and stronger alpha + smoothing to match UI style
        visualizer = GradCAMVisualizer(colormap='turbo', alpha=0.75, smoothing_sigma=7.0)
        
        # Simple heatmap overlay
        heatmap_overlay = visualizer.create_overlay(img_resized, cam_map, alpha=0.6)
        
        # Use JPEG encoding for faster processing
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        _, heatmap_buffer = cv2.imencode('.jpg', heatmap_overlay, encode_params)
        heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(heatmap_buffer.tobytes()).decode('utf-8')
        
        # Use same image for overlay for speed
        overlay_base64 = heatmap_base64

        # Additional visualizations: multiregion overlay, region boundaries, labeled overlay, infection mask
        try:
            # Ensure enhanced visualizer parameters for the extra assets
            visualizer = GradCAMVisualizer(colormap='turbo', alpha=0.75, smoothing_sigma=7.0)

            # Prepare infection_results structure expected by visualizer
            infection_results_struct = {}
            for k in ['left_upper', 'left_lower', 'right_upper', 'right_lower']:
                infection_results_struct[k] = {
                    'infection_percentage': float(infection_pcts.get(k, 0.0))
                }

            multiregion_img = visualizer.create_multiregion_overlay_enhanced(img_resized, cam_map, regions, infection_results_struct)
            boundaries_img = visualizer.create_region_boundaries(img_resized, regions)
            labeled_img = visualizer.create_labeled_overlay(img_resized, regions, infection_results_struct)
            heatmap_vis_img = visualizer.create_heatmap_visualization(cam_map)

            # Infection binary mask visualization (thresholded)
            thr = 0.45
            infection_mask = (cam_map > thr).astype(np.float32)
            # Color mask: red where infected
            mask_colored = np.zeros_like(img_resized, dtype=np.uint8)
            mask_colored[:, :, 2] = (infection_mask * 255).astype(np.uint8)
            # Blend mask with original for visibility
            mask_overlay = cv2.addWeighted(img_resized, 0.6, mask_colored, 0.4, 0)

            # Encode each additional image to JPEG base64
            def _encode_jpeg_b64(img_np):
                _, buf = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')

            multiregion_base64 = _encode_jpeg_b64(multiregion_img)
            boundaries_base64 = _encode_jpeg_b64(boundaries_img)
            labeled_base64 = _encode_jpeg_b64(labeled_img)
            heatmap_vis_base64 = _encode_jpeg_b64(heatmap_vis_img)
            mask_overlay_base64 = _encode_jpeg_b64(mask_overlay)
        except Exception as e:
            print(f"[WARNING] Failed to create extra visualizations: {e}")
            multiregion_base64 = None
            boundaries_base64 = None
            labeled_base64 = None
            heatmap_vis_base64 = None
            mask_overlay_base64 = None
        
        step_times['visualization'] = time.perf_counter() - step_start
        print(f"[PERF] Visualization: {step_times['visualization']:.3f}s")
        
        processing_time = time.perf_counter() - start_time
        
        # Build response with region infection data
        step_start = time.perf_counter()
        regions_list = [
            {"region": "Left-Upper Lung", "percentage": infection_pcts.get("left_upper", 0)},
            {"region": "Left-Lower Lung", "percentage": infection_pcts.get("left_lower", 0)},
            {"region": "Right-Upper Lung", "percentage": infection_pcts.get("right_upper", 0)},
            {"region": "Right-Lower Lung", "percentage": infection_pcts.get("right_lower", 0)},
        ]
        
        # Send full-resolution flattened CAM and include shape so frontend
        # can reconstruct exact size for high-quality visualizations.
        # Note: this increases payload size but yields accurate Grad-CAM overlays.
        cam_flat = cam_map.flatten().tolist()
        cam_shape = cam_map.shape  # (H, W)
        
        # Debug: report which visual assets were created (helps troubleshoot missing overlays)
        print(f"[DEBUG] Assets present - gradcam_heatmap_vis: {heatmap_vis_base64 is not None}, multiregion: {multiregion_base64 is not None}, region_boundaries: {boundaries_base64 is not None}, labeled: {labeled_base64 is not None}, infection_mask: {mask_overlay_base64 is not None}")

        body = {
            "prediction": "pneumonia" if pred_idx == 1 else "normal",
            "confidence": confidence,
            "processing_time": round(processing_time, 3),
            "gradcam_weights": cam_flat,
            "cam_shape": cam_shape,
            "feature_importances": importances[:100] if len(importances) > 100 else importances,  # Limit size
            "regions": regions_list,
            "total_infected_percentage": infection_pcts.get("total_infection", 0),
            "raw_total_infected_percentage": infection_pcts.get("total_infection", 0),
            "region_infection": {
                "left_upper": infection_pcts.get("left_upper", 0),
                "left_lower": infection_pcts.get("left_lower", 0),
                "right_upper": infection_pcts.get("right_upper", 0),
                "right_lower": infection_pcts.get("right_lower", 0),
                "total_infection": infection_pcts.get("total_infection", 0),
            },
            "gradcam_overlay": overlay_base64,
            "gradcam_heatmap": heatmap_base64,
            "gradcam_heatmap_vis": heatmap_vis_base64,
            "multiregion_overlay": multiregion_base64,
            "region_boundaries": boundaries_base64,
            "labeled_overlay": labeled_base64,
            "infection_mask_overlay": mask_overlay_base64,
            "performance_breakdown": {  # Optional: include timing breakdown
                "image_load": round(step_times.get('image_load', 0), 3),
                "preprocessing": round(step_times.get('preprocessing', 0), 3),
                "inference": round(step_times.get('inference', 0), 3),
                "gradcam": round(step_times.get('gradcam', 0), 3),
                "segmentation": round(step_times.get('segmentation', 0), 3),
                "infection_analysis": round(step_times.get('infection_analysis', 0), 3),
                "visualization": round(step_times.get('visualization', 0), 3),
            }
        }
        
        step_times['response_build'] = time.perf_counter() - step_start
        print(f"[PERF] Response build: {step_times['response_build']:.3f}s")
        print(f"[PERF] Total time: {processing_time:.3f}s")
        print(f"[PERF] Breakdown: {step_times}")
        
        origin = request.headers.get("origin")
        headers = {}
        if origin:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Vary"] = "Origin"

        return JSONResponse(status_code=200, content=body, headers=headers)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"[ERROR] Inference failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal Server Error: {str(e)}")

@app.get("/api/health")
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint - returns server status"""
    try:
        model_status = isinstance(MODEL, ResNet18Pneumonia) if MODEL is not None else False
        body = {
            "status": "ok",
            "device": str(DEVICE),
            "model_loaded": model_status,
            "message": "Backend is running"
        }
    except Exception as e:
        body = {
            "status": "error",
            "message": f"Health check error: {str(e)}"
        }
    
    origin = request.headers.get("origin")
    headers = {}
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"
    else:
        # Allow all origins if no origin header (for direct requests)
        headers["Access-Control-Allow-Origin"] = "*"
    
    return JSONResponse(status_code=200, content=body, headers=headers)

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting PneumoAI Backend Server")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {MODEL_PATH.exists()}")
    print("=" * 60)
    print("Server will be available at:")
    print("  - http://127.0.0.1:8000")
    print("  - http://localhost:8000")
    print("\nAPI endpoints:")
    print("  - GET  http://127.0.0.1:8000/api/health")
    print("  - POST http://127.0.0.1:8000/api/infer")
    print("=" * 60)
    print("\nStarting server...")
    try:
        # Use 0.0.0.0 to listen on all interfaces (allows connections from any IP)
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\n✗ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Check if port 8000 is already in use: netstat -ano | findstr :8000")
        print("2. Make sure all dependencies are installed: pip install -r ml/server/requirements.txt")
        print("3. Check the error message above for details")
