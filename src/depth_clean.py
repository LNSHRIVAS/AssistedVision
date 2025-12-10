"""
Optimized Depth Estimation with Resolution Control
"""

import cv2
import torch
import numpy as np
import time
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

class DepthEstimator:
    """
    Optimized depth estimation with:
    - Explicit resolution control
    - Temporal caching
    - Real metrics
    """
    
    def __init__(self,
                 device: str = 'cpu',
                 model_name: str = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
                 max_depth: float = None,
                 target_size: tuple = (256, 192)):  # NEW!
        """
        Args:
            target_size: (width, height) for model input
        """
        self.available = False
        self.device = device
        self.model = None
        self.processor = None
        self.is_metric = 'Metric' in model_name
        self.model_name = model_name
        self.target_size = target_size
        
        if not _HAS_TRANSFORMERS:
            print("[DepthEstimator] transformers library not available")
            return
        
        # Auto-detect max_depth
        if max_depth is None:
            max_depth = 80.0 if 'Outdoor' in model_name else 20.0
        
        try:
            print(f"[DepthEstimator] Loading {model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # CRITICAL: Override processor's default size
            self.processor.size = {"height": target_size[1], "width": target_size[0]}
            
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'max_depth'):
                config.max_depth = max_depth
            
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name, config=config)
            self.model.to(self.device).eval()
            self.available = True
            
            print(f"✅ Depth model loaded | Target size: {target_size} | Max depth: {max_depth}m")
        except Exception as exc:
            print(f"[DepthEstimator] Failed to load model: {exc}")
            self.available = False
        
        # Temporal caching
        self.cached_depth = None
        self.last_inference_time = 0
        self.cache_interval = 0.3  # 300ms cache for more stability
        
        # Metrics
        self.total_calls = 0
        self.cache_hits = 0
        self.inference_times = []
    
    def _predict_depth_map(self, frame: np.ndarray, force_compute: bool = False) -> np.ndarray:
        """
        Run depth model with caching.
        """
        if not self.available:
            return None
        
        self.total_calls += 1
        
        # Check cache
        time_since_last = time.time() - self.last_inference_time
        if not force_compute and self.cached_depth is not None and time_since_last < self.cache_interval:
            self.cache_hits += 1
            return self.cached_depth  # USE CACHE
        
        try:
            inference_start = time.time()
            
            # Resize BEFORE model
            frame_resized = cv2.resize(frame, self.target_size)
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Run model
            inputs = self.processor(images=img_rgb, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values)
                predicted_depth = outputs.predicted_depth
            
            # Resize to original frame size
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],  # Original size
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Cache
            self.cached_depth = depth
            self.last_inference_time = time.time()
            
            inference_time = (time.time() - inference_start) * 1000
            self.inference_times.append(inference_time)
            
            return depth
            
        except Exception as exc:
            print(f"[DepthEstimator] Prediction failed: {exc}")
            return None
    
    def estimate_full_map(self, frame: np.ndarray) -> np.ndarray:
        """Get full depth map"""
        return self._predict_depth_map(frame)
    
    def get_stats(self):
        """Get performance statistics"""
        cache_rate = (self.cache_hits / max(1, self.total_calls)) * 100
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        
        return {
            'total_calls': self.total_calls,
            'cache_hits': self.cache_hits,
            'cache_rate': cache_rate,
            'avg_inference_ms': avg_inference,
            'target_size': self.target_size,
        }
