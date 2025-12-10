"""
Optimized Floor Segmentation with Resolution Control
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import time

class FloorSegmentationModel:
    """
    Optimized semantic segmentation with:
    - Explicit resolution control
    - Temporal caching
    - Real FPS metrics
    """
    
    DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL, 
                 device: str = "cpu",
                 target_size: Tuple[int, int] = (256, 192)):  # NEW!
        """
        Args:
            target_size: (width, height) to resize input BEFORE passing to model
                        Default 256×192 for speed/accuracy balance
        """
        self.device = torch.device(device)
        self.target_size = target_size  # User-controlled resolution
        
        cache_dir = os.environ.get("HF_HOME")
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        # CRITICAL: Override processor's default size
        self.processor.size = {"height": target_size[1], "width": target_size[0]}
        
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Floor class IDs (same as before)
        id2label = getattr(self.model.config, "id2label", {})
        safe_path_keywords = (
            "sidewalk", "path", "road", "floor", "pavement",
            "rug", "carpet", "mat", "concrete", "asphalt"
        )
        off_path_keywords = (
            "grass", "field", "earth", "sand", "dirt", "gravel", "ground"
        )
        
        self.safe_path_ids: List[int] = [
            idx for idx, label in id2label.items()
            if any(word in label.lower() for word in safe_path_keywords)
        ]
        self.off_path_ids: List[int] = [
            idx for idx, label in id2label.items()
            if any(word in label.lower() for word in off_path_keywords)
            and idx not in self.safe_path_ids
        ]
        
        # Default: combine both (indoor mode)
        self.floor_label_ids: List[int] = self.safe_path_ids + self.off_path_ids
        self.outdoor_mode = False  # Can be toggled
        
        if not self.floor_label_ids:
            self.floor_label_ids = list(id2label.keys())
        
        print(f"  Safe path IDs: {self.safe_path_ids}")
        print(f"  Off-path IDs: {self.off_path_ids}")
        
        # Temporal caching
        self.cached_mask = None
        self.last_inference_time = 0
        self.cache_interval = 0.3  # 300ms cache - more frequent updates
        
        # Metrics
        self.total_calls = 0
        self.cache_hits = 0
        self.inference_times = []
        
        print(f"✅ SegFormer loaded | Target size: {target_size} | Device: {device}")
    
    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray, force_compute: bool = False) -> np.ndarray:
        """
        Return boolean mask with temporal caching.
        
        Args:
            frame_bgr: Input BGR frame (any size)
            force_compute: Skip cache, force new inference
        
        Returns:
            Boolean mask at ORIGINAL frame size
        """
        self.total_calls += 1
        
        if frame_bgr is None:
            return None
        
        # Check cache
        time_since_last = time.time() - self.last_inference_time
        if not force_compute and self.cached_mask is not None and time_since_last < self.cache_interval:
            self.cache_hits += 1
            return self.cached_mask  # USE CACHE
        
        # Run inference
        inference_start = time.time()
        
        # Resize to target size BEFORE model
        frame_resized = cv2.resize(frame_bgr, self.target_size)
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Process (now at target_size, not default 512×512!)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Upsample to ORIGINAL frame size (not intermediate size)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=frame_bgr.shape[:2],  # Original size
            mode="bilinear",
            align_corners=False,
        )
        
        pred_labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Select IDs based on mode
        if self.outdoor_mode:
            # Outdoor: only detect safe paths (sidewalk, road), NOT grass
            active_ids = self.safe_path_ids
        else:
            # Indoor: detect all floor-like surfaces
            active_ids = self.floor_label_ids
        
        mask = np.isin(pred_labels, active_ids).astype(np.bool_)
        
        # Cache result
        self.cached_mask = mask
        self.last_inference_time = time.time()
        
        inference_time = (time.time() - inference_start) * 1000
        self.inference_times.append(inference_time)
        
        return mask
    
    def set_outdoor_mode(self, enabled: bool):
        """
        Toggle outdoor mode.
        
        Outdoor mode: Only detect sidewalk/road/pavement (safe paths)
        Indoor mode: Detect all floor-like surfaces including grass
        """
        self.outdoor_mode = enabled
        self.cached_mask = None  # Clear cache
        mode_str = "OUTDOOR (sidewalk only)" if enabled else "INDOOR (all floor)"
        print(f"Floor detection mode: {mode_str}")
    
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
            'outdoor_mode': self.outdoor_mode,
        }
