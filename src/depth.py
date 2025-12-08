import cv2
import torch
import numpy as np
class DepthEstimator:
    def __init__(self, device='cpu'):
        self.available = False
        self.device = device
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.midas.to(self.device).eval()
            self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            self.available = True
        except Exception:
            self.available = False
    def estimate(self, frame, bboxes):
        if self.available:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = self.transforms(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.midas(inp)
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False).squeeze().cpu().numpy()
            
            # If no bboxes, return full depth map
            if not bboxes:
                return pred
            
            depths = []
            h,w = img.shape[:2]
            for box in bboxes:
                x1,y1,x2,y2 = map(int, box)
                x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
                crop = pred[y1:y2+1, x1:x2+1]
                depths.append(float(np.median(crop)) if crop.size>0 else float(np.median(pred)))
            return depths
        else:
            h,w = frame.shape[:2]
            # If no bboxes, return dummy depth map
            if not bboxes:
                return np.ones((h, w), dtype=np.float32)
            
            depths = []
            for box in bboxes:
                area = max(1.0, (box[2]-box[0])*(box[3]-box[1]))
                depths.append(float(1.0 / (area / (w*h) + 1e-6)))
            return depths
