def compute_risk(bbox, depth, frame_shape):
    h,w = frame_shape[:2]
    area_norm = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) / (w*h + 1e-6)
    cx = (bbox[0]+bbox[2]) / 2.0
    cy = (bbox[1]+bbox[3]) / 2.0
    bottom_prox = cy / h
    depth_norm = 1.0 / (depth + 1e-6) if depth>1e-6 else 1.0
    risk = 0.5*area_norm + 0.3*bottom_prox + 0.2*depth_norm
    return max(0.0, min(1.0, risk))

def compute_ttc(depth, vy):
    if depth <= 0: return None
    dist = 1.0 / depth
    speed = max(1e-3, abs(vy))
    return float(dist / speed)
