def compute_risk(bbox, depth, frame_shape):
    h, w = frame_shape[:2]
    
    # Bounding box metrics
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    area_norm = (box_w * box_h) / (w * h + 1e-6)
    
    # Position metrics
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bottom_y = bbox[3]  # Bottom edge of box
    
    # Proximity based on vertical position (objects at bottom are closer)
    bottom_prox = bottom_y / h
    
    # Center proximity (objects in center of view are more relevant)
    center_dist = abs(cx - w/2) / (w/2)
    center_factor = 1.0 - (center_dist * 0.3)  # Up to 30% reduction for edge objects
    
    # Depth-based risk (smaller depth values = closer = higher risk)
    # Depth is normalized: typically 0.1-2.0, where lower = closer
    if depth > 0.01:
        depth_risk = min(1.0, 2.0 / depth)  # Closer objects get higher risk
    else:
        depth_risk = 1.0
    
    # Size-based risk (larger objects in frame = closer/more dangerous)
    size_risk = min(1.0, area_norm * 4.0)  # Scale up area importance
    
    # Combined risk calculation
    # Weight: 50% depth, 30% size, 20% position
    base_risk = (0.5 * depth_risk) + (0.3 * size_risk) + (0.2 * bottom_prox)
    
    # Apply center factor
    risk = base_risk * center_factor
    
    # Critical proximity boosters
    if area_norm > 0.2:  # Object covers >20% of frame
        risk *= 1.8
    elif area_norm > 0.1:  # Object covers >10% of frame
        risk *= 1.4
    
    # Bottom zone booster (likely on collision course)
    if bottom_prox > 0.7 and area_norm > 0.05:
        risk *= 1.5
    
    # Very close depth booster
    if depth < 0.3:
        risk *= 2.0
    elif depth < 0.5:
        risk *= 1.5
    
    return max(0.0, min(1.0, risk))

def compute_ttc(depth, vy):
    if depth <= 0: return None
    dist = 1.0 / depth
    speed = max(1e-3, abs(vy))
    return float(dist / speed)
