import cv2
import numpy as np
def bbox_to_polygon(frame, bbox, num_points=24, min_contour_area=80):
    x1,y1,x2,y2 = map(int, bbox)
    h,w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
    crop = frame[y1:y2+1, x1:x2+1]
    if crop.size == 0: return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area: continue
        cnt = cnt.squeeze(1)
        if cnt.ndim != 2 or cnt.shape[0] < 6: continue
        deltas = np.sqrt(((cnt[1:] - cnt[:-1])**2).sum(axis=1))
        cumlen = np.concatenate(([0.0], deltas.cumsum()))
        total = cumlen[-1]
        if total == 0: continue
        target_locs = np.linspace(0, total, num_points, endpoint=False)
        resampled = []
        idx = 0
        for t in target_locs:
            while idx < len(cumlen)-1 and cumlen[idx+1] < t:
                idx += 1
            if idx == len(cumlen)-1:
                p = cnt[-1]
            else:
                t0, t1 = cumlen[idx], cumlen[idx+1]
                p0, p1 = cnt[idx], cnt[idx+1]
                alpha = 0 if t1==t0 else (t - t0) / (t1 - t0)
                p = (1-alpha)*p0 + alpha*p1
            resampled.append([int(p[0]) + x1, int(p[1]) + y1])
        return np.array(resampled, dtype=np.int32)
    return None

def polygons_from_detections(frame, detections, num_points=24):
    polys = []
    for d in detections:
        poly = bbox_to_polygon(frame, d[:4], num_points=num_points)
        polys.append(poly)
    return polys
