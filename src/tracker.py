import numpy as np
from collections import deque
from utils import xyxy_to_xywh, iou

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, frame_idx):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        cx, cy, w, h = xyxy_to_xywh(bbox)
        self.x = np.array([cx, cy, 0., 0., w, h], dtype=float)
        # Reduced uncertainty for better stability
        self.P = np.eye(6) * 5.0
        dt = 1.0
        self.F = np.eye(6)
        self.F[0,2] = dt
        self.F[1,3] = dt
        # Reduced process noise for smoother tracking
        self.Q = np.eye(6) * 0.5
        # Reduced measurement noise for more responsive updates
        self.R = np.eye(4) * 2.0
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.last_frame = frame_idx
        self.trace = deque(maxlen=30)
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()
    def update(self, bbox, frame_idx):
        meas = xyxy_to_xywh(bbox)
        z = np.array([meas[0], meas[1], meas[2], meas[3]])
        H = np.zeros((4,6)); H[0,0]=1; H[1,1]=1; H[2,4]=1; H[3,5]=1
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.last_frame = frame_idx
        cx, cy, w, h = self.x[0], self.x[1], self.x[4], self.x[5]
        self.trace.append((frame_idx, cx, cy))
    def get_state(self):
        cx, cy, _, _, w, h = self.x
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        return np.array([x1, y1, x2, y2])
    def estimate_velocity(self):
        if len(self.trace) >= 2:
            f1, x1, y1 = self.trace[-2]
            f2, x2, y2 = self.trace[-1]
            dt = max(1, f2 - f1)
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
            return vx, vy
        return 0.0, 0.0

class TrackerManager:
    def __init__(self, iou_threshold=0.25, max_age=20):
        self.trackers = []
        self.iou_threshold = iou_threshold  # Lower threshold = more strict matching
        self.max_age = max_age  # Reduced max age for faster cleanup of lost tracks
    def update(self, detections, frame_idx=0):
        trks = [t.predict() for t in self.trackers]
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx][:4], frame_idx)
        for d in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d][:4], frame_idx)
            self.trackers.append(new_trk)
        survivors = []
        results = []
        for trk in self.trackers:
            # More strict: require at least 2 hits before reporting
            if trk.time_since_update < self.max_age and trk.hits >= 2:
                survivors.append(trk)
                state = trk.get_state()
                vx, vy = trk.estimate_velocity()
                results.append({
                    "id": trk.id,
                    "bbox": state.tolist(),
                    "velocity": [float(vx), float(vy)],
                    "age": trk.age,
                    "last_frame": trk.last_frame
                })
            elif trk.time_since_update < self.max_age:
                # Keep alive but don't report until confirmed
                survivors.append(trk)
        self.trackers = survivors
        # list of trackers' dictionary
        return results
    def associate(self, detections, trks):
        if len(trks) == 0:
            return [], list(range(len(detections))), []
        cost_matrix = np.zeros((len(trks), len(detections)), dtype=float)
        for t, trk in enumerate(trks):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = 1.0 - iou(trk, det[:4])
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched = []
            matched_rows = set(); matched_cols = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r,c] <= (1.0 - self.iou_threshold):
                    matched.append((r,c)); matched_rows.add(r); matched_cols.add(c)
            unmatched_trks = [i for i in range(len(trks)) if i not in matched_rows]
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_cols]
            return matched, unmatched_dets, unmatched_trks
        except Exception:
            matched = []
            matched_dets = set()
            for t in range(cost_matrix.shape[0]):
                c = int(cost_matrix[t].argmin())
                if cost_matrix[t,c] <= (1.0 - self.iou_threshold) and c not in matched_dets:
                    matched.append((t,c)); matched_dets.add(c)
            matched_rows = set([m[0] for m in matched]); matched_cols = set([m[1] for m in matched])
            unmatched_trks = [i for i in range(len(trks)) if i not in matched_rows]
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_cols]
            return matched, unmatched_dets, unmatched_trks
