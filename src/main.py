"""
Usage:
    python main.py --video ../data/demo.mp4 --yolo yolov8n.pt --output output/out.mp4
demo video at ../data/demo.mp4 (or pass --video).
"""
import os, sys, argparse, time, json
sys.path.append(os.path.dirname(__file__))
from detection import YoloDetector
from tracker import TrackerManager
from depth import DepthEstimator
from risk import compute_risk, compute_ttc
from prob_risk import compute_risk_prob
from instaYOLO_seg import polygons_from_detections
from viz import draw_overlay
from tts import TTSWorker
import cv2
def process(video_path, yolo_weights, output_path, imgsz=320, skip_depth=5, no_show=False):
    det = YoloDetector(weights=yolo_weights, imgsz=imgsz, conf=0.25)
    tracker = TrackerManager(iou_threshold=0.3, max_age=30)
    depth = DepthEstimator(device='cpu')
    tts = TTSWorker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    frame_idx = 0
    logf = open(os.path.join(os.path.dirname(output_path) or '.', 'output_log.jsonl'), 'w')
    ego = (w/2, h)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1; t0 = time.time()
            dets = det.detect(frame)
            bboxes = [d[:4] for d in dets]
            # depth every skip_depth frames
            depths = depth.estimate(frame, bboxes) if frame_idx % skip_depth == 0 else depth.estimate(frame, bboxes)
            polys = polygons_from_detections(frame, dets)
            tracks = tracker.update(dets, frame_idx)
            objs = []
            for i, tr in enumerate(tracks):
                tb = tr['bbox']; vx, vy = tr['velocity']
                # match detection by max iou
                best_idx = None; best_iou = 0.0
                for idx, bb in enumerate(bboxes):
                    from utils import iou
                    val = iou(tb, bb)
                    if val > best_iou: best_iou = val; best_idx = idx
                depth_val = depths[best_idx] if best_idx is not None and best_idx < len(depths) else 1.0
                det_risk = compute_risk(tb, depth_val, frame.shape)
                # prepare object state for probabilistic risk
                cx = (tb[0]+tb[2])/2; cy = (tb[1]+tb[3])/2
                obj_state = [cx, cy, vx, vy]
                prob_risk, ttc_prob = compute_risk_prob(ego, obj_state, obj_cov=None, dt=0.2, horizon=3.0, n_samples=250, radius=40.0)
                # fuse risks
                fused = 0.6*det_risk + 0.4*prob_risk
                objs.append({
                    'id': int(tr['id']),
                    'bbox': [float(x) for x in tb],
                    'risk': float(fused),
                    'class': -1,
                    'polygon': polys[i].tolist() if polys and polys[i] is not None else None,
                    'ttc': float(ttc_prob) if ttc_prob is not None else None
                })
            # TTS for highest risk
            if objs:
                top = max(objs, key=lambda o: o['risk'])
                if top['risk'] > 0.6:
                    side = 'left' if (top['bbox'][0]+top['bbox'][2])/2 < w*0.4 else 'right' if (top['bbox'][0]+top['bbox'][2])/2 > w*0.6 else 'ahead'
                    msg = f"Caution: object {side}. Risk {top['risk']:.2f}."
                    tts.speak(msg, urgency=min(1.0, (top['risk']-0.6)/0.4))
            vis = draw_overlay(frame, objs)
            writer.write(vis)
            if not no_show:
                cv2.imshow('AssistedVision', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            logf.write(json.dumps({'frame': frame_idx, 'objects': objs}) + '\n')
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx} processed in {time.time()-t0:.3f}s, objs={len(objs)}")
    finally:
        cap.release(); writer.release(); logf.close(); tts.stop(); cv2.destroyAllWindows()
        print('Finished. Output at', output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--yolo', default='yolov8n.pt')
    parser.add_argument('--output', default='output/out.mp4')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--skip_depth', type=int, default=3)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    process(args.video, args.yolo, args.output, imgsz=args.imgsz, skip_depth=args.skip_depth, no_show=args.no_show)
