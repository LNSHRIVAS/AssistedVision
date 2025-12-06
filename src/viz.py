import cv2
import math
import numpy as np
def risk_to_color(risk):
    # green: safe
    if risk < 0.4:
        return (0, 255, 0)        
    # yellow: caution
    elif risk < 0.7:
        return (0, 255, 255)
    # orange: warning
    elif risk < 0.9:
        return (0, 165, 255)      
    # red: danger
    else:
        return (0, 0, 255)
def draw_overlay(frame, objects, passable_gaps=None):
    vis = frame.copy(); h,w = frame.shape[:2]
    
    # Draw passable gaps first (as background highlights)
    if passable_gaps:
        for gap in passable_gaps:
            # Get gap boundaries
            left_obj = gap['left_obj']
            right_obj = gap['right_obj']
            left_x = int((left_obj['bbox'][0] + left_obj['bbox'][2]) / 2)
            right_x = int((right_obj['bbox'][0] + right_obj['bbox'][2]) / 2)
            
            # Draw green vertical lines to indicate gap boundaries
            cv2.line(vis, (left_x, 0), (left_x, h), (0, 255, 0), 2)
            cv2.line(vis, (right_x, 0), (right_x, h), (0, 255, 0), 2)
            
            # Draw gap info at top
            gap_center_x = (left_x + right_x) // 2
            gap_width_cm = int(gap['width'] * 100)
            gap_text = f"GAP: {gap_width_cm}cm"
            if not gap['is_perpendicular']:
                gap_text += " (angled)"
            
            # Background for text
            text_size = cv2.getTextSize(gap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis, 
                         (gap_center_x - text_size[0]//2 - 5, 10),
                         (gap_center_x + text_size[0]//2 + 5, 35),
                         (0, 200, 0), -1)
            
            cv2.putText(vis, gap_text, 
                       (gap_center_x - text_size[0]//2, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for obj in objects:
        x1,y1,x2,y2 = map(int, obj['bbox']); color = risk_to_color(obj['risk'])
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        if obj.get('polygon') is not None:
            pts = np.array(obj['polygon'], dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(vis, [pts], True, color, 1)
        label = f"ID{obj['id']} R{obj['risk']:.2f}"
        cv2.putText(vis, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # polar overlay
    ph,pw = 180,180; overlay = np.zeros((ph,pw,3), dtype=np.uint8) + 40; center=(pw//2, ph-8)
    for obj in objects:
        x1,y1,x2,y2 = obj['bbox']; cx=(x1+x2)/2; cy=(y1+y2)/2; dx = cx - w/2; dy = h - cy
        angle = math.atan2(dx, dy); r = max(0.01, 1.0-obj['risk']); max_r=65; radpix=int(max_r*(1.0-r))
        x=int(center[0]+math.sin(angle)*radpix); y=int(center[1]-math.cos(angle)*radpix)
        cv2.circle(overlay, (x,y), 6, risk_to_color(obj['risk']), -1)
    oy,ox = h-ph-8, w-pw-8
    # Check bounds before overlay
    if oy >= 0 and ox >= 0 and oy+ph <= h and ox+pw <= w:
        vis[oy:oy+ph, ox:ox+pw] = cv2.addWeighted(vis[oy:oy+ph, ox:ox+pw], 0.6, overlay, 0.4, 0)
    # legend
    entries = [('Safe',(0,255,0)),('Caution',(0,255,255)),('Warning',(0,165,255)),('Danger',(0,0,255))]
    for i,(txt,col) in enumerate(entries):
        y=10+i*20; cv2.rectangle(vis,(10,y),(28,y+14),col,-1); cv2.putText(vis, txt, (34,y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)
    return vis
