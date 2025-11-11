import os
import time
import cv2
import numpy as np
import torch
from project_utils import Case, cfg, paths
from ultralytics import YOLO

case = Case
obj_cfg = cfg.get("objects")
Model = obj_cfg.get("model")
Conf = float(obj_cfg.get("conf"))
Iou= float(obj_cfg.get("iou")) # 0.25
Max_det = int(obj_cfg.get("max_det"))
W_list = set([s.lower() for s in obj_cfg.get("classes")])
Views = obj_cfg.get("views")
Prev= bool(obj_cfg.get("write_det_preview"))# false
# First Pass
Img = int(obj_cfg.get("imgsz")) # 1280
Agnostic =bool(obj_cfg.get("agnostic_nms"))
Dev = str(obj_cfg.get("device"))
Frame_st = int(obj_cfg.get("frame_stride"))
Max_frame = int(obj_cfg.get("max_frames")) # 0 for all
Log = int(obj_cfg.get("log_every"))
Check_point = int(obj_cfg.get("checkpoint_every")) # 500
# second pass
Enable_sec_pass = bool(obj_cfg.get("second_pass")) # Ture
Redetect =set([s.lower() for s in obj_cfg.get("redetect_on")])
Crop_scale= float(obj_cfg.get("crop_scale"))
Sec_conf = float(obj_cfg.get("second_conf"))
Sec_iou = float(obj_cfg.get("second_iou")) # 0.35

# Inputs
det_dir = os.path.join(paths.get("output"), "detections")
vis_dir = os.path.join(paths.get("output"), "visualizations")

def video(view):
    vroot = {"L": paths.get("rgb_l"), "M": paths.get("rgb_m"), "R": paths.get("rgb_r")}
    vpath =os.path.join(vroot[view], f"{case}-{view}_color.avi")
    cap = cv2.VideoCapture(vpath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, vpath, w, h, float(fps), int(T)

# Allowed Classes
def allow(name):
    n = (name or "").lower()
    return True if not W_list else (n in W_list)

def save_file(path, frames, boxes, confs, ids, names):
    np.savez_compressed(path,
        frames = np.asarray(frames, dtype=np.int32),
        boxes = np.asarray(boxes,dtype=object),
        confs = np.asarray(confs, dtype=object),
        ids = np.asarray(ids, dtype=object),
        names= np.asarray(names, dtype=object))

def xy_xy_clip(x1,y1,x2,y2,w,h):
    return max(0,x1),max(0,y1), min(w-1,x2), min(h-1,y2)

def scale_box(cx, cy, bw, bh, scale, W, H):
    nw, nh = bw*scale, bh*scale
    x1 = int(round(cx - nw/2)); y1 = int(round(cy - nh/2))
    x2 = int(round(cx + nw/2)); y2 = int(round(cy + nh/2))
    return xy_xy_clip(x1, y1, x2, y2, W,H)

def iou(a, b):
    ax1, ay1, ax2, ay2= a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1,by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2,by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, (ax2-ax1) * (ay2-ay1))
    b_area = max(0, (bx2-bx1) * (by2-by1))
    union = a_area + b_area - inter + 1e-6 # 1e-7
    return inter/union

def merge_dets(base_boxes, base_names, base_confs, add_boxes, add_names, add_confs, iou_thr=0.5, agnostic= True): # threshold 0.45
    for b, n, c in zip(add_boxes, add_names, add_confs):
        keep = True
        for bb, bn in zip(base_boxes, base_names):
            if iou(bb, b) > iou_thr and (bn == n or agnostic):
                keep = False
                break
        if keep:
            base_boxes.append(b); base_names.append(n); base_confs.append(c)

# ultralytics predict for a single image
def predict_image(model, image, conf, iou, imgsz, agnostic, max_det):
    res = model.predict(source = image, verbose = False, conf=conf, iou=iou,
        max_det=max_det, imgsz=imgsz, agnostic_nms=agnostic, device=Dev if Dev!="auto" else None)[0]
    f_boxes, f_confs, f_ids, f_names = [], [], [], []
    if res.boxes is not None and len(res.boxes) > 0:
        cls_ids = res.boxes.cls.detach().cpu().numpy().astype(int)
        confs_np = res.boxes.conf.detach().cpu().numpy().astype(float)
        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(float)
        for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs_np):
            name = (model.names.get(int(cid), str(int(cid))) or str(int(cid))).lower()
            if allow(name):
                f_boxes.append([x1, y1, x2, y2])
                f_confs.append(cf)
                f_ids.append(int(cid))
                f_names.append(name)
    return f_boxes, f_confs, f_ids, f_names


def choose_device(): # Auto
    global Dev
    if Dev == "auto":
        Dev = "cuda" if torch.cuda.is_available() else "cpu"
    return Dev

def checkpoint(path, frames, boxes_all, confs_all, ids_all, names_all, last_idx):
    tmp = path +".part"
    save_file(tmp, frames, boxes_all, confs_all, ids_all, names_all)

#______________________________________________________

def main():
    dev= choose_device()
    print(Model, dev)
    model = YOLO(Model)
    _ = model.predict(source = np.zeros((64,64,3), dtype= np.uint8), verbose = False, conf = 0.01, iou = 0.5, imgsz=64, max_det=1, device=dev)
    for view in Views:
        cap, vpath, W, H,fps, T = video(view)
        out_npz = os.path.join(det_dir, f"dets_{view}_{case}.npz")
        print(f"Detecting {view} from {vpath} to {out_npz}")

        writer = None
        if Prev:
            fourcc =cv2.VideoWriter_fourcc(*"XVID")
            out_vid = os.path.join(vis_dir, f"dets_preview_{view}_{case}.avi")
            writer = cv2.VideoWriter(out_vid, fourcc, fps / max(1, Frame_st), (W, H))
        frames, boxes_all, confs_all, ids_all, names_all = [], [], [], [], []
        # loop
        idx = 0
        processed = 0
        t0 = time.time()

        while True:
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame = cap.read()
            if not ok:
                break
            # stride
            if (idx % Frame_st) !=0:
                idx += 1
                continue

            f_boxes, f_confs, f_ids, f_names = predict_image(model, frame, Conf, Iou, Img, Agnostic, Max_det)
            if Enable_sec_pass and f_boxes:
                crops = []
                for b, n in zip(f_boxes,f_names):
                    if n in Redetect:
                        x1,y1, x2, y2 = map(int, b)
                        bw, bh = x2-x1, y2-y1
                        cx, cy = x1 + bw/2.0, y1 + bh/2.0
                        rx1, ry1, rx2,ry2 = scale_box(cx, cy, bw, bh, Crop_scale, W,H)
                        crops.append((rx1, ry1, rx2, ry2))
                for (rx1, ry1, rx2, ry2) in crops:
                    crop = frame[ry1:ry2, rx1:rx2]
                    if crop.size == 0:
                        continue
                    add_boxes, add_confs, add_ids, add_names = predict_image(model, crop, Sec_conf, Sec_iou, Img, Agnostic, Max_det)
                    # map back plus filter
                    mb, mn, mc = [],[], []
                    for (x1,y1,x2,y2), n, c in zip(add_boxes, add_names, add_confs):
                        gx1 = float(rx1 + x1); gy1 = float(ry1 + y1)
                        gx2 = float(rx1 + x2); gy2 = float(ry1 + y2)
                        gx1, gy1, gx2, gy2 = xy_xy_clip(gx1, gy1, gx2, gy2, W, H)
                        # ignore near full crop
                        if (gx2 - gx1) * (gy2 - gy1) < 0.95 * (rx2 - rx1) * (ry2 - ry1):
                            mb.append([gx1, gy1, gx2, gy2]); mn.append(n); mc.append(c)
                    merge_dets(f_boxes, f_names, f_confs, mb, mn, mc, iou_thr=Sec_iou, agnostic=Agnostic)

            # it is only for frames we processed due to stride
            frames.append(idx)
            boxes_all.append(np.array(f_boxes, dtype=np.float32))
            confs_all.append(np.array(f_confs, dtype=np.float32))
            ids_all.append(np.array(f_ids, dtype=np.int32))
            names_all.append(np.array(f_names, dtype=object))

            if writer is not None:
                vis = frame.copy()
                for b, n in zip(f_boxes, f_names):
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(vis, n, (x1,max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0), 2)
                writer.write(vis)

            processed += 1
            idx += 1
            if processed % Log== 0:
                dt = time.time() - t0
                fps_eff = processed / max(dt, 1e-6)
                print(f"{view}, processed {processed} frames, \nfps={fps_eff} \nsecond_pass= {Enable_sec_pass}")

            if processed %Check_point == 0:
                checkpoint(out_npz, frames, boxes_all, confs_all, ids_all, names_all, idx)
            if Max_frame > 0 and processed >= Max_frame:
                print(f"{view}, reached max frames = {Max_frame}.")
                break

        cap.release()
        if writer is not None:
            writer.release()

        # save full or partial set
        save_file (out_npz, frames, boxes_all, confs_all, ids_all, names_all)
        print(f"Saved detections= {out_npz}, frames={len(frames)}")

if __name__ == "__main__":
    main()
