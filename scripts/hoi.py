import os
import cv2
import numpy as np
from project_utils import cfg, paths, Case, camera, axis, proj_xyz_to_uv, BONES,load_labels, active_at, act_name

case = Case
obj_cfg = cfg.get("objects")
Inter_px = int(obj_cfg.get("interact_px"))
Sup_draw =set([str(s).lower() for s in obj_cfg.get("suppress_draw")]) #person
Alias = {(str(k) or "").lower(): (str(v)or "").lower() for k, v in obj_cfg.get("class_aliases").items()}
sync_cfg = cfg.get("sync")
Skele_fps = float(sync_cfg.get("skele_fps"))
Frame_off = {
    "L": int(sync_cfg.get("frame_offset_L")),
    "M": int(sync_cfg.get("frame_offset_M")),
    "R": int(sync_cfg.get("frame_offset_R"))}

ali_dir = os.path.join(paths.get("aligned", os.path.join(paths.get("output"), "aligned")))
skel_paths = {
    "L": os.path.join(ali_dir, f"aligned_skeleton_L_{case}.npy"),
    "M": os.path.join(ali_dir, f"aligned_skeleton_M_{case}.npy"),
    "R": os.path.join(ali_dir, f"aligned_skeleton_R_{case}.npy")}
skeles={}
for v, p in skel_paths.items():
    if os.path.isfile(p):
        arr = np.load(p).astype(np.float32)
        if arr.ndim ==3 and arr.shape [1:] == (25,3):
            skeles[v] = arr
        else:
            print("bad skeleton shape", v, arr.shape)
    else:
        print("Missing aligned skeleton", v, p)

rgb_paths = {
    "L": os.path.join(paths.get("rgb_l",""), f"{case}-L_color.avi"),
    "M": os.path.join(paths.get("rgb_m",""), f"{case}-M_color.avi"),
    "R": os.path.join(paths.get("rgb_r",""), f"{case}-R_color.avi")}
caps, sizes,fps_view, ratio_vm = {}, {}, {},{}
for v in ("L","M","R"):
    if v in skeles:
        cap = cv2.VideoCapture(rgb_paths[v])
        caps[v] =cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sizes[v] = (w, h)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fps_view[v] = fps
        ratio_vm[v] = Skele_fps/fps
        # print(v,"=", w, h, ratio_vm[v])

det_dir = os.path.join(paths.get("output"),"detections")
dets = {}
for v in caps:
    npz = os.path.join(det_dir, f"dets_{v}_{case}.npz")
    d = np.load(npz, allow_pickle=True)
    dets[v] = {"frames": d["frames"], "boxes": d["boxes"], "confs": d["confs"],
        "ids": d["ids"], "names": d["names"]}
    # print("dets loaded", npz)

res_dir =os.path.join(paths.get("output"), "results")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writers, out_paths = {},{}
for v in caps:
    w, h = sizes[v]
    outp = os.path.join(res_dir, f"hoi_{v}_{case}.avi")
    writers[v] = cv2.VideoWriter(outp, fourcc, fps_view[v], (w, h))
    out_paths[v] = outp

# Camera bits
Kmap = {v: camera(v) for v in caps}
AXmap = {v: axis(v)   for v in caps}
Hand_joints = [7, 11, 21, 22, 23, 24]

def box_center(b):
    x1, y1, x2, y2 = b
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0], np.float32)

def associate(uv_25x2, boxes, names, px_thr):
    if boxes.size ==0:
        return []
    out= []
    for hj in Hand_joints:
        p = uv_25x2[hj]
        dmin,argmin,aname = 1e9, -1, None
        for bi, b in enumerate(boxes):
            c = box_center(b)
            d = float(np.linalg.norm(p-c))
            if d < dmin:
                dmin, argmin, aname = d, bi, names[bi] if len(names) > bi else "obj"
        if dmin < px_thr and argmin >= 0:
            out.append((hj, argmin, aname, dmin))
    return out

def draw_boxes(img, boxes, names):
    for i, b in enumerate(boxes):
        name = (names[i] if i < len(names) else "obj") or "obj"
        if name.lower() in Sup_draw:
            continue
        x1, y1, x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.putText(img, name, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def draw_interactions(img, uv, boxes, inter):
    drawn = set()
    for (hj, bi, name, _) in inter:
        if (name or "").lower() in Sup_draw:
            continue
        key = (bi, (name or "").lower())
        if key in drawn:
            continue  # skip duplicate same object links
        p = tuple(np.round(uv[hj]).astype(int))
        c = tuple(np.round(box_center(boxes[bi])).astype(int))
        cv2.line(img, p, c, (255, 255, 0), 2, cv2.LINE_AA)
        mid = ((p[0] + c[0]) // 2, (p[1] + c[1]) // 2)
        cv2.putText(img, f"{name}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        drawn.add(key)

def alias(n, active_ids):
    n = (n or "").lower()
    n = Alias.get(n,n)
    if any(a in (20,) for a in active_ids) and n in ("helmet","hat","sports ball","baseball glove"):
        n = "cap"
    if any(a in (22,28) for a in active_ids) and n in ("backpack", "handbag", "suitcase"):
        n = "jacket"
    if any(a in (30,) for a in active_ids) and n in ("remote","mouse"):
        n = "watch"
    return n

def frame_to_idx(fidx, ratio, offset, T):
    i = int(round(fidx * ratio)) + offset
    return max(0,min(i, T-1))

labels_cache = {v: load_labels(case, v) for v in caps}
active_views = set(caps.keys())
fidx = 0
while active_views:
    frames = {}
    to_close =[]
    for v in list(active_views):
        ok, frm = caps[v].read()
        if not ok:
            to_close.append(v)
            continue
        frames[v] = frm

    for v in to_close:
        caps[v].release()
        writers[v].release()
        active_views.remove(v)
    if not frames:
        break

    for v, img in frames.items():
        w,h = sizes[v]
        sk_idx = frame_to_idx(fidx,ratio_vm[v], Frame_off[v],skeles[v].shape[0])
        sk = skeles[v][sk_idx]
        uv, valid = proj_xyz_to_uv(sk, Kmap[v], w, h, AXmap[v]) #proj_xyz_to_uv takes sign as the 5th positional arg
        boxes = dets[v]["boxes"][fidx] if fidx < len(dets[v]["boxes"]) else np.empty((0,4),np.float32)
        raw_names = dets[v]["names"][fidx] if fidx < len(dets[v]["names"]) else np.array([], dtype = object)

        active_ids = active_at(labels_cache[v], fidx)
        caption = ", ".join([act_name(aid, cfg) for aid in active_ids]) if active_ids else ""

        names = np.array([alias(rn, active_ids) for rn in raw_names], dtype =object)

        for (i, j) in BONES:
            p1=tuple(np.round(uv[i]).astype(int))
            p2 = tuple(np.round(uv[j]).astype(int))
            cv2.line(img, p1, p2, (255,0,0), 2, cv2.LINE_AA)
        for p in uv:
            cv2.circle(img, tuple(np.round(p).astype(int)), 3, (255,0,0), -1)

        draw_boxes(img,boxes, names)
        inter = associate(uv, boxes, names, Inter_px)
        draw_interactions(img, uv, boxes, inter)

        hud = f"{v} valid= {int(np.sum(valid))}/25"
        if caption:
            hud += f"action={caption}"
        cv2.putText(img, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0),2, cv2.LINE_AA)
        writers[v].write(img)
    fidx += 1

for v in list(caps.keys()):
    try:
        caps[v].release()
    except:
        pass
for v in list(writers.keys()):
    try:
        writers[v].release()
    except:
        pass

print("Human object Interaction Videos")
for v, p in out_paths.items():
    if os.path.isfile(p):
        print(f"{v} view video in {p}")