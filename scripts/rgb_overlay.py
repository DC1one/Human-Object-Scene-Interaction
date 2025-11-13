import os
import cv2
import numpy as np
from project_utils import cfg, paths, viz, camera, axis, proj_xyz_to_uv, BONES, bgr, Case

case = Case
# Configuration
sync = cfg.get("sync", {})
skele_fps = float(sync.get("skele_fps"))
offset = {
    "L": int(sync.get("frame_offset_L")),
    "M": int(sync.get("frame_offset_M")),
    "R": int(sync.get("frame_offset_R"))} # -30
auto_off = bool(sync.get("auto_offset")) # false
auto_max = int(sync.get("auto_offset_max"))
auto_n = int(sync.get("auto_offset_probe"))
scene_label = str(sync.get("scene_l"))

"""
Inputs (Aligned skeletons + RGB videos)
"""
# Aligned skeletons for each view
ali_dir = paths.get("aligned", os.path.join(paths.get("output","."), "aligned"))
skel_fp = {
    "L": os.path.join(ali_dir, f"aligned_skeleton_L_{case}.npy"),
    "M": os.path.join(ali_dir, f"aligned_skeleton_M_{case}.npy"),
    "R": os.path.join(ali_dir, f"aligned_skeleton_R_{case}.npy")}
skel = {}
for v, fp in skel_fp.items():
    arr = np.load(fp)
    if arr.ndim==3 and arr.shape[1:] == (25, 3):
        skel[v] = arr.astype(np.float32)
    else:
        print(f" Bad shape for {v}={arr.shape}")

# load rgb videos of each view
vid_fp = {
    "L": os.path.join(paths.get("rgb_l",""), f"{case}-L_color.avi"),
    "M": os.path.join(paths.get("rgb_m",""), f"{case}-M_color.avi"),
    "R": os.path.join(paths.get("rgb_r",""), f"{case}-R_color.avi")}
caps, size, fps, ratio = {}, {}, {}, {}
for v in ("L","M","R"):
    if v not in skel:
        continue
    vp=vid_fp.get(v, "")
    cap=cv2.VideoCapture(vp)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS) or 30.0
    caps[v], size[v], fps[v] = cap, (w,h), float(f)
    ratio[v] = skele_fps / fps[v]

#----------------------------------------------------------


vis_dir = os.path.join(paths.get("output","."), "visualizations")
os.makedirs(vis_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
wr, outp = {}, {}
for v in caps:
    w,h = size[v]
    op = os.path.join(vis_dir, f"skeleton_overlay_{v}_{case}.avi")
    wr[v] = cv2.VideoWriter(op, fourcc, fps[v], (w,h))
    outp[v] = op

Kmat = {v: camera(v) for v in caps}
AX   = {v: axis(v) for v in caps}

# Draw skeleton
def draw_skel(img, uv, color):
    for i,j in BONES:
        p1 = tuple(np.round(uv[i]).astype(int))
        p2 = tuple(np.round(uv[j]).astype(int))
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    for p in uv:
        cv2.circle(img, tuple(np.round(p).astype(int)), 3, color, -1)

# Offset
def est_off(v, N, mx): # N = 300, mx = 60
    cap = caps[v]
    Tvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Tskele = skel[v].shape[0]
    N = int(min(N, Tvid-1, Tskele-1))
    if N<10:
        return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev,vd = None, []
    for _ in range(N):
        ok, frm = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            vd.append(float(np.mean(np.abs(g.astype(np.int16)-prev.astype(np.int16)))))
        prev = g
    vd = np.asarray(vd, np.float32)
    disp = np.linalg.norm(skel[v][1:N] - skel[v][:N-1], axis = 2).sum(axis = 1)
    vd = (vd - vd.mean()) / (vd.std() + 1e-6)
    disp =(disp - disp.mean()) / (disp.std() + 1e-6)
    best_off, best= 0, -1e9
    for o in range(-mx, mx+1):
        if o < 0:
            a, b = vd[-o:], disp[:len(vd)+o]
        elif o > 0:
            a, b = vd[:len(vd)-o], disp[o:]
        else:
            a, b = vd, disp[:len(vd)]
        if len(a) < 10:
            continue
        s = float(np.dot(a, b) / len(a))
        if s > best:
            best, best_off = s, o
    return int(best_off)

if auto_off:
    for v in list(caps.keys()):
        o = est_off(v, N=auto_n, mx=auto_max)
        offset[v] = int(offset.get(v, 0)) + o
        print(f"Auto offset {v} = {o} to total {offset[v]}")

sample = float(viz.get("frame_sample", 1.0))
step = max(1, int(round(1.0 / sample))) if 0 < sample <= 1.0 else 1

# Loop length
def T(cap): return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
limits = []
for v in caps:
    Tvid = T(caps[v])
    Tskele = skel[v].shape[0]
    max_ok = int(np.floor((Tskele := Tskele -1 - abs(offset[v])) / max(1e-6, ratio[v])))
    limits.append(min(Tvid, max(0, max_ok)))
T_loop = int(min(limits)) if limits else 0

# Main loop
fidx, wrote = 0, 0
while fidx<T_loop:
    if step>1:
        for v in caps:
            caps[v].set(cv2.CAP_PROP_POS_FRAMES, fidx)
    frames = {}
    for v in list(caps.keys()):
        ok, frm = caps[v].read()
        if not ok:
            caps[v].release(); wr[v].release()
            del caps[v], wr[v], Kmat[v],AX[v], size[v], fps[v], ratio[v]
            continue
        frames[v] = frm
    if not frames:
        break

    for v, img in frames.items():
        w, h = size[v]
        si = int(round(fidx * ratio[v]))+offset[v]
        si = max(0, min(si, skel[v].shape[0]-1))
        uv,valid = proj_xyz_to_uv(skel[v][si], Kmat[v], w, h, AX[v])
        draw_skel(img, uv, {"L": bgr("r"), "M": bgr("b"), "R": bgr("g")}.get(v, bgr("w")))
        cv2.putText(img, f"{v} Joints valid = {int(np.sum(valid))}/25 ",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2, cv2.LINE_AA)
        cv2.putText(img, scene_label, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        wr[v].write(img)

    wrote += 1
    fidx += step

for v in list(caps.keys()): caps[v].release()
for v in list(wr.keys()):   wr[v].release()

for v, p in outp.items():
    if os.path.isfile(p):
        print(f"{v} video saved path {p}")