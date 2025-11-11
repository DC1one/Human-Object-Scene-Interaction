import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Configration
CFG = "/home/kakarot/coding_task/config/settings.yaml"

def load_cfg(path=CFG):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_cfg()
paths = cfg.get("paths")
cam = cfg.get("camera")
viz = cfg.get("visualization")
RGB_W = int(viz.get("rgb_width"))
RGB_H = int(viz.get("rgb_height"))
Case = str(cfg.get("case"))

# skeleton bones and joints
BONES = [
    (3,2),(2,20),(20,8),(20,4),(20,1),(1,0), # upper body
    (8,9),(9,10),(10,11),(11,24),(11,23), # left arm
    (4,5),(5,6),(6,7),(7,21),(7,22),# right arm
    (0,16),(16,17),(17,18),(18,19),# left leg
    (0,12),(12,13),(13,14),(14,15)# right leg
]

def bgr(name):
    col = {"r": (0,0,255), "g": (0,255,0), "b": (255,0,0), "w": (255,255,255)}
    return col.get(name.lower(), (255,255,255))

#Camera matrix for all three views
def camera(view):
    v = view.upper()
    try:
        if v== "L": fx,fy,cx,cy = cam["fx_l"],cam["fy_l"],cam["cx_l"],cam["cy_l"]
        elif v== "M": fx,fy,cx,cy = cam["fx_m"],cam["fy_m"],cam["cx_m"],cam["cy_m"]
        elif v== "R": fx,fy,cx,cy = cam["fx_r"],cam["fy_r"],cam["cx_r"],cam["cy_r"]
        else:
            return np.eye(3, dtype=np.float32)
    except KeyError:
        return np.eye(3, dtype=np.float32)
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)

def axis(view):
    v = view.upper()
    key = f"axis_sign_{v.lower()}"
    if key in cam:
        return tuple(cam[key])
    return 1.0, -1.0 ,1.0

def skel_line(line):
    vals = line.strip().split()
    if len(vals) not in (75,150):
        return None
    a = np.asarray(list(map(float, vals)), np.float32)
    if len(a)==75:
        return np.stack([a.reshape(25,3)],0)
    p1,p2 = a[:75].reshape(25,3), a[75:].reshape(25,3)
    if np.allclose(p2,0,atol=1e-7):
        return np.stack([p1],0)
    return np.stack([p1,p2],0)

# load skeleton T P joints 3
def load_skel(path):
    if not os.path.isfile(path):
        # print(f"skeleton file missing {path}")
        return np.zeros((0, 1, 25, 3),np.float32)
    frames, max_p = [], 1
    for ln in open(path):
        arr =skel_line(ln)
        if arr is None:
            continue
        frames.append(arr)
        max_p = max(max_p, arr.shape[0])
    if not frames:
        return np.zeros((0,1,25,3),np.float32)
    out = []
    for a in frames:
        pad = np.zeros((max_p,25,3),np.float32)
        pad[:a.shape[0]] = a
        out.append(pad)
    return np.stack(out,0)

# remove all zero frames
def rem_zeros(tp253):
    m = np.any(tp253!=0,axis=(1,2,3))
    return tp253[m]

# select person
def pick_primary(tp253):
    T,P = tp253.shape[:2]
    if P==1:
        return tp253[:, 0, :, :]
    mot = []
    for p in range(P):
        tr = tp253[:, p, :, :].reshape(T,-1)
        diff = np.diff(tr,0)
        mot.append(np.sum(np.linalg.norm(diff,1)))
    best = int(np.argmax(mot))
    return tp253[:, best, :, :]

# Projection
def proj_xyz_to_uv(xyz, Kmat, w=RGB_W, h=RGB_H, sign=(1.0,-1.0,1.0)):
    pts = np.asarray(xyz,np.float32)
    sx,sy,sz = sign
    pts = pts * np.array([sx,sy,sz],np.float32)
    uvw = (Kmat @ pts.T).T
    c = uvw[:, 2] + 1e-9
    u, v = uvw[:, 0] /c, uvw[:,1] /c
    ok = (c>0)&np.isfinite(u)&np.isfinite(v)
    uv = np.stack([np.clip(u,0,w-1), np.clip(v,0,h-1)],-1)
    return uv.astype(np.float32), ok

# 3d Plotting of skeleton
def plot_skel3d(skel, bones=None, ax=None, color="r"):
    if bones is None:
        bones=BONES
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i, j in bones:
        x = [skel[i,0],skel[j,0]]; y=[skel[i,1],skel[j,1]]; z=[skel[i,2],skel[j,2]]
        ax.plot(x,y,z,c=color,lw=2)
    ax.scatter(skel[:,0],skel[:,1],skel[:,2],c="g",s=15)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(20, -70)
    return ax

# labels
def lbl_path(view, case):
    v = view.upper()
    base=paths.get(f"label_{v.lower()}","")
    return os.path.join(base,f"{case}-{v}.label")

def load_labels(case,view):
    fp=lbl_path(view,case)
    if not os.path.isfile(fp): return []
    acts=[]
    for ln in open(fp):
        p=ln.strip().split(",")
        if len(p)<3: continue
        try:
            a,s,e=int(p[0]),int(p[1]),int(p[2])
            acts.append((a,s,e))
        except: pass
    return acts

def active_at(acts,idx):
    return [a for (a,s,e) in acts if s <= idx <= e]

def act_name(aid,cfg_=None):
    amap=(cfg_ or cfg).get()
    return amap.get(str(aid),f"Action {aid}")
