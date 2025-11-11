import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from project_utils import paths, load_skel, rem_zeros, pick_primary, plot_skel3d, BONES, Case

"""
Umeyama Similarity Copyright: Carlo Nicolini, 2013
Idea adapted from (https://github.com/clementinboittiaux/umeyama-python/blob/main/umeyama.py)
"""
def umeyama(X, Y, scale = True):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    if X.shape != Y.shape or X.ndim != 2:
        return 1.0, np.eye(3), np.zeros(3)
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    x, y = X - mx, Y - my

    C = (y.T @ x) / n
    U, S, Vt = np.linalg.svd(C)
    V = Vt.T
    R = U @ V.T
    if np.linalg.det(R) < 0:
        D = np.eye(3); D[-1,-1] = -1
        R = U @ D @ V.T
        S[-1] *= -1

    c = S.sum() / ((x**2).sum() / n + 1e-12) if scale else 1.0
    t = my-c * (R @ mx)
    return c, R, t

def similarity(P, c, R, t):
    return (c * (P @ R.T)) + t

def load_primary(path_txt):
    sk = load_skel(path_txt)
    sk = rem_zeros(sk)
    if sk.shape [0] == 0:
        print(f"Empty skeleton {path_txt}")
        return None
    return pick_primary(sk).astype(np.float32)

def align(src, ref):
    T = min(len(src), len(ref))
    src, ref = src[:T], ref[:T]
    c, R, t = umeyama(src.mean(0),ref.mean(0))
    out =np.empty_like(src, np.float32)
    for i in range(T):
        out[i]=similarity(src[i], c, R, t).astype(np.float32)
    return out, ref, (c, R, t)

def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def main():
    ap = argparse.ArgumentParser("align L and R skeletons to the reference M)")
    ap.add_argument("--ref", choices=["L", "M", "R"], default="M")
    ap.add_argument("--debug-plot", action="store_true")
    args = ap.parse_args()

    case = Case
    refv = args.ref.upper()

    skel_paths = {
        "L": os.path.join(paths.get("skel_l",""), f"{case}-L.txt"),
        "M": os.path.join(paths.get("skel_m",""), f"{case}-M.txt"),
        "R": os.path.join(paths.get("skel_r",""), f"{case}-R.txt"),
    }
    out_dir = paths.get("aligned","/home/kakarot/coding_task/outputs/aligned")
    out_files = {
        "L": os.path.join(out_dir,f"aligned_skeleton_L_{case}.npy"),
        "M": os.path.join(out_dir,f"aligned_skeleton_M_{case}.npy"),
        "R": os.path.join(out_dir,f"aligned_skeleton_R_{case}.npy"),
    }

    tracks={v:load_primary(p) for v,p in skel_paths.items()}
    avail = [t for t in tracks.values() if t is not None]

    T = min(len(t) for t in avail)
    ref_trim = tracks[refv][:T]

    save(out_files[refv], ref_trim)
    print(f"{refv} saved T= {T}")

    for v in ("L","M","R"):
        if v==refv:
            continue
        trk =tracks.get(v)
        if trk is None:
            print(f"{v} missing")
            continue
        aligned, _, (c, R, t) = align(trk[:T], ref_trim)
        save(out_files[v], aligned)
        print(f"{v} to {refv} saved T = {len(aligned)} ")

    if args.debug_plot:
        for v in ("L", "M", "R"):
            fp = out_files[v]
            if not os.path.isfile(fp):
                continue
            seq=np.load(fp)
            if len(seq)==0:
                continue
            pick = [0, len(seq)//2, len(seq)-1] if len(seq)>=3 else range(len(seq))
            for i in pick:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(111, projection = "3d")
                plot_skel3d(seq[i], bones = BONES, ax = ax, color="r")
                ax.set_title(f"{v} to {refv} :: frame {i}")
                ax.set_xlim([-2,2]); ax.set_ylim([-2,2]); ax.set_zlim([0,4])
                plt.tight_layout(); plt.show(block=True)

if __name__ == "__main__":
    main()