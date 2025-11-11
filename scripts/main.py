import os
import sys
import subprocess
from project_utils import Case

case = Case
script = os.path.dirname(os.path.abspath(__file__))
Py= sys.executable
Scripts = {
    "skeleton":os.path.join(script, "skeleton.py"),
    "objects": os.path.join(script, "objects.py"),
    "rgb_overlay":os.path.join(script, "rgb_overlay.py"),
    "hoi":os.path.join(script, "hoi.py")}

def run(name, args=None):
    args = args or []
    path = Scripts[name]
    if not os.path.isfile(path):
        sys.exit(1)
    cmd = [Py, path] + args
    print(f"Running {name} script:")
    try:
        subprocess.run(cmd, check =True)
    except subprocess.CalledProcessError as e:
        print(f"File {name} failed (exit {e.returncode})")
        sys.exit(e.returncode)

def main():
    # Align skeletons, the default reference frame is M, change here if you want
    run("skeleton",["--ref", "M"])
    # Detections of objects per view
    run("objects")
    # Draw skeleton overlays per view L M R
    run("rgb_overlay")
    # Human Object Interaction
    run("hoi")

if __name__ == "__main__":
    main()