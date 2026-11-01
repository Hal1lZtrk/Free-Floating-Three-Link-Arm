"""3D animation of the free-floating arm.

    python run_animation.py            # interactive window
    python run_animation.py --gif      # writes results/free_floating_arm.gif
    python run_animation.py --mp4      # writes results/free_floating_arm.mp4 (needs ffmpeg)
    python run_animation.py --t-end 30 # longer run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freefloat import ThreeLinkArm                       # noqa: E402
from freefloat.animation import export, interactive      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--no-show", action="store_true", help="(accepted for symmetry with run_simulation.py)")
    a = ap.parse_args()
    arm = ThreeLinkArm()
    traj = arm.simulate(a.t_end, 0.01)
    if a.gif:
        print(export(arm, traj, os.path.join(RESULTS, "free_floating_arm.gif"), fps=20, stride=5))
    elif a.mp4:
        print(export(arm, traj, os.path.join(RESULTS, "free_floating_arm.mp4"), fps=25, stride=4))
    else:
        interactive(arm, traj)


if __name__ == "__main__":
    main()
