"""3D animation of the free-floating arm (matplotlib).  Opens a window or exports GIF / MP4."""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from . import plotting as pl


def build(arm, traj, stride=4, half=0.6):
    idx = np.arange(0, len(traj.t), stride)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    def frame(n):
        k = idx[n]
        ax.cla()
        pl.draw_arm(ax, arm, traj, k)
        pl.setup_axes(ax, traj.com[0], half)
        ax.set_title(f"Free-floating three-link arm   t = {traj.t[k]:5.2f} s\n"
                     f"q1 = {np.rad2deg(traj.q[k, 0]):7.1f}°   q3 = {np.rad2deg(traj.q[k, 1]):7.1f}°", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)

    return fig, FuncAnimation(fig, frame, frames=len(idx), interval=40, repeat=True), len(idx)


def export(arm, traj, path, fps=25, stride=4, dpi=90):
    """Write a .gif (Pillow) or .mp4 (ffmpeg) file."""
    fig, anim, n = build(arm, traj, stride)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".mp4"):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found: install it or use --gif")
        writer = FFMpegWriter(fps=fps, bitrate=1800)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path


def interactive(arm, traj, stride=2):
    fig, anim, _ = build(arm, traj, stride)
    fig._anim = anim                       # keep a reference
    plt.show()
