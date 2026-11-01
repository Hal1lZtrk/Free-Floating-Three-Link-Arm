"""Figures.  Every figure is saved as PNG in ``results/figures`` and, by default, also opened in a
window (call :func:`show` at the end of a script).  Use ``--no-show`` or ``FREEFLOAT_NO_SHOW=1``
for headless runs.
"""
import os
import sys

import matplotlib

SHOW = not (os.environ.get("FREEFLOAT_NO_SHOW") or "--no-show" in sys.argv)
if SHOW and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    SHOW = False
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#d1495b", "#2a9d8f", "#1f5fbf"]
LINK_NAMES = ["Link 1", "Link 2", "Link 3"]
VIEW = dict(elev=24, azim=115)           # 3D viewpoint

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.dpi": 100})


def _finish(fig, path):
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=140)
    if not SHOW:
        plt.close(fig)


def show():
    if SHOW and plt.get_fignums():
        plt.show()


# ------------------------------------------------------------------ 3D drawing helpers
def link_polylines(arm, traj, k):
    """For every link: input-joint -> CoG -> output-joint polyline (world coordinates)."""
    lines = []
    for i, l in enumerate(arm.links):
        R, P = traj.rot[k, i], traj.pos[k, i]
        p_in = P + R @ (-l.in_rot.T @ l.in_pos)
        p_out = P + R @ (-l.out_pos)
        lines.append(np.array([p_in, P, p_out]))
    return lines


def setup_axes(ax, center, half=0.6):
    """Cube of side 2*half centred on the (fixed) centre of mass of the chain."""
    c = np.asarray(center, dtype=float)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(**VIEW)


def draw_arm(ax, arm, traj, k, trail=True):
    for i, line in enumerate(link_polylines(arm, traj, k)):
        ax.plot(*line.T, color=COLORS[i], lw=5, solid_capstyle="round", label=LINK_NAMES[i])
        ax.scatter(*line[1], color=COLORS[i], s=22, edgecolor="k", zorder=5)
    for p in traj.joints[k][1:3]:
        ax.scatter(*p, color="k", s=28, zorder=6)
    ax.scatter(*traj.com[k], marker="x", color="k", s=50, zorder=7, label="centre of mass")
    if trail:
        ax.plot(*traj.joints[:k + 1, 3].T, color="0.5", lw=0.8)


# ------------------------------------------------------------------ result figures
def plot_joints(traj, path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(traj.t, np.rad2deg(traj.q[:, 0]), color=COLORS[0], label=r"$q_1$ (joint 1)")
    ax[0].plot(traj.t, np.rad2deg(traj.q[:, 1]), color=COLORS[2], label=r"$q_3$ (joint 3)")
    ax[0].set_ylabel("joint angle [deg]")
    ax[0].legend(ncol=2)
    ax[0].set_title("Joint motion of the free-floating arm")
    ax[1].plot(traj.t, traj.q_dot[:, 0], color=COLORS[0])
    ax[1].plot(traj.t, traj.q_dot[:, 1], color=COLORS[2])
    ax[1].set_ylabel("joint rate [rad/s]")
    ax[2].plot(traj.t, traj.tau[:, 0], color=COLORS[0], label=r"$\tau_1$")
    ax[2].plot(traj.t, traj.tau[:, 1], color=COLORS[2], ls="--", label=r"$\tau_3$")
    ax[2].set_ylabel("actuator torque [Nm]")
    ax[2].set_xlabel("time [s]")
    _finish(fig, path)


def plot_conservation(traj, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].semilogy(traj.t, np.maximum(np.linalg.norm(traj.lin_momentum, axis=1), 1e-18), color=COLORS[0])
    ax[0].set_title("Linear momentum  |p|  [kg m/s]")
    ax[1].semilogy(traj.t, np.maximum(np.linalg.norm(traj.ang_momentum, axis=1), 1e-18), color=COLORS[1])
    ax[1].set_title("Angular momentum  |L|  [kg m²/s]")
    drift = np.linalg.norm(traj.com - traj.com[0], axis=1)
    ax[2].semilogy(traj.t, np.maximum(drift, 1e-18), color=COLORS[2])
    ax[2].set_title("Centre-of-mass drift  [m]")
    for a in ax:
        a.set_xlabel("time [s]")
    fig.suptitle("Conservation laws (numerical error only, should stay near machine precision)")
    _finish(fig, path)


def plot_energy(traj, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(traj.t, traj.kinetic, color=COLORS[2], lw=2, label="kinetic energy T(t)")
    ax[0].plot(traj.t, traj.work, color="k", ls="--", label=r"work of the actuators $\int \tau\,\dot q\,dt$")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("energy [J]")
    ax[0].legend()
    ax[1].semilogy(traj.t, np.maximum(np.abs(traj.kinetic - traj.work), 1e-18), color=COLORS[0])
    ax[1].set_xlabel("time [s]")
    ax[1].set_title("|T - W|  [J]")
    fig.suptitle("Energy balance")
    _finish(fig, path)


def plot_twists(traj, path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i in range(3):
        ax[0].plot(traj.t, np.linalg.norm(traj.twist[:, i, :3], axis=1), color=COLORS[i], label=LINK_NAMES[i])
        ax[1].plot(traj.t, np.linalg.norm(traj.twist[:, i, 3:], axis=1), color=COLORS[i])
    ax[0].set_ylabel(r"$\|v\|$ of the CoG [m/s]")
    ax[1].set_ylabel(r"$\|\omega\|$ [rad/s]")
    ax[1].set_xlabel("time [s]")
    ax[0].legend(ncol=3)
    ax[0].set_title("Body twists (screw theory): linear and angular velocity of every link")
    _finish(fig, path)


def plot_snapshots(arm, traj, times, path, ncols=4):
    idx = [int(np.argmin(np.abs(traj.t - t))) for t in times]
    rows = int(np.ceil(len(idx) / ncols))
    fig = plt.figure(figsize=(3.6 * ncols, 3.4 * rows))
    for n, k in enumerate(idx):
        ax = fig.add_subplot(rows, ncols, n + 1, projection="3d")
        draw_arm(ax, arm, traj, k, trail=False)
        setup_axes(ax, traj.com[0])
        ax.set_title(f"t = {traj.t[k]:.1f} s", fontsize=10)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    fig.suptitle("Snapshots of the free-floating arm (x marks the centre of mass, which never moves)")
    _finish(fig, path)


def plot_tip_path(arm, traj, path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(3):
        ax.plot(*traj.pos[:, i].T, color=COLORS[i], lw=1.4, label=f"CoG {LINK_NAMES[i]}")
    ax.plot(*traj.joints[:, 3].T, color="0.4", lw=1.0, ls="--", label="free end")
    ax.scatter(*traj.com[0], marker="x", color="k", s=70, label="centre of mass (fixed)")
    setup_axes(ax, traj.com[0])
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Paths of the three centres of gravity")
    _finish(fig, path)
