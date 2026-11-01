"""Simulate the free-floating three-link arm and produce all figures.

    python run_simulation.py                 # 10 s, figures open in windows
    python run_simulation.py --t-end 30      # 30 s
    python run_simulation.py --no-show       # only write the files (no windows)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freefloat import ThreeLinkArm                       # noqa: E402
from freefloat import plotting as pl                     # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(RESULTS, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=10.0, help="simulation time [s] (default 10, as in the assignment)")
    ap.add_argument("--dt", type=float, default=0.01, help="output sampling [s]")
    ap.add_argument("--no-show", action="store_true", help="do not open figure windows")
    a = ap.parse_args()

    arm = ThreeLinkArm()
    traj = arm.simulate(a.t_end, a.dt)

    summary = dict(
        t_end=a.t_end,
        solver="scipy DOP853, rtol=1e-11, atol=1e-12",
        rhs_evaluations=int(traj.nfev),
        total_mass_kg=float(sum(l.m for l in arm.links)),
        q_final_deg=[float(v) for v in np.rad2deg(traj.q[-1])],
        q_min_deg=[float(v) for v in np.rad2deg(traj.q.min(axis=0))],
        q_max_deg=[float(v) for v in np.rad2deg(traj.q.max(axis=0))],
        peak_joint_rate_rad_s=[float(v) for v in np.abs(traj.q_dot).max(axis=0)],
        peak_kinetic_energy_J=float(traj.kinetic.max()),
        max_linear_momentum=float(np.abs(traj.lin_momentum).max()),
        max_angular_momentum=float(np.abs(traj.ang_momentum).max()),
        max_com_drift_m=float(np.linalg.norm(traj.com - traj.com[0], axis=1).max()),
        max_energy_balance_error_J=float(np.abs(traj.kinetic - traj.work).max()),
        max_rotation_orthogonality_error=float(max(np.abs(R.T @ R - np.eye(3)).max()
                                                   for Rs in traj.rot for R in Rs)),
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(os.path.join(RESULTS, "trajectory.npz"), t=traj.t, q=traj.q, q_dot=traj.q_dot,
                        tau=traj.tau, pos=traj.pos, rot=traj.rot, twist=traj.twist, joints=traj.joints,
                        kinetic=traj.kinetic, work=traj.work, com=traj.com)
    for k, v in summary.items():
        print(f"{k:34s} {v}")

    pl.plot_joints(traj, os.path.join(FIG, "joint_motion.png"))
    pl.plot_conservation(traj, os.path.join(FIG, "conservation_laws.png"))
    pl.plot_energy(traj, os.path.join(FIG, "energy_balance.png"))
    pl.plot_twists(traj, os.path.join(FIG, "body_twists.png"))
    pl.plot_snapshots(arm, traj, np.linspace(0, a.t_end, 8), os.path.join(FIG, "snapshots.png"))
    pl.plot_tip_path(arm, traj, os.path.join(FIG, "cog_paths.png"))
    pl.show()


if __name__ == "__main__":
    main()
