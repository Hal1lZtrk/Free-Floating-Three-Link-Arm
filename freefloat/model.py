"""Free-floating three-link arm: kinematics, forward dynamics and simulation.

Formulation
-----------
Every link has a body-fixed frame at its centre of gravity (CoG).  The state is

    y = [P1 (3), R1 (9), V1 (6), theta (2), theta_dot (2), W (1)]

i.e. the pose and body twist of link 1 (the "floating base"), the two relative joint
angles and the work done by the actuators (used for the energy check).  Poses and twists of links 2 and 3 follow from the chain kinematics

    T_{c_j, c_{j+1}} = T_a  Rot_z(theta_j)  T_b^{-1}            (pose of link j+1 in link j)
    V_{j+1}          = X_j V_j + s_j theta_dot_j                (twists, CoG frames)

with ``X_j = Ad(T_{c_{j+1}, c_j})`` and ``s_j`` the joint screw.  Differentiating the twist
relation gives the accelerations.  Together with the Newton-Euler equation of every link

    M_i a_i + [m w x v ; w x J w] = W_i                         (body frame)

and the joint wrenches (5 unknown constraint components + the known actuator torque about the
joint axis, action = -reaction) this is one linear system in
``[a_1, theta_ddot, lambda_1, lambda_2]`` (18 unknowns) that is solved at every time step.
No generalised coordinates are chosen for the base, so there are no singularities and the
equations are the Euler equations of the rigid bodies plus the joint constraints (the
bond-graph structure of the original assignment), solved with an adaptive ODE integrator.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import params as P
from .spatial import (ad, adjoint, pose_compose, pose_inverse, rot_z, skew,
                      spatial_inertia)

E6 = np.zeros(6)
E6[5] = 1.0                      # unit twist: rotation about the joint axis (z)
PHI = np.zeros((6, 5))           # constraint wrench directions in the joint frame
PHI[:5, :5] = np.eye(5)          # Fx Fy Fz tau_x tau_y (tau_z is the actuator)


@dataclass
class Trajectory:
    t: np.ndarray                 # (N,)
    pos: np.ndarray               # (N, 3, 3)    CoG position of each link, world frame
    rot: np.ndarray               # (N, 3, 3, 3) orientation of each link, world frame
    twist: np.ndarray             # (N, 3, 6)    body twists [v; w] in the CoG frames
    theta: np.ndarray             # (N, 2)       chain joint angles (link j -> j+1)
    theta_dot: np.ndarray         # (N, 2)
    q: np.ndarray                 # (N, 2)       joint variables with the assignment's sign convention
    q_dot: np.ndarray
    tau: np.ndarray               # (N, 2)       actuator torques (assignment convention)
    joints: np.ndarray            # (N, 4, 3)    base end, joint 1, joint 2, free end (world)
    kinetic: np.ndarray           # (N,)
    work: np.ndarray              # (N,)         integral of actuator power
    lin_momentum: np.ndarray      # (N, 3)       total linear momentum, world
    ang_momentum: np.ndarray      # (N, 3)       total angular momentum about the world origin
    com: np.ndarray               # (N, 3)       centre of mass of the whole chain
    nfev: int = 0


class ThreeLinkArm:
    def __init__(self, links=P.LINKS, joint_sign=P.JOINT_SIGN,
                 amplitude=P.TORQUE_AMPLITUDE, omega=P.TORQUE_OMEGA):
        self.links = tuple(links)
        self.sign = np.asarray(joint_sign, dtype=float)
        self.amp = np.asarray(amplitude, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.n = len(self.links)
        self.M = [spatial_inertia(l.m, l.J) for l in self.links]
        # joint frames: J on link j (output frame), J' on link j+1 (input frame), both in CoG coordinates
        self.Ta = [(np.eye(3), -l.out_pos) for l in self.links[:-1]]
        # pose of J' (input frame of link j+1) in the CoG frame of link j+1
        self.Tb = [(l.in_rot.T, -l.in_rot.T @ l.in_pos) for l in self.links[1:]]
        self.G = [adjoint(R, p) for R, p in self.Tb]
        self.Ginv_T = [np.linalg.inv(G).T for G in self.G]
        self.s = [G @ E6 for G in self.G]

    # ----------------------------------------------------------------- actuation
    def torque(self, t):
        """Actuator torques in the assignment's convention [Nm]."""
        return self.amp * np.cos(self.omega * t)

    def chain_torque(self, t):
        """Torque on the downstream link about +z of the joint frame (chain convention)."""
        return self.sign * self.torque(t)

    # ----------------------------------------------------------------- kinematics
    def relative_pose(self, j, theta):
        """Pose of link j+1 in link j: (R, p) = T_a Rot_z(theta) T_b^{-1}."""
        Ra, pa = self.Ta[j]
        Rb, pb = self.Tb[j]
        Rbi, pbi = pose_inverse(Rb, pb)
        R1, p1 = pose_compose(Ra, pa, rot_z(theta), np.zeros(3))
        return pose_compose(R1, p1, Rbi, pbi)

    def poses(self, P1, R1, theta):
        Ps, Rs = [np.asarray(P1, float)], [np.asarray(R1, float)]
        for j in range(self.n - 1):
            Rr, pr = self.relative_pose(j, theta[j])
            Rn, Pn = pose_compose(Rs[-1], Ps[-1], Rr, pr)
            Rs.append(Rn)
            Ps.append(Pn)
        return Ps, Rs

    def twists(self, V1, theta, theta_dot):
        Vs = [np.asarray(V1, float)]
        Xs = []
        for j in range(self.n - 1):
            Rr, pr = self.relative_pose(j, theta[j])
            X = adjoint(*pose_inverse(Rr, pr))           # twist of link j in the frame of link j+1
            Xs.append(X)
            Vs.append(X @ Vs[-1] + self.s[j] * theta_dot[j])
        return Vs, Xs

    def joint_points(self, Ps, Rs):
        """World positions: free input end of link 1, joint 1, joint 2, free output end of link 3."""
        l1, l3 = self.links[0], self.links[-1]
        pts = [Ps[0] + Rs[0] @ (-l1.in_rot.T @ l1.in_pos)]
        for j in range(self.n - 1):
            pts.append(Ps[j] + Rs[j] @ self.Ta[j][1])
        pts.append(Ps[-1] + Rs[-1] @ (-l3.out_pos))
        return np.array(pts)

    # ----------------------------------------------------------------- dynamics
    def forward_dynamics(self, t, V1, theta, theta_dot):
        """Accelerations: returns (a1, theta_ddot) and the joint constraint wrenches."""
        n, nj = self.n, self.n - 1
        Q = self.chain_torque(t)
        Vs, Xs = self.twists(V1, theta, theta_dot)
        nz = 6 + nj + 5 * nj
        # accelerations of the links as affine functions of z = [a1, th_dd, lam_1, lam_2]
        A = [np.zeros((6, nz)) for _ in range(n)]
        c = [np.zeros(6) for _ in range(n)]
        A[0][:, :6] = np.eye(6)
        for j in range(nj):
            A[j + 1] = Xs[j] @ A[j]
            A[j + 1][:, 6 + j] += self.s[j]
            c[j + 1] = Xs[j] @ c[j] - ad(self.s[j] * theta_dot[j]) @ (Xs[j] @ Vs[j])
        # wrench exerted by joint j on link j+1 (CoG frame): W = Ginv_T (PHI lam + e_z Q)
        Wm, w0 = [], []
        for j in range(nj):
            Lm = np.zeros((5, nz))
            Lm[:, 6 + nj + 5 * j: 6 + nj + 5 * (j + 1)] = np.eye(5)
            Wm.append(self.Ginv_T[j] @ PHI @ Lm)
            w0.append(self.Ginv_T[j] @ (E6 * Q[j]))
        rows, rhs = [], []
        for i in range(n):
            v, w = Vs[i][:3], Vs[i][3:]
            m, J = self.links[i].m, self.links[i].J
            bias = np.concatenate([m * np.cross(w, v), np.cross(w, J @ w)])
            Mi = self.M[i]
            Zi = Mi @ A[i]
            r = -(Mi @ c[i]) - bias
            if i >= 1:                                   # wrench from joint i-1 on this link
                Zi = Zi - Wm[i - 1]
                r = r + w0[i - 1]
            if i < nj:                                   # reaction of joint i on this link
                Zi = Zi + Xs[i].T @ Wm[i]
                r = r - Xs[i].T @ w0[i]
            rows.append(Zi)
            rhs.append(r)
        z = np.linalg.solve(np.vstack(rows), np.concatenate(rhs))
        return z[:6], z[6:6 + nj], z[6 + nj:]

    def _rhs(self, t, y):
        R1 = y[3:12].reshape(3, 3)
        V1 = y[12:18]
        th, thd = y[18:20], y[20:22]
        a1, thdd, _ = self.forward_dynamics(t, V1, th, thd)
        d = np.empty(23)
        d[0:3] = R1 @ V1[:3]
        d[3:12] = (R1 @ skew(V1[3:])).reshape(9)
        d[12:18] = a1
        d[18:20] = thd
        d[20:22] = thdd
        d[22] = float(np.sum(self.torque(t) * self.sign * thd))      # actuator power
        return d

    # ----------------------------------------------------------------- simulation
    def initial_state(self):
        l1 = self.links[0]
        y = np.zeros(23)
        y[0:3] = l1.P0
        y[3:12] = l1.R0.reshape(9)
        return y

    def simulate(self, t_end=P.T_END, dt=0.01, rtol=1e-11, atol=1e-12, y0=None, method="DOP853"):
        t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
        y0 = self.initial_state() if y0 is None else y0
        sol = solve_ivp(self._rhs, (0.0, t_end), y0, method=method, t_eval=t_eval,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return self._postprocess(sol.t, sol.y.T, sol.nfev)

    def _postprocess(self, t, Y, nfev=0):
        N, n = len(t), self.n
        pos = np.zeros((N, n, 3))
        rot = np.zeros((N, n, 3, 3))
        tw = np.zeros((N, n, 6))
        joints = np.zeros((N, n + 1, 3))
        kin = np.zeros(N)
        L = np.zeros((N, 3))
        Pm = np.zeros((N, 3))
        com = np.zeros((N, 3))
        mtot = sum(l.m for l in self.links)
        th, thd = Y[:, 18:20], Y[:, 20:22]
        tau = np.array([self.torque(ti) for ti in t])
        for k in range(N):
            Ps, Rs = self.poses(Y[k, 0:3], Y[k, 3:12].reshape(3, 3), th[k])
            Vs, _ = self.twists(Y[k, 12:18], th[k], thd[k])
            for i in range(n):
                pos[k, i], rot[k, i], tw[k, i] = Ps[i], Rs[i], Vs[i]
                v_w, w_w = Rs[i] @ Vs[i][:3], Rs[i] @ Vs[i][3:]
                p_w = self.links[i].m * v_w
                Pm[k] += p_w
                L[k] += np.cross(Ps[i], p_w) + Rs[i] @ self.links[i].J @ Vs[i][3:]
                com[k] += self.links[i].m * Ps[i] / mtot
                kin[k] += 0.5 * Vs[i] @ self.M[i] @ Vs[i]
            joints[k] = self.joint_points(Ps, Rs)
        q = self.sign * th
        q_dot = self.sign * thd
        work = Y[:, 22]
        return Trajectory(t=t, pos=pos, rot=rot, twist=tw, theta=th, theta_dot=thd, q=q,
                          q_dot=q_dot, tau=tau, joints=joints, kinetic=kin, work=work,
                          lin_momentum=Pm, ang_momentum=L, com=com, nfev=nfev)
