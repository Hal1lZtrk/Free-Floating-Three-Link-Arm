"""Parameters of the arm (data and conventions of the original course assignment).

The model is three rigid links of a UR5-type arm (link lengths and offsets follow the
UR5 DH parameters) connected by two revolute joints.  Nothing is fixed to the ground: the
chain floats freely and only the two internal joint torques act on it.

Conventions
-----------
* Twists are ``[v; w]`` (linear part first), expressed in the centre-of-gravity (CoG) frame
  of each link.
* Link *i* has an **input** and an **output** frame.  The CoG frame has pose
  ``(in_rot, in_pos)`` in the input frame and pose ``(I, out_pos)`` in the output frame.
* In the original schematic joint 1 is connected "backwards" (its reference side is the
  input frame of link 2) and joint 3 forwards.  This only changes the sign convention of the
  joint variable ``q`` and of the actuator torque; it is handled by ``JOINT_SIGN`` below.
"""
from dataclasses import dataclass

import numpy as np

from .spatial import rot_x


@dataclass(frozen=True)
class Link:
    name: str
    m: float                 # mass [kg]
    J: np.ndarray            # inertia tensor about the CoG, in the CoG frame [kg m^2]
    in_rot: np.ndarray       # 'R' block: orientation of the CoG frame in the input frame
    in_pos: np.ndarray       # 'R' block: position of the CoG frame in the input frame [m]
    out_pos: np.ndarray      # 'R1' block: position of the CoG frame in the output frame [m]
    P0: np.ndarray           # initial CoG position in the world frame [m]
    R0: np.ndarray           # initial orientation of the CoG frame in the world frame


I3 = np.eye(3)

LINK1 = Link(
    name="Link 1",
    m=8.393,
    J=np.array([[0.0149, 0.0, 0.0], [0.0, 0.3564, 0.0], [0.0, 0.0, 0.3553]]),
    in_rot=I3,
    in_pos=np.array([-0.425 + 0.2125, 0.0, 0.11336]),
    out_pos=np.array([0.2125, 0.0, 0.11336]),
    P0=np.array([-0.21249, -0.113336, 0.089159]),
    R0=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
)

LINK2 = Link(
    name="Link 2",
    m=2.33,
    J=np.array([[0.0025, 0.0, 0.0034], [0.0, 0.0551, 0.0], [0.0034, 0.0, 0.0546]]),
    in_rot=I3,
    in_pos=np.array([-0.39225 + 0.15, 0.0, 0.0265]),
    out_pos=np.array([0.15, 0.0, 0.0265]),
    P0=np.array([-0.66754, -0.026501, 0.089156]),
    R0=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
)

LINK3 = Link(
    name="Link 3",
    m=1.1219,
    J=np.array([[0.0149, 0.0, 0.0], [0.0, 0.3564, 0.0], [0.0, 0.0, 0.3553]]),
    in_rot=rot_x(np.pi / 2),
    in_pos=np.array([0.0, -0.01634, 0.10915 - 0.0018]),
    out_pos=np.array([0.0, -0.02561, 0.00193]),
    P0=np.array([-0.81724, -0.107352, 0.07281]),
    R0=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float),
)

LINKS = (LINK1, LINK2, LINK3)

# Sign of the assignment's joint variable q w.r.t. the chain order link1 -> link2 -> link3
# (joint 1 is connected backwards in the original schematic).  theta_chain = JOINT_SIGN * q.
JOINT_SIGN = (-1.0, +1.0)

# Actuation: both joints get tau(t) = AMPLITUDE * cos(OMEGA * t)  [Nm]
TORQUE_AMPLITUDE = (1.0, 1.0)
TORQUE_OMEGA = (5.0, 5.0)       # rad/s

T_END = 10.0                    # s   (simulation time of the assignment)
