#!/usr/bin/env python3
"""MuJoCo model of the SAR hexapod: frame, 18 servos, USC-32, phone, battery box, RPLIDAR, straps, wiring.

  pip install mujoco pillow
  python3 sim/hexapod_sim.py build                # writes sim/hexapod.xml (every command rebuilds it first)
  python3 sim/hexapod_sim.py view                 # 3D viewer, robot standing (plain python on macOS, not mjpython)
  python3 sim/hexapod_sim.py view --vx 1          # walking: hexapod.pose() frames, the same pulses the USC-32 gets
  python3 sim/hexapod_sim.py render               # PNG stills + walk.gif into sim/renders/
  python3 sim/hexapod_sim.py selftest

Leg lengths, body mounts, channels, DIR and TRIM are imported from python/hexapod.py, so measuring the robot
and editing that file (then rebuilding) updates this model too. Joint zero is the IK's zero: femur level,
tibia straight down. A pulse becomes a joint angle the way pose() made it, deg = DIR*(us - 1500 - TRIM)/US_PER_DEG,
so the sim hits the same 500/2500 us limits as the real servos (joint ranges are those limits).
Coxa 30 and femur 85 mm are measured; tibia 90 mm, the body mounts and every other size here are read off
photos, and masses are datasheet values or guesses (MASS). Treat sim speeds and clearances as unmeasured.
Viewer: the collision shapes are hidden in geom group 3 (press 3), sites in group 4.
"""
import argparse, math, os, sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import hexapod as hx  # noqa: E402  leg lengths, mounts, channels, DIR, TRIM, gait

XML = os.path.join(HERE, "hexapod.xml")
C, FE, T = hx.COXA, hx.FEMUR, hx.TIBIA
MM = 0.001
MASS = dict(servo=55,                    # g; MG995/MG996R datasheet
            phone=215,                   # Galaxy M33 5G spec
            lidar=170,                   # RPLIDAR A1M8 spec
            battery=4 * 47 + 70,         # 4x INR18650-26E (~47 g) + printed case/BMS (guess)
            usc=25, plate=80, bracket=22, femur=15, tibia=25, horn=5, riser=15, cables=40, standoffs=12)  # guesses
STACK_Z = 44.0                           # phone/battery stack sits on a riser, just above the coxa brackets
BOARD = (-22.0, 0.0, 32.5)               # USC-32 PCB bottom, on a foam block behind the stack

MATS = {  # rgba, specular, shininess, reflectance, emission
    "alu": (".045 .045 .05 1", .55, .55, 0, 0), "servo": (".08 .08 .085 1", .25, .3, 0, 0),
    "seam": (".15 .15 .16 1", .2, .2, 0, 0), "label": (".78 .6 .22 1", .7, .6, 0, 0),
    "brass": (".8 .64 .3 1", .9, .8, 0, 0), "screw": (".72 .72 .75 1", 1, .9, .05, 0),
    "socket": (".1 .1 .1 1", .2, .2, 0, 0), "pcb": (".07 .3 .72 1", .35, .5, 0, 0),
    "hdr_s": (".95 .78 .1 1", .3, .3, 0, 0), "hdr_k": (".04 .04 .04 1", .3, .3, 0, 0),
    "pin": (".85 .7 .3 1", .9, .8, 0, 0), "chip": (".05 .05 .05 1", .4, .5, 0, 0),
    "term": (".12 .38 .85 1", .3, .4, 0, 0), "led": (".25 .55 1 1", 1, 1, 0, 2),
    "smd": (".55 .45 .3 1", .3, .3, 0, 0), "plug": (".06 .06 .06 1", .2, .3, 0, 0),
    "phone": (".07 .09 .17 1", .9, .95, .08, 0), "glass": (".02 .02 .025 1", 1, 1, .25, 0),
    "ring": (".6 .6 .63 1", 1, .9, .1, 0), "flash": (".95 .9 .7 1", .5, .5, 0, .3),
    "pla": (".44 .5 .58 1", .1, .2, 0, 0), "lidar": (".07 .07 .075 1", .2, .25, 0, 0),
    "lidar2": (".13 .13 .14 1", .3, .3, 0, 0), "window": (".01 .01 .015 1", 1, 1, .3, 0),
    "velcro": (".2 .2 .21 1", 0, 0, 0, 0), "foam": (".09 .09 .09 1", 0, 0, 0, 0),
    "tie": (".05 .05 .05 1", .4, .4, 0, 0), "tape": (".95 .95 .93 1", .15, .1, 0, 0),
    "nylon": (".9 .9 .86 1", .2, .2, 0, 0), "rubber": (".04 .04 .04 1", .3, .3, 0, 0),
    "wire_o": ("1 .5 .04 1", .4, .5, 0, 0), "wire_r": (".85 .1 .07 1", .4, .5, 0, 0),
    "wire_b": (".36 .18 .07 1", .4, .5, 0, 0), "wire_k": (".04 .04 .04 1", .4, .5, 0, 0),
}


def num(*v):
    return " ".join(f"{x:.6g}" for x in v)


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


class Frame:
    """Geoms for one MuJoCo body, written in mm in a local frame (origin o, rotation R in the body)."""

    def __init__(self, out, o=(0, 0, 0), R=np.eye(3)):
        self.out, self.o, self.R = out, np.asarray(o, float), np.asarray(R, float)

    def at(self, o=(0, 0, 0), R=np.eye(3)):
        return Frame(self.out, self.p(o), self.R @ R)

    def p(self, v):
        return self.o + self.R @ np.asarray(v, float)

    def geom(self, attrs, mat, cls):
        self.out.append(f'<geom class="{cls}"{f" material={chr(34)}{mat}{chr(34)}" if mat else ""} {attrs}/>')

    def box(self, lo, hi, mat, cls="vis"):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.geom(f'type="box" pos="{num(*self.p((lo + hi) / 2) * MM)}" quat="{num(*quat(self.R))}" '
                  f'size="{num(*abs(hi - lo) / 2 * MM)}"', mat, cls)

    def cyl(self, a, b, r, mat, cls="vis", typ="cylinder"):
        self.geom(f'type="{typ}" fromto="{num(*self.p(a) * MM, *self.p(b) * MM)}" size="{r * MM:.6g}"', mat, cls)

    def cap(self, a, b, r, mat, cls="vis"):
        self.cyl(a, b, r, mat, cls, "capsule")

    def sph(self, c, r, mat, cls="vis"):
        self.geom(f'type="sphere" pos="{num(*self.p(c) * MM)}" size="{r * MM:.6g}"', mat, cls)

    def ell(self, c, radii, mat):
        self.geom(f'type="ellipsoid" pos="{num(*self.p(c) * MM)}" quat="{num(*quat(self.R))}" '
                  f'size="{num(*np.asarray(radii) * MM)}"', mat, "vis")

    def mesh(self, name, mat, cls="vis"):
        self.geom(f'type="mesh" mesh="{name}" pos="{num(*self.o * MM)}" quat="{num(*quat(self.R))}"', mat, cls)


def axes(fr, o, x, z):
    """Sub-frame at o whose local x and z point along the given parent directions."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    return fr.at(o, np.column_stack([x, np.cross(z, x), z]))


# ---- flat parts: 2D outline extruded into a mesh ----
MESHES = {}


def cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def ear_clip(P):
    """Triangles of a simple CCW polygon (works for the concave tibia claw)."""
    idx, tris = list(range(len(P))), []
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross2(P[i], P[j], P[l]) <= 1e-9:
                continue
            if any(cross2(P[i], P[j], P[m]) >= 0 and cross2(P[j], P[l], P[m]) >= 0 and cross2(P[l], P[i], P[m]) >= 0
                   for m in idx if m not in (i, j, l)):
                continue
            tris.append((i, j, l))
            del idx[k]
            break
        else:
            raise ValueError("outline self-intersects")
    return tris + [tuple(idx)]


def prism(name, poly, t0, t1):
    """Register a plate mesh: outline in local xy (mm), thickness from z=t0 to z=t1."""
    if name not in MESHES:
        P = [tuple(map(float, p)) for p in poly]
        if sum(cross2((0, 0), P[i - 1], P[i]) for i in range(len(P))) < 0:
            P = P[::-1]
        n = len(P)
        f = [t for a, b, c in ear_clip(P) for t in ((a + n, b + n, c + n), (c, b, a))]
        f += [t for i in range(n) for t in ((i, (i + 1) % n, (i + 1) % n + n), (i, (i + 1) % n + n, i + n))]
        MESHES[name] = ([(x, y, t0) for x, y in P] + [(x, y, t1) for x, y in P], f)
    return name


def hull(pts):
    pts = sorted(set((round(x, 5), round(y, 5)) for x, y in pts))

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and cross2(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    return half(pts)[:-1] + half(pts[::-1])[:-1]


def circles(*cs, n=28):
    """Convex outline around circles (x, y, r): rounded plates, links, brackets."""
    return hull([(x + r * math.cos(a), y + r * math.sin(a)) for x, y, r in cs
                 for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])


def rrect(w, h, r):
    return circles(*[(sx * (w / 2 - r), sy * (h / 2 - r), r) for sx in (-1, 1) for sy in (-1, 1)], n=12)


def bez(p0, p1, p2, p3, n=14):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
    return list((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def claw():
    """Tibia plate (x outward, z up, knee axis at 0): covers the knee servo, then curves out and down to the tip
    at (0, -T). ponytail: shape traced from the photos, scaled by TIBIA; the slot cut-out is left out."""
    inner = bez((12, -15), (0.36 * T, -0.12 * T), (0.42 * T, -0.62 * T), (-1.5, -T + 1.5))
    outer = bez((2, -T), (0.62 * T, -0.55 * T), (13 + 0.5 * T, 40), (13, 38))
    return [(-13, -15)] + inner + outer + [(-13, 38)], inner, outer[::-1]


# ---- hardware ----
def screw(fr, c, n, r=2.2):
    c, n = np.asarray(c, float), np.asarray(n, float)
    fr.cyl(c, c + 1.3 * n, r, "screw")
    fr.cyl(c + 1.25 * n, c + 1.4 * n, r * 0.45, "socket")


def servo(fr, label=True):
    """Standard servo (MG995/MG996R size) in its own frame: output shaft at the case top, shaft +z, case toward -x.
    Returns where the lead leaves the case."""
    fr.box((-30.5, -10, -37), (10, 10, 0), "servo")
    for z in (-4.6, -31):
        fr.box((-30.7, -10.2, z), (10.2, 10.2, z + 0.6), "seam")
    fr.box((-37.25, -9.5, -10), (16.75, 9.5, -7.5), "servo")            # mounting ears
    for x in (-34, 13.5):
        for y in (-5, 5):
            screw(fr, (x, y, -7.5), (0, 0, 1), 1.8)
    fr.cyl((0, 0, 0), (0, 0, 3), 6.5, "servo")                           # top boss
    fr.cyl((-10.5, 0, 0), (-10.5, 0, 1.2), 4.5, "servo")                 # gear-train bump
    fr.cyl((0, 0, 3), (0, 0, 4.5), 3, "brass")                           # output spline
    if label:
        for y in (-10.08, 10.02):
            fr.box((-27, y, -29), (-3, y + 0.06, -12), "label")
    fr.box((-31.5, -3.2, -36), (-30.5, 3.2, -32), "servo")               # lead grommet
    return fr.p((-31.5, 0, -34))


def horn(fr):
    """Round aluminium horn (25 mm) on the spline of the servo whose frame is fr."""
    fr.cyl((0, 0, 4.5), (0, 0, 7.5), 12.5, "alu")


def horn_screws(fr, c, n):
    c, n = np.asarray(c, float), np.asarray(n, float)
    u = np.cross(n, (0, 0, 1) if abs(n[2]) < 0.9 else (1, 0, 0))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    for a in range(4):
        th = math.pi / 4 + a * math.pi / 2
        screw(fr, c + 8 * (math.cos(th) * u + math.sin(th) * v), n, 1.8)
    screw(fr, c, n, 2.4)


def spline(pts, k=4):
    P = [np.asarray(p, float) for p in pts]
    P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return out + [P[-2]]


def cable(fr, pts, wires=("wire_b", "wire_r", "wire_o"), r=0.6):
    """Servo lead (flat 3-wire ribbon) or, with one wire, a round cable through the points (mm, frame coords)."""
    pts = spline(pts)
    off = []
    for i in range(len(pts)):
        t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        n = np.cross(t, (0, 0, 1))
        n = n if np.linalg.norm(n) > 1e-6 else np.cross(t, (1, 0, 0))
        off.append(n / (np.linalg.norm(n) + 1e-12) * 2 * r)
    for i in range(len(pts) - 1):
        if np.linalg.norm(pts[i + 1] - pts[i]) < 1e-3:
            continue
        for k, w in enumerate(wires):
            s = k - (len(wires) - 1) / 2
            fr.cap(pts[i] + s * off[i], pts[i + 1] + s * off[i + 1], r, w)


def inertial(items):
    """<inertial> from (grams, centre mm, box size mm) parts."""
    m = np.array([g for g, _, _ in items], float) / 1000
    c = np.array([cc for _, cc, _ in items], float) * MM
    com = m @ c / m.sum()
    I = np.zeros((3, 3))
    for mi, ci, (_, _, s) in zip(m, c, items):
        a, b, h = np.asarray(s, float) * MM
        d = ci - com
        I += np.diag(mi / 12 * np.array([b * b + h * h, a * a + h * h, a * a + b * b])) + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return (f'<inertial pos="{num(*com)}" mass="{m.sum():.6g}" '
            f'fullinertia="{num(I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])}"/>')


def plug_top(ch):
    """Board-frame position where the servo plug on channel ch takes its lead (S1-16 on the -y edge, 17-32 on +y)."""
    k, ys = (ch - 1) % 16, (-1 if ch <= 16 else 1)
    return np.array([-19.05 + k * 2.54, ys * 17.76, 18.0])


def ranges(name):
    """Joint limits (rad): the angles pose() maps to 500 and 2500 us for this leg's DIR and TRIM."""
    return [sorted(math.radians(d * (us - 1500 - t) / hx.US_PER_DEG) for us in (500, 2500))
            for d, t in zip(hx.DIR[name], hx.TRIM[name])]


# ---- assembly ----
def usc_board(tor):
    b = tor.at(BOARD)
    b.box((-20, -10, -12), (20, 10, 0), "foam")                           # foam block on the top plate
    b.box((-31, -22, 0), (31, 22, 1.6), "pcb")
    for ys in (-1, 1):                                                   # 2 x 16 servo headers: S / V+ / GND rows
        for dy, mat in ((20.3, "hdr_s"), (17.76, "hdr_k"), (15.22, "hdr_k")):
            y = ys * dy
            b.box((-20.32, y - 1.27, 1.6), (18.8, y + 1.27, 4.1), mat)
            for k in range(16):
                x = -19.05 + k * 2.54
                b.box((x - .32, y - .32, 4.1), (x + .32, y + .32, 10), "pin")
    b.box((-6, -6, 1.6), (6, 6, 2.8), "chip")                             # MCU
    b.box((-12, 2, 1.6), (-8, 4, 3.2), "ring")                           # crystal
    b.box((-31.5, 4, 1.6), (-24, 12, 5.6), "ring")                       # mini-USB (to the phone's OTG lead)
    b.box((22, -11, 1.6), (30, 11, 11), "term")                          # VS / GND / VSS terminal
    for y in (-7, 0, 7):
        b.cyl((26, y, 11), (26, y, 11.3), 2, "screw")
    b.box((14, 6, 1.6), (16.5, 7.6, 2.6), "led")
    for x, y in ((9, -4), (9, 0), (-10, -8), (-14, 8), (12, 10), (-2, 10), (18, -8)):
        b.box((x - 1, y - .6, 1.6), (x + 1, y + .6, 2.2), "smd")
    return b


def stack(tor):
    """Phone upright (landscape, rear camera forward) against the grey battery box, lidar on top, strapped."""
    R = np.column_stack([(0, 1, 0), (0, 0, 1), (1, 0, 0)])               # plate outline in the y-z plane
    z0 = STACK_Z
    tor.box((18, -9, 20.5), (44, 9, z0), "foam")                          # riser block
    ph = tor.at((41.7, 0, z0 + 38.45), R)                                # Galaxy M33: 165.4 x 76.9 x 9.4
    ph.mesh(prism("phone", rrect(165.4, 76.9, 8), -4.7, 4.7), "phone")
    ph.box((-80, -36, -4.8), (80, 36, -4.7), "glass")                    # screen (faces the battery box)
    ph.box((44, 12, 4.7), (76, 34, 5.5), "phone")                        # camera island
    for u in (50, 60, 70):
        ph.cyl((u, 27, 5.5), (u, 27, 6.6), 5, "ring")
        ph.cyl((u, 27, 6.6), (u, 27, 6.8), 4, "glass")
    ph.cyl((50, 17, 5.5), (50, 17, 6.4), 2.6, "glass")
    ph.cyl((60, 17, 5.5), (60, 17, 6.0), 2.2, "flash")
    ph.box((-82.9, -4, -1.6), (-82.6, 4, 1.6), "socket")                 # USB-C
    ph.box((5, 38.4, -1.4), (15, 39.2, 1.4), "phone")                    # power / volume keys
    ph.box((20, 38.4, -1.4), (45, 39.2, 1.4), "phone")
    bx = tor.at((26, -10, z0 + 38), R)                                   # grey printed case (4S 18650 pack)
    bx.mesh(prism("battery", rrect(150, 76, 5), -11, 11), "pla")
    bx.box((75, 10, -4), (75.6, 17, 4), "socket")                        # USB port at the +y end
    top = z0 + 76.9
    tor.box((15, -38, top), (46.4, 4, top + 2), "foam")
    for lo, hi in (((46.4, 5, z0), (48, 35, top)), ((13.4, 5, z0), (15, 35, top)), ((13.4, 5, top), (48, 35, top + 2))):
        tor.box(lo, hi, "velcro")
    for lo, hi in (((46.4, -86, 57), (47.2, 66, 59.5)), ((14.2, -86, 57), (15, 66, 59.5)), ((14.2, -86.8, 57), (47.2, -86, 59.5))):
        tor.box(lo, hi, "tie")                                           # zip tie round the stack
    tor.box((10, -54, 54), (14.2, -46, 62), "tie")
    tor.at((12, -50, 58.25), rot("z", 154)).box((0, -2.4, -0.6), (85, 2.4, 0.6), "tie")
    li = tor.at((30.7, 0, top + 2), rot("z", 180))                       # RPLIDAR A1, motor end toward the back
    li.mesh(prism("lidar_base", circles((0, 0, 35.2), (46, 0, 13)), 0, 12), "lidar")
    li.cyl((0, 0, 12), (0, 0, 21), 34, "lidar2")
    li.cyl((0, 0, 21), (0, 0, 29), 34.4, "window")
    li.cyl((0, 0, 29), (0, 0, 37), 35, "lidar")
    li.cyl((0, 0, 37), (0, 0, 37.3), 31, "lidar2")
    li.cyl((0, 0, 37.3), (0, 0, 37.5), 12.5, "lidar")
    for a in (90, 210, 330):
        li.cyl((24 * math.cos(math.radians(a)), 24 * math.sin(math.radians(a)), 37.3),
               (24 * math.cos(math.radians(a)), 24 * math.sin(math.radians(a)), 37.6), 2.2, "socket")
    li.cyl((46, 0, 12), (46, 0, 17), 7, "lidar2")                        # motor pulley
    li.cyl((46, 0, 17), (46, 0, 18), 2.5, "screw")
    li.cyl((46, 0, -14), (46, 0, 0), 12, "lidar")                        # motor
    li.box((-10, -42, 2), (12, -36, 10), "rubber")                       # USB adapter
    tor.box((22, 65, z0 + 48), (30, 75, z0 + 55), "rubber")               # USB plug in the battery box
    cable(tor, [(26, 75, z0 + 51.5), (26, 92, z0 + 58), (30, 78, top + 8), (30, 48, top + 8), li.p((0, -42, 6))],
          ("wire_k",), 1.8)
    tor.box((37.7, -92, z0 + 35.5), (45.7, -82.7, z0 + 41.5), "rubber")   # OTG lead: phone USB-C -> USC mini-USB
    cable(tor, [(41.7, -92, z0 + 38.5), (40, -104, z0 + 26), (28, -96, z0 + 10), (0, -70, 56), (-25, -45, 60),
                (-45, -25, 50), (-58, 8, 42), (BOARD[0] - 34, 8, BOARD[2] + 3.6)], ("wire_k",), 1.8)
    tor.box((-31, -52, 57), (-19, -38, 63), "rubber")                    # OTG adapter
    cable(tor, [(20, -86, z0 + 13), (22, -98, z0 + 4), (20, -70, 34), (8, -35, 36), (4, -7, BOARD[2] + 11.3)],
          ("wire_r", "wire_k"), 0.9)                                     # pack -> VS/GND
    cable(tor, [(26, -86, z0 + 8), (10, -72, 52), (-20, -40, 56), (-60, -15, 50), (-85, -5, 30), (-95, 0, 14)],
          ("wire_r", "wire_k"), 0.9)                                     # charge lead, DC jack dangling at the back
    tor.cyl((-95, 0, 14), (-95, 0, -4), 4.2, "rubber")
    tor.cyl((-95, 0, -4), (-95, 0, -9), 2.8, "ring")
    return ph


def leg(name, chans, mx, my, ang, tor, rng):
    """Coxa -> femur -> tibia bodies (XML) for one leg; its coxa servo and the leads' torso runs go into tor."""
    s = -1 if name[0] == "R" else 1           # side the femur/knee servo cases sit on (mirrored left/right)
    plug = lambda ch: tor.at(BOARD).p(plug_top(ch))
    ylo, yhi = sorted((s * 1.5, s * 38.5))
    front = mx > 30
    via = [(46, math.copysign(30, my), 40), (12, math.copysign(18, my), 40)] if front else []

    def to_board(start, ch, extra=()):
        loop = (rng.uniform(-48, 0), rng.uniform(-28, 28), rng.uniform(52, 78))
        p = plug(ch)
        cable(tor, [start, *extra, *via, loop, p + (0, 0, 14), p])

    # coxa servo sits in the body, shaft up at the mount point, case pointing inward
    cs = tor.at((mx, my, 30.5), rot("z", ang + 180))
    c_exit = servo(cs)
    u_in = -np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0])
    to_board(c_exit, chans[0], [c_exit + 6 * u_in + (0, 0, 8), c_exit + 10 * u_in + (0, 0, 26)])

    cx = []
    fc = Frame(cx)
    horn(fc.at((0, 0, 30.5)))
    # C-bracket from the coxa horn round the femur servo, bolted to its ears; the servo body hangs out sideways
    fc.at((0, 0, 38)).mesh(prism(f"coxa_bracket_{name[0]}", circles((0, 0, 14), (18, -4 * s, 1), (43, -4 * s, 1),
                                                                 (43, 16 * s, 1), (18, 16 * s, 1)), 0, 2.5), "alu")
    blo, bhi = sorted((s * 3, s * 16))
    fc.box((40.5, blo, -21.5), (43, bhi, 40.5), "alu")
    fc.box((18, blo, -21.5), (43, bhi, -19), "alu")
    horn_screws(fc, (0, 0, 40.5), (0, 0, 1))
    for x in (25, 35):
        screw(fc, (x, s * 10.25, 40.5), (0, 0, 1), 1.8)
        screw(fc, (x, s * 10.25, -21.5), (0, 0, -1), 1.8)
    fs = axes(fc, (C, s * 1.5, 0), (0, 0, -1), (0, -s, 0))
    f_exit = servo(fs)
    cable(fc, [f_exit, f_exit + (-4, s * 3, 10), (10, s * 20, 48.5), (0, 0, 49)])
    fc.box((18, min(ylo, -4 * s), -21.5), (43, max(yhi, -4 * s), 40.5), None, "col")
    to_board((mx, my, 49), chans[1], [(mx, my, 56) + 12 * u_in])
    cable(fc, [(C, s * 44, 0), (C - 8, s * 46, 30), (10, s * 30, 46.5), (0, 0, 46)])  # knee lead passing through
    to_board((mx, my, 46), chans[2], [(mx, my, 52) + 8 * u_in])

    fm = []
    ff = Frame(fm)
    for x in (0, FE):
        horn(axes(ff, (x, s * 1.5, 0), (0, 0, -1), (0, -s, 0)))
        horn_screws(ff, (x, -s * 9, 0), (0, -s, 0))
    ff.at((0, -s * 7.5, 0), rot("x", 90)).mesh(prism("femur", circles((0, 0, 13), (FE, 0, 14)), -1.5, 1.5), "alu")
    cable(ff, [(FE, s * 44, 0), (FE - 12, s * 40, 20), (FE - 22, -s * 11, 12), (22, -s * 11, 12), (12, s * 40, 20),
               (0, s * 44, 0)])
    ff.box((-13, min(-s * 6, -s * 9), -13), (FE + 14, max(-s * 6, -s * 9), 13), None, "col")

    tb = []
    ft = Frame(tb)
    k_exit = servo(axes(ft, (0, s * 1.5, 0), (0, 0, -1), (0, -s, 0)))
    cable(ft, [k_exit, (0, s * 42, 26), (0, s * 44, 10), (0, s * 44, 0)])
    outline, inner, outer = claw()
    ft.at((0, 0, 0), rot("x", 90)).mesh(prism("tibia", outline, -1.5, 1.5), "alu")
    for x, z in ((-8, 31), (8, 31), (-8, -9)):
        screw(ft, (x, -s * 1.5, z), (0, -s, 0), 1.8)
    ft.ell((0.5, 0, -T + 6.5), (6, 4.5, 8), "tape")                       # white tape on the foot tip
    ft.at((2.5, 0.5, -T + 11), rot("y", 35)).ell((0, 0, 0), (4.5, 3.8, 5.5), "tape")
    ft.box((-11, ylo, -11), (11, yhi, 31), None, "col")
    mids = [(np.array(inner[i]) + np.array(outer[i])) / 2 for i in (0, 4, 8, 11)]
    for a, b in zip(mids, mids[1:]):
        ft.cap((a[0], 0, a[1]), (b[0], 0, b[1]), 4, None, "col")
    ft.sph((0, 0, -T + 6), 6, None, "col")

    (rc, rf, rt), q = ranges(name), quat(rot("z", ang))
    inert = lambda *it: inertial(list(it))
    g = MASS
    return f"""
    <body name="{name}_coxa" pos="{num(mx * MM, my * MM, 0)}" quat="{num(*q)}">
      {inert((g['bracket'], (25, s * 15, 15), (50, 40, 60)), (g['servo'], fs.p((-10.25, 0, -18.5)), (20, 37, 40)),
             (g['horn'], (0, 0, 36.5), (25, 25, 3)))}
      <joint name="{name}_coxa" axis="0 0 1" range="{num(*rc)}"/>
      {chr(10).join(cx)}
      <body name="{name}_femur" pos="{num(C * MM, 0, 0)}">
        {inert((g['femur'], (FE / 2, -s * 7.5, 0), (FE + 28, 3, 28)), (g['horn'], (0, -s * 4.5, 0), (25, 3, 25)),
               (g['horn'], (FE, -s * 4.5, 0), (25, 3, 25)))}
        <joint name="{name}_femur" axis="0 -1 0" range="{num(*rf)}"/>
        {chr(10).join(fm)}
        <body name="{name}_tibia" pos="{num(FE * MM, 0, 0)}">
          {inert((g['servo'], (0, s * 20, 10.25), (20, 37, 40)), (g['tibia'], (0.25 * T, 0, -0.35 * T), (0.6 * T, 3, T + 40)),
                 (2, (0, 0, -T + 8), (10, 10, 12)))}
          <joint name="{name}_tibia" axis="0 -1 0" range="{num(*rt)}"/>
          {chr(10).join(tb)}
          <site name="{name}_foot" pos="{num(0, 0, -T * MM)}" size="0.008" group="4" rgba="0 1 0 .4"/>
        </body>
      </body>
    </body>"""


def build():
    MESHES.clear()
    rng = np.random.default_rng(7)
    tg = []
    tor = Frame(tg)
    mounts = [(mx, my) for _, _, mx, my, _ in hx.LEGS]
    plate = prism("body_plate", circles(*[(x, y, 17) for x, y in mounts]), 0, 2.5)
    tor.at((0, 0, -9)).mesh(plate, "alu")
    tor.at((0, 0, 18)).mesh(plate, "alu")
    for x, y in ((68, 0), (-68, 0), (28, 52), (28, -52), (-28, 52), (-28, -52)):
        tor.cyl((x, y, -6.5), (x, y, 18), 3, "screw")                     # standoffs between the plates
        screw(tor, (x, y, 20.5), (0, 0, 1))
        screw(tor, (x, y, -9), (0, 0, -1))
    usc_board(tor)
    ph = stack(tor)
    legs = [leg(name, chans, mx, my, ang, tor, rng) for name, chans, mx, my, ang in hx.LEGS]
    tor.box((-77, -67, -9), (77, 67, 20.5), None, "col")
    tor.box((13.4, -86, STACK_Z), (48, 83, STACK_Z + 79), None, "col")
    tor.cyl((30.7, 0, STACK_Z + 78.9), (30.7, 0, STACK_Z + 116.4), 35, None, "col")

    z0 = STACK_Z
    g = MASS
    items = [(g["plate"], (0, 0, -7.75), (154, 134, 2.5)), (g["plate"], (0, 0, 19.25), (154, 134, 2.5)),
             (g["usc"], np.add(BOARD, (0, 0, 2)), (62, 44, 8)), (g["riser"], (31, 0, 32), (26, 18, 24)),
             (g["phone"], (41.7, 0, z0 + 38.45), (9.4, 165, 77)), (g["battery"], (26, -10, z0 + 38), (22, 150, 76)),
             (g["lidar"], (30.7, 0, z0 + 96), (97, 70, 37)), (g["cables"], (-10, 0, 55), (100, 100, 40)),
             (g["standoffs"], (0, 0, 6), (140, 100, 25))]
    items += [(g["servo"], tor.at((mx, my, 30.5), rot("z", ang + 180)).p((-10.25, 0, -18.5)), (40, 20, 37))
              for _, _, mx, my, ang in hx.LEGS]
    cam = ph.p((70, 27, 7)) * MM
    imu = ph.p((0, 0, 0)) * MM

    mats = "\n    ".join(f'<material name="{k}" rgba="{v[0]}" specular="{v[1]}" shininess="{v[2]}" '
                         f'reflectance="{v[3]}" emission="{v[4]}"/>' for k, v in MATS.items())
    meshes = "\n    ".join(f'<mesh name="{k}" scale="{MM} {MM} {MM}" vertex="{num(*np.ravel(v))}" '
                           f'face="{" ".join(str(i) for i in np.ravel(f))}"/>' for k, (v, f) in MESHES.items())
    joints = [f"{name}_{j}" for name, *_ in hx.LEGS for j in ("coxa", "femur", "tibia")]
    acts = "\n    ".join(f'<position name="{j}" joint="{j}" ctrlrange="{num(*r)}"/>'
                         for j, r in zip(joints, [r for name, *_ in hx.LEGS for r in ranges(name)]))
    feet = "\n    ".join(f'<touch name="{name}_foot" site="{name}_foot"/>' for name, *_ in hx.LEGS)
    tg = "\n      ".join(tg)
    return f"""<!-- Generated by sim/hexapod_sim.py from python/hexapod.py; edit those, not this file. -->
<mujoco model="hexapod_sar">
  <compiler angle="radian" inertiafromgeom="false" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1440" azimuth="140" elevation="-22"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient=".32 .32 .32" diffuse=".45 .45 .45" specular=".1 .1 .1"/>
    <map znear="0.002"/>
  </visual>
  <default>
    <joint armature="0.003" damping="0.05" frictionloss="0.01"/>
    <position kp="20" dampratio="1" forcerange="-1.08 1.08"/>
    <default class="vis"><geom contype="0" conaffinity="0" group="2"/></default>
    <default class="col"><geom contype="1" conaffinity="0" group="3" rgba=".9 .2 .2 .35" friction="0.9 0.02 0.001"/></default>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".85 .87 .9" rgb2=".38 .4 .43" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="flat" mark="random" random="0.012" rgb1=".5 .5 .49" markrgb=".93 .93 .9" width="1024" height="1024"/>
    <material name="floor" texture="floor" texrepeat="4 4" reflectance="0"/>
    {mats}
    {meshes}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 3" dir="0.35 0.45 -1" diffuse=".7 .7 .7" specular=".2 .2 .2" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 3" dir="-0.5 -0.3 -1" diffuse=".25 .25 .27" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" condim="3" friction="0.9 0.02 0.001"/>
    <body name="torso" pos="0 0 {T * MM + 0.001:.6g}">
      <freejoint name="root"/>
      {inertial(items)}
      <camera name="phone_cam" pos="{num(*cam)}" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <site name="imu" pos="{num(*imu)}" size="0.005" group="4"/>
      {tg}
      {"".join(legs)}
    </body>
  </worldbody>
  <actuator>
    {acts}
  </actuator>
  <sensor>
    {feet}
    <accelerometer name="phone_acc" site="imu"/>
    <gyro name="phone_gyro" site="imu"/>
  </sensor>
  <keyframe>
    <key name="stand" qpos="0 0 {T * MM + 0.001:.6g} 1 0 0 0 {' '.join(['0'] * 18)}" ctrl="{' '.join(['0'] * 18)}"/>
  </keyframe>
</mujoco>
"""


# ---- driving it like the USC-32 ----
def angles(pulses):
    """USC pulses {channel: us} -> 18 joint angles (rad) in actuator order."""
    return np.radians([hx.DIR[n][j] * (pulses[ch] - 1500 - hx.TRIM[n][j]) / hx.US_PER_DEG
                       for n, chans, *_ in hx.LEGS for j, ch in enumerate(chans)])


def gait(t, vx=0.0, vy=0.0, turn=0.0, settle=1.0):
    """Joint targets at sim time t: stand, then one pose() frame every FRAME_MS, blended linearly like the USC's T."""
    stand = angles(hx.pose(hx.STAND))
    if t < settle or not (vx or vy or turn):
        return stand
    f = (t - settle) * 1000 / hx.FRAME_MS
    k = int(f)
    prev = angles(hx.pose((k - 1) % hx.FRAMES, vx, vy, turn)) if k else stand
    return prev + (f - k) * (angles(hx.pose(k % hx.FRAMES, vx, vy, turn)) - prev)


def load():
    xml = build()
    with open(XML, "w") as fh:
        fh.write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d


def view(a):
    import mujoco.viewer
    m, d = load()

    def ctrl(m, d):
        d.ctrl[:] = gait(d.time, a.vx, a.vy, a.turn)

    mujoco.set_mjcb_control(ctrl)
    mujoco.viewer.launch(m, d)


def render(a):
    from PIL import Image
    m, d = load()
    out = os.path.join(HERE, "renders")
    os.makedirs(out, exist_ok=True)
    r = mujoco.Renderer(m, 1080, 1440)
    cam = mujoco.MjvCamera()
    torso = m.body("torso").id
    for _ in range(300):                                  # settle on the floor
        d.ctrl[:] = gait(d.time)
        mujoco.mj_step(m, d)
    for name, az, el, dist, dz in (("hero", 140, -24, 0.62, 0.03), ("rear", -35, -30, 0.6, 0.03),
                                   ("side", 90, -6, 0.62, 0.0), ("top", 90, -89.9, 0.7, 0.0),
                                   ("closeup_leg", -55, -14, 0.3, None)):
        cam.lookat[:] = d.xpos[torso] + (0, 0, dz) if dz is not None else d.xpos[m.body("LM_femur").id] + (0, .045, -.02)
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        r.update_scene(d, cam)
        Image.fromarray(r.render()).save(os.path.join(out, f"{name}.png"))
    r.update_scene(d, "phone_cam")
    Image.fromarray(r.render()).save(os.path.join(out, "phone_cam.png"))
    r.close()
    r = mujoco.Renderer(m, 360, 480)                      # walking loop
    frames, fps = [], 15
    while d.time < 1.0 + 3 * hx.FRAMES * hx.FRAME_MS / 1000:
        d.ctrl[:] = gait(d.time - 0.6, vx=1)
        mujoco.mj_step(m, d)
        if len(frames) < d.time * fps:
            cam.lookat[:] = d.xpos[torso] + (0, 0, 0.02)
            cam.azimuth, cam.elevation, cam.distance = 125, -20, 0.62
            r.update_scene(d, cam)
            frames.append(Image.fromarray(r.render()))
    frames[0].save(os.path.join(out, "walk.gif"), save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
    print("renders in", out)


def selftest():
    m, d = load()
    assert m.nu == 18 and m.njnt == 19, (m.nu, m.njnt)
    assert np.allclose(angles(hx.pose(hx.STAND)), 0, atol=1e-3)            # stand pose == joint zero
    for i, (name, _, mx, my, ang) in enumerate(hx.LEGS):                  # FK of the model == hexapod.ik()
        for tgt in [(100, 20, -70), (80, -30, -60), (120, 0, -40)]:
            d.qpos[:] = 0
            d.qpos[3] = 1
            d.qpos[7 + 3 * i:10 + 3 * i] = np.radians(hx.ik(*tgt))
            mujoco.mj_kinematics(m, d)
            want = (np.array([mx, my, 0]) + rot("z", ang) @ tgt) * MM
            assert np.allclose(d.site(f"{name}_foot").xpos, want, atol=1e-6), (name, tgt, d.site(f"{name}_foot").xpos, want)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    up = lambda: d.xmat[m.body("torso").id].reshape(3, 3)[2, 2]
    while d.time < 1.5:
        d.ctrl[:] = gait(d.time)
        mujoco.mj_step(m, d)
    assert up() > 0.95 and d.qpos[2] > 0.5 * T * MM, (up(), d.qpos[2])      # stands without collapsing
    x0 = d.qpos[0]
    while d.time < 1.5 + 3 * hx.FRAMES * hx.FRAME_MS / 1000:
        d.ctrl[:] = gait(d.time - 0.5, vx=1)
        mujoco.mj_step(m, d)
    assert up() > 0.9 and d.qpos[0] > x0, (up(), d.qpos[0] - x0)           # walks forward, stays upright
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "view", "render", "selftest"])
    ap.add_argument("--vx", type=float, default=0)
    ap.add_argument("--vy", type=float, default=0)
    ap.add_argument("--turn", type=float, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        load()
        print("wrote", XML)
    else:
        {"view": view, "render": render, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
