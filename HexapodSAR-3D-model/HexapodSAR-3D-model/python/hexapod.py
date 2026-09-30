#!/usr/bin/env python3
"""18-servo hexapod gait, two legs at a time for a USC-32 board (SSC-32 style serial: #<ch>P<us>...T<ms>).

  python3 hexapod.py selftest
  python3 hexapod.py center --port /dev/tty.usbserial-XXXX   # all servos 1500us: fit the horns in this pose
  python3 hexapod.py stand  --port /dev/tty.usbserial-XXXX
  python3 hexapod.py walk   --port /dev/tty.usbserial-XXXX --vx 1 --secs 5   # vx, vy, turn in -1..1
  add --dry-run to print commands instead of sending. First runs: robot on a box, legs in the air.
"""
import argparse, math, time

# ---- calibration: measure your robot and fill these in ----
BAUD = 9600                              # USC-32 manual: 9600 8N1 only
COXA, FEMUR, TIBIA = 30.0, 85.0, 90.0    # mm: coxa axis->femur axis, femur axis->tibia axis, tibia axis->foot tip
REACH, HEIGHT = COXA + FEMUR, TIBIA      # stand pose; these defaults == all servos at 1500us
US_PER_DEG = 2000 / 180                  # ponytail: nominal MG996R, measure yours (command 90 deg, check with a protractor)
# name, (coxa, femur, tibia) board channels, mount x, y (mm; x forward, y left), mount angle (deg, pointing outward)
LEGS = [                                 # ponytail: channel map is a guess, fill it in from your wiring; S21+ only give a 3.3 V signal
    ("RF", (1, 2, 3),    60, -40,  -45),
    ("RM", (4, 5, 6),     0, -50,  -90),
    ("RR", (7, 8, 9),   -60, -40, -135),
    ("LR", (10, 11, 12), -60, 40,  135),
    ("LM", (13, 14, 15),  0,  50,   90),
    ("LF", (16, 17, 18), 60,  40,   45),
]
DIR = {name: (1, -1, -1) if name[0] == "L" else (1, 1, 1) for name, *_ in LEGS}   # left femurs + tibias mirrored
TRIM = {                                # us; stand pose set on the robot 30 Sep 14:00 (stand = 1500 + trim)
    "RF": (-140, 170, 720), "RM": (380, 160, 470), "RR": (-210, 120, 490),
    "LR": (430, -210, -460), "LM": (-40, -210, -570), "LF": (190, -340, -700)}
# Legs that step together, in order: two at a time (diagonal pairs, then the middles), 4 feet always down.
# ponytail: one at a time (slower, 5 feet down) is [["RR"], ["RM"], ["RF"], ["LR"], ["LM"], ["LF"]]
GROUPS = [["RF", "LR"], ["RM", "LM"], ["RR", "LF"]]
RAISE = 15.0                            # mm the body rises while pairs step (grounded feet push down)
STRIDE, LIFT = 80.0, 80.0              # mm; stride sets the coxa swing (80 ≈ 35°) and speed, LIFT how high a foot clears
TURN_DEG_PER_MM = 0.3                   # body rotation per step at turn = 1, per mm of stride (60 mm -> 18°)
FRAME_MS = 200                          # 9600 baud: an 18-servo frame (~154 B) takes ~160 ms
SWING = 2                               # frames per pair in the air: up + halfway forward, then down at the front
FRAMES = SWING * len(GROUPS) + 2        # per cycle: each pair steps; then 2 frames move the body
STAND = FRAMES - 1                      # the last frame of a cycle is the stand pose


def ik(x, y, z):
    """Foot target in leg frame (mm, x outward, z up) -> (coxa, femur, tibia) degrees from the all-1500us pose."""
    coxa = math.atan2(y, x)
    r = math.hypot(x, y) - COXA
    d = max(abs(FEMUR - TIBIA) + 1e-6, min(math.hypot(r, z), FEMUR + TIBIA - 1e-6))  # clamp to reachable
    femur = math.atan2(z, r) + math.acos((FEMUR**2 + d**2 - TIBIA**2) / (2 * FEMUR * d))
    knee = math.acos((FEMUR**2 + TIBIA**2 - d**2) / (2 * FEMUR * TIBIA))
    return math.degrees(coxa), math.degrees(femur), math.degrees(knee) - 90


def foot(frame, name):
    """(s, h): the foot sits s * (travel per step) ahead of neutral and h * LIFT up."""
    if frame >= SWING * len(GROUPS):      # all feet down and slide back together: the body moves forward
        return (0.5 if frame == SWING * len(GROUPS) else 0.0), 0.0
    k = frame - SWING * next(i for i, g in enumerate(GROUPS) if name in g)
    if k < 0:
        return 0.0, 0.0                   # not stepped yet this cycle
    return [(0.5, 1.0), (1.0, 0.0)][min(k, 1)]   # top of the arc halfway forward, then land at the front


def pose(frame, vx=0.0, vy=0.0, turn=0.0):
    """Frame 0..FRAMES-1 of the walk cycle -> {channel: us}. pose(STAND) is the stand pose.
    The foot follows an arc in the air (side view) and a straight line (top view); IK turns that into all three
    joints, as offsets from the calibrated stand pose. LIFT sets how high a foot clears."""
    out = {}
    raise_ = RAISE if frame < SWING * len(GROUPS) else 0.0   # body up while pairs step, down for the slide
    for name, chans, mx, my, ang in LEGS:
        a = math.radians(ang)
        fx, fy = mx + REACH * math.cos(a), my + REACH * math.sin(a)   # neutral foot, body frame
        s, z = foot(frame, name)
        g = s * math.radians(TURN_DEG_PER_MM * STRIDE) * turn                            # turn: rotate the foot around the body centre
        dx = fx * math.cos(g) - fy * math.sin(g) + s * vx * STRIDE - mx  # then shift it along the walk direction
        dy = fx * math.sin(g) + fy * math.cos(g) + s * vy * STRIDE - my
        lx, ly = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
        th = math.atan2(ly, lx)   # coxa-only horizontal motion: swing around the coxa pivot at the stand distance
        degs = ik(REACH * math.cos(th), REACH * math.sin(th), -HEIGHT - raise_ + z * (LIFT + raise_))   # z = share of the lift
        for ch, deg, sign, trim in zip(chans, degs, DIR[name], TRIM[name]):
            out[ch] = max(500, min(2500, round(1500 + sign * deg * US_PER_DEG + trim)))
    return out


def cmd(pulses, ms):
    return ("".join(f"#{ch}P{us}" for ch, us in sorted(pulses.items())) + f"T{ms}\r\n").encode()


def selftest():
    def fk(c, f, t):
        c, f, k = map(math.radians, (c, f, t + 90))
        r = COXA + FEMUR * math.cos(f) + TIBIA * math.cos(f - math.pi + k)
        return r * math.cos(c), r * math.sin(c), FEMUR * math.sin(f) + TIBIA * math.sin(f - math.pi + k)
    assert all(abs(v) < 1e-9 for v in ik(COXA + FEMUR, 0, -TIBIA))
    for target in [(100, 20, -70), (80, -30, -60), (120, 0, -40)]:
        assert all(abs(a - b) < 1e-6 for a, b in zip(fk(*ik(*target)), target)), target
    trims = {ch: t for name, chans, *_ in LEGS for ch, t in zip(chans, TRIM[name])}
    assert all(us == 1500 + trims[ch] for ch, us in pose(STAND, 1, 0, 0.5).items())   # stand == 1500 + trim
    for f in range(FRAMES):
        p = pose(f, 1, 0, 0.5)
        assert len(p) == 18
        lifted = [n for n, *_ in LEGS if foot(f, n)[1] > 0]
        assert len(lifted) <= max(map(len, GROUPS)), (f, lifted)   # never more than one group in the air
    st = pose(STAND)
    for f in range(FRAMES):                                 # a lifted leg raises its femur (+us right, -us left)
        p = pose(f, 1, 0, 0.5)
        for name, (c, fe, t), *_ in LEGS:
            if foot(f, name)[1] > 0:
                assert (p[fe] - st[fe]) * DIR[name][1] > 0, (f, name, st[fe], p[fe])
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "center", "stand", "walk"])
    ap.add_argument("--port", default="/dev/tty.usbserial-XXXX")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vx", type=float, default=0)
    ap.add_argument("--vy", type=float, default=0)
    ap.add_argument("--turn", type=float, default=0)
    ap.add_argument("--secs", type=float, default=5)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.dry_run:
        send = lambda b: print(b.decode().strip())
    else:
        import serial  # pip install pyserial
        send = serial.Serial(a.port, BAUD).write
    stand = cmd(pose(STAND), 1000)
    if a.cmd == "center":
        send(cmd({ch: 1500 for _, chans, *_ in LEGS for ch in chans}, 1000))
    elif a.cmd == "stand":
        send(stand)
    else:
        if len(stand) * 10 / BAUD * 1000 > FRAME_MS:
            print(f"warning: {BAUD} baud can't keep up with {FRAME_MS} ms frames; raise BAUD or FRAME_MS")
        t0, frame = time.time(), 0
        try:
            while time.time() - t0 < a.secs:
                send(cmd(pose(frame % FRAMES, a.vx, a.vy, a.turn), FRAME_MS))
                frame += 1
                time.sleep(FRAME_MS / 1000)
        finally:
            send(stand)


if __name__ == "__main__":
    main()
