Hexapod SAR robot - 3D model (MuJoCo)

Just look at it:   sim/renders/  (PNG pictures + walk.gif)

Open the 3D model:
  pip install mujoco pillow
  python3 sim/hexapod_sim.py view            (standing; drag to rotate)
  python3 sim/hexapod_sim.py view --vx 1     (walking with the robot's real gait)

sim/hexapod.xml is the model file on its own - it also opens in any MuJoCo viewer.
Leg lengths, mounts, trims come from python/hexapod.py; edit that and run
  python3 sim/hexapod_sim.py build
to regenerate it. Only coxa 30 mm and femur 85 mm are measured; other sizes are
estimated from photos, so sim speeds/clearances are not real measurements.
