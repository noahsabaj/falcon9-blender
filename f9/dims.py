"""Falcon 9 Block 5 (standard fairing) master dimensions.

Units are metres, Z up, vehicle axis on the world Z axis, z = 0 at the plane of the
Merlin 1D nozzle exits.  Azimuth (theta) is measured from +X toward +Y; looking at the
vehicle from outside, increasing theta runs to the viewer's right.

Sources: SpaceX Falcon User's Guide (2025) for the 70 m stack, 3.66 m core, 5.2 m x
13.2 m fairing and fairing outline; AXM 1:100 Block 5 (CRS-18) paper model sheets for
livery placement, leg planform and interstage details; launch photographs for the
interstage / second-stage split.
"""
import math

R = 1.83                     # core radius (3.66 m)
RF = 2.60                    # fairing radius (5.2 m)

# --- vertical stations -----------------------------------------------------------
Z_NOZZLE_EXIT = 0.0
Z_HEATSHIELD = 1.15          # base heat shield plane (engines protrude below it)
Z_SKIRT_TOP = 2.25           # black aft skirt / octaweb fairing -> white tank
Z_S1_TOP = 41.58             # first-stage forward joint = interstage bottom
Z_IS_TOP = 48.13             # interstage top = exposed second stage bottom
Z_FAIRING_BASE = 56.85       # fairing separation plane / PAF ring
FAIRING_LEN = 13.15
Z_TIP = Z_FAIRING_BASE + FAIRING_LEN   # 70.00 m overall

# --- azimuths (degrees) ------------------------------------------------------------
AZ_RACEWAY = 0.0
AZ_LEGS = (45.0, 135.0, 225.0, 315.0)
AZ_GRIDFINS = AZ_LEGS
AZ_LOGOS = (90.0, 270.0)
AZ_LATCHES = (0.0, 120.0, 240.0)
AZ_GN2_PODS = (90.0, 270.0)
AZ_FAIRING_SEAM = (90.0, 270.0)

# --- Merlin 1D (sea level) -------------------------------------------------------
M1D_EXIT_R = 0.46            # nozzle exit inner radius (0.92 m)
M1D_THROAT_R = 0.115
M1D_Z_THROAT = 1.10          # throat sits just above the heat shield; bell ~1.1 m
M1D_RING_R = 1.26            # outer-ring engine pitch radius
M1D_RING_AZ = tuple(22.5 + 45.0 * k for k in range(8))

# --- MVac -----------------------------------------------------------------------
MVAC_EXIT_R = 1.65           # 3.3 m niobium extension exit
MVAC_Z_EXIT = 43.00
MVAC_Z_EXT_TOP = 46.60       # extension / regen-nozzle joint
MVAC_EXT_TOP_R = 0.55
MVAC_Z_THROAT = 47.35

# --- landing legs (stowed) ------------------------------------------------------
LEG_Z_HINGE = 1.75           # bottom of the two prongs (hinge line)
LEG_Z_FACE_TOP = 10.15       # top of the leg body (footpad cap rises ~0.9 m above)
LEG_R_IN = 1.845             # inner (back) surface radius
LEG_DEPTH = 0.24             # radial depth of the leg box
LEG_DEPLOY_DEG = 120.0       # deployed rotation about the hinge line (~18.8 m span)

# --- grid fins (stowed, hanging) ------------------------------------------------
GF_Z_HINGE = 42.05
GF_WIDTH = 1.25              # tangential
GF_ROOT = 0.35               # gabled root height
GF_LATTICE_LEN = 1.25        # lattice frame length below the root
GF_DEPTH = 0.25              # lattice depth (radial when stowed)
GF_R_IN = 1.90               # inner face radius when stowed
GF_PITCH = 0.15              # perpendicular blade spacing (~0.21 m diamond diagonal)
GF_BLADE = 0.012
GF_FRAME = 0.045

# --- fairing outline (z above fairing base, radius) ------------------------------
FAIRING_BASE_RING = ((R, 0.0), (R, 0.17), (2.07, 0.21))
FAIRING_BOATTAIL_TOP = 1.22
FAIRING_CYL_TOP = 8.36
# Ogive / blunt nose (smoothed from User's Guide Fig. 4-3)
FAIRING_NOSE = (
    (8.36, 2.600), (8.72, 2.578), (9.02, 2.548), (9.46, 2.482), (9.76, 2.410),
    (10.06, 2.315), (10.36, 2.208), (10.65, 2.090), (10.95, 1.968), (11.25, 1.833),
    (11.55, 1.676), (11.85, 1.495), (12.15, 1.285), (12.44, 1.050), (12.74, 0.790),
    (12.89, 0.635), (13.00, 0.495), (13.07, 0.375), (13.115, 0.255), (13.140, 0.130),
    (13.150, 0.0),
)
FAIRING_WALL = 0.035


def az(deg):
    return math.radians(deg)
