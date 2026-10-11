"""World, lighting, ground, cameras and render defaults."""
import math

import bmesh
import bpy
from mathutils import Vector

from . import materials as MT
from . import util as U

SUN_AZ = 25.0          # degrees, same convention as the vehicle azimuths
SUN_EL = 32.0
GROUND_Z = -3.32       # deployed footpads touch down here

# name: (azimuth deg, camera z, horizontal distance, target z, lens mm, (res x, res y), deployed, ground)
CAMERAS = {
    "Cam_Full": (62.0, 30.0, 118.0, 35.0, 50.0, (1100, 2000), False, False),
    "Cam_Base": (68.0, 2.2, 13.5, 3.6, 30.0, (1600, 1000), False, False),
    "Cam_Engines": (30.0, -6.2, 6.0, 0.6, 24.0, (1400, 1000), False, False),
    "Cam_Interstage": (55.0, 42.4, 13.0, 43.9, 30.0, (1600, 1000), False, False),
    "Cam_GridFin": (38.0, 41.0, 5.2, 41.5, 30.0, (1400, 1000), False, False),
    "Cam_Cutaway": (45.0, 45.5, 11.0, 45.2, 35.0, (1200, 1400), False, False),
    "Cam_Livery": (80.0, 30.0, 46.0, 30.5, 50.0, (1000, 1800), False, False),
    "Cam_Fairing": (68.0, 58.0, 27.0, 61.0, 50.0, (1000, 1600), False, False),
    "Cam_Landed": (35.0, 2.5, 44.0, 8.0, 35.0, (1600, 1000), True, True),
}


def _world(scn):
    w = bpy.data.worlds.new("F9_Sky")
    scn.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (400, 0)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (150, 0)
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.location = (-150, 0)
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "HOSEK_WILKIE"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(SUN_EL)
    sky.sun_rotation = math.radians(SUN_AZ - 90.0)
    sky.altitude = 10.0
    sky.air_density = 1.0
    sky.aerosol_density = 1.2
    bg.inputs["Strength"].default_value = 0.30
    # below the horizon: a soft ground tone instead of the black lower hemisphere
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-650, -250)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-450, -250)
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-250, -250)
    mr.inputs["From Min"].default_value = -0.04
    mr.inputs["From Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.location = (0, 0)
    sock_a = [s for s in mix.inputs if s.name == "A" and s.type == "RGBA"][0]
    sock_b = [s for s in mix.inputs if s.name == "B" and s.type == "RGBA"][0]
    sock_f = [s for s in mix.inputs if s.name == "Factor" and s.type == "VALUE"][0]
    out_c = [s for s in mix.outputs if s.type == "RGBA"][0]
    sock_a.default_value = (0.42, 0.40, 0.37, 1.0)
    nt.links.new(mr.outputs["Result"], sock_f)
    nt.links.new(sky.outputs["Color"], sock_b)
    nt.links.new(out_c, bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return w


def _sun(coll):
    ld = bpy.data.lights.new("F9_Sun", "SUN")
    ld.energy = 4.0
    ld.angle = math.radians(0.53)
    ld.color = (1.0, 0.97, 0.92)
    ob = bpy.data.objects.new("F9_Sun", ld)
    coll.objects.link(ob)
    el, az = math.radians(SUN_EL), math.radians(SUN_AZ)
    d = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    ob.location = d * 120.0
    return ob


def _ground(coll):
    bm = bmesh.new()
    s = 1500.0
    vs = [bm.verts.new((x, y, GROUND_Z)) for x, y in ((-s, -s), (s, -s), (s, s), (-s, s))]
    bm.faces.new(vs)
    mat = MT.principled("ground_concrete", MT.srgb("#8C8A85"), rough=0.9, rough_var=0.05, noise_scale=0.6,
                        bump=0.15)
    ob = U.finish("F9_Ground", bm, coll, [mat], smooth=False)
    return ob


def _camera(coll, name, az, cz, dist, tz, lens, res, deployed, ground):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start = 0.2
    cd.clip_end = 3000.0
    ob = bpy.data.objects.new(name, cd)
    coll.objects.link(ob)
    a = math.radians(az)
    pos = Vector((dist * math.cos(a), dist * math.sin(a), cz))
    tgt = Vector((0.0, 0.0, tz))
    ob.location = pos
    ob.rotation_euler = (tgt - pos).to_track_quat("-Z", "Y").to_euler()
    ob["res_x"], ob["res_y"] = res
    ob["deployed"] = deployed
    ob["ground"] = ground
    return ob


def build(coll):
    scn = bpy.context.scene
    us = scn.unit_settings
    us.system = "METRIC"
    us.length_unit = "METERS"
    us.scale_length = 1.0
    _world(scn)
    _sun(coll)
    ground = _ground(coll)
    ground.hide_render = True
    cams = {name: _camera(coll, name, *spec) for name, spec in CAMERAS.items()}
    scn.camera = cams["Cam_Full"]
    # cutaway: render.py removes a 126 deg wedge facing the camera (between the grid fins
    # at 315 / 135 deg) so the MVac, domes and thrust cone show in section
    cut = cams["Cam_Cutaway"]
    cut["cut"] = "45,63"
    cut["cut_keep"] = "MVac_,Interstage_CentrePusher"
    cut["cut_z"] = "36,54"

    r = scn.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y = CAMERAS["Cam_Full"][5]
    r.resolution_percentage = 100
    scn.cycles.samples = 64
    scn.cycles.use_denoising = True
    scn.cycles.max_bounces = 6
    scn.view_settings.view_transform = "AgX"
    scn.frame_start, scn.frame_end = 1, 1

    # viewport framing for whoever opens the file: 70 m vehicle needs a long clip range
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start = 0.05
                    space.clip_end = 3000.0
    return cams, ground
