"""Deploy rig: one control empty drives the four legs, their telescoping pushers and the
four grid fins.

F9_Rig["leg_deploy"]     0 = stowed, 1 = landing legs fully deployed (dims.LEG_DEPLOY_DEG, 120 deg)
F9_Rig["gridfin_deploy"] 0 = stowed, 1 = grid fins deployed (90 deg)
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import dims as D
from . import util as U

PUSHER_LEN = 3.8
ANCHOR_Z = 9.95        # pusher attachment on the stage (stowed leg covers it)
MOUNT_Z = 5.75         # pusher attachment on the leg (stowed position)
PUSHER_R = D.LEG_R_IN + D.LEG_DEPTH / 2


def _prop(ob, name, desc):
    ob[name] = 0.0
    ui = ob.id_properties_ui(name)
    ui.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, description=desc)


def _drive(target_ob, rig, prop, scale):
    fc = target_ob.driver_add("rotation_euler", 1)
    drv = fc.driver
    drv.type = "SCRIPTED"
    var = drv.variables.new()
    var.name = "d"
    var.type = "SINGLE_PROP"
    var.targets[0].id_type = "OBJECT"
    var.targets[0].id = rig
    var.targets[0].data_path = f'["{prop}"]'
    drv.expression = f"d * {scale:.6f}"
    # remove the default modifier so the driver value is used as-is
    for m in list(fc.modifiers):
        fc.modifiers.remove(m)


def _tube_mesh(name, r, z0, z1, mat, segs=20):
    bm = bmesh.new()
    U.add_cyl(bm, r, r, z1 - z0, segs, Matrix.Translation((0.0, 0.0, (z0 + z1) / 2)), mat=0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(40.0))
    return me


def build(colls, M, root, s1, legs, fins):
    c_legs = colls["Legs"]
    rig = U.empty("F9_Rig", colls.get("Master", colls["Stage1"]), parent=root, location=(0.0, 0.0, 72.0), size=3.0, kind="ARROWS")
    _prop(rig, "leg_deploy", "Landing legs: 0 stowed, 1 deployed")
    _prop(rig, "gridfin_deploy", "Grid fins: 0 stowed, 1 deployed")

    for ob in legs:
        ob.rotation_mode = "XYZ"
        _drive(ob, rig, "leg_deploy", math.radians(D.LEG_DEPLOY_DEG))
    for ob in fins:
        ob.rotation_mode = "XYZ"
        _drive(ob, rig, "gridfin_deploy", -math.pi / 2)

    # telescoping pushers: outer tube on the stage, inner rod on the leg, middle stage
    # halfway between; Damped Track keeps all three on the anchor-mount line
    for i, (leg, a) in enumerate(zip(legs, D.AZ_LEGS)):
        th = math.radians(a)
        p_anchor = Vector((PUSHER_R * math.cos(th), PUSHER_R * math.sin(th), ANCHOR_Z))
        p_mount = Vector((PUSHER_R * math.cos(th), PUSHER_R * math.sin(th), MOUNT_Z))
        anchor = U.empty(f"Leg_{i + 1}_PusherAnchor", c_legs, parent=s1, location=p_anchor, size=0.2)
        mount = U.empty(f"Leg_{i + 1}_PusherMount", c_legs, size=0.2)
        mount.parent = leg
        mount.matrix_parent_inverse = leg.matrix_world.inverted()
        mount.location = p_mount

        # stage-side bracket for the anchor
        bm = bmesh.new()
        m = U.frame_at(th, (D.R + PUSHER_R) / 2, ANCHOR_Z)
        U.add_box(bm, (PUSHER_R - D.R + 0.06, 0.22, 0.30), m, mat=0)
        U.add_cyl(bm, 0.05, 0.05, 0.24, 16, U.frame_at(th, PUSHER_R, ANCHOR_Z) @
                  Matrix.Rotation(math.pi / 2, 4, "X"), mat=1)
        U.finish(f"Leg_{i + 1}_PusherBracket", bm, c_legs, [M["black"], M["steel"]], parent=s1, sharp=40.0)

        outer = bpy.data.objects.new(f"Leg_{i + 1}_Pusher_Outer",
                                     _tube_mesh(f"Leg_{i + 1}_Pusher_Outer", 0.085, 0.0, PUSHER_LEN, M["carbon"]))
        c_legs.objects.link(outer)
        outer.parent = s1
        outer.location = p_anchor
        con = outer.constraints.new("DAMPED_TRACK")
        con.target = mount
        con.track_axis = "TRACK_Z"

        inner = bpy.data.objects.new(f"Leg_{i + 1}_Pusher_Inner",
                                     _tube_mesh(f"Leg_{i + 1}_Pusher_Inner", 0.052, 0.0, PUSHER_LEN, M["steel"]))
        c_legs.objects.link(inner)
        con = inner.constraints.new("COPY_LOCATION")
        con.target = mount
        con = inner.constraints.new("DAMPED_TRACK")
        con.target = anchor
        con.track_axis = "TRACK_Z"
        inner.parent = s1

        mid = bpy.data.objects.new(f"Leg_{i + 1}_Pusher_Mid",
                                   _tube_mesh(f"Leg_{i + 1}_Pusher_Mid", 0.068, -PUSHER_LEN / 2, PUSHER_LEN / 2,
                                              M["alu"]))
        c_legs.objects.link(mid)
        mid.parent = s1
        con = mid.constraints.new("COPY_LOCATION")
        con.target = anchor
        con = mid.constraints.new("COPY_LOCATION")
        con.target = mount
        con.influence = 0.5
        con = mid.constraints.new("DAMPED_TRACK")
        con.target = mount
        con.track_axis = "TRACK_Z"
    return rig
