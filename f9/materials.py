"""Procedural materials (Principled BSDF, no image textures)."""
import bpy

_CACHE = {}


def srgb(hexstr):
    h = hexstr.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (out[0], out[1], out[2], 1.0)


def _base(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bsdf


def _noise(nt, scale, detail=6.0, rough=0.55, coord="Object", loc=(-700, 0)):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (loc[0] - 250, loc[1])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.location = loc
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    nz.inputs["Roughness"].default_value = rough
    nt.links.new(tc.outputs[coord], nz.inputs["Vector"])
    return nz


def _maprange(nt, src, lo, hi, loc=(-350, 0)):
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = loc
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    mr.inputs["To Min"].default_value = lo
    mr.inputs["To Max"].default_value = hi
    nt.links.new(src, mr.inputs["Value"])
    return mr


def principled(key, color, metallic=0.0, rough=0.5, rough_var=0.0, noise_scale=8.0,
               coat=0.0, coat_rough=0.1, bump=0.0, spec=0.5, diffuse_display=None, back=None):
    """back: optional (color, metallic, roughness) shown on back faces (inside of
    single-surface skins, e.g. bare tank walls in a cutaway)."""
    if key in _CACHE:
        return _CACHE[key]
    m, nt, b = _base("F9_" + key)
    if back is not None:
        out = nt.nodes["Material Output"]
        out.location = (800, 0)
        bb = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bb.location = (300, -700)
        bb.inputs["Base Color"].default_value = back[0]
        bb.inputs["Metallic"].default_value = back[1]
        bb.inputs["Roughness"].default_value = back[2]
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        geo.location = (300, 300)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.location = (600, 0)
        nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
        nt.links.new(b.outputs["BSDF"], mix.inputs[1])
        nt.links.new(bb.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    b.inputs["Base Color"].default_value = color
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    if coat > 0:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = coat_rough
    if rough_var > 0 or bump > 0:
        nz = _noise(nt, noise_scale)
        if rough_var > 0:
            mr = _maprange(nt, nz.outputs["Fac"], max(0.0, rough - rough_var), min(1.0, rough + rough_var))
            nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
        if bump > 0:
            bp = nt.nodes.new("ShaderNodeBump")
            bp.location = (0, -300)
            bp.inputs["Strength"].default_value = bump
            bp.inputs["Distance"].default_value = 0.002
            nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
            nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
    m.diffuse_color = diffuse_display or color
    m.metallic = metallic
    m.roughness = rough
    _CACHE[key] = m
    return m


def nozzle_metal(key="nozzle_m1d", z_exit=0.0, z_top=1.6):
    """Merlin regen nozzle: satin Inconel with heat tint toward the exit (object Z)."""
    if key in _CACHE:
        return _CACHE[key]
    m, nt, b = _base("F9_" + key)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-900, 0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-700, 0)
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-500, 0)
    mr.inputs["From Min"].default_value = z_exit
    mr.inputs["From Max"].default_value = z_top
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-300, 0)
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.16, 0.15, 0.15, 1.0)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.46, 0.44, 0.42, 1.0)
    e = cr.elements.new(0.08)
    e.color = (0.30, 0.27, 0.25, 1.0)
    e = cr.elements.new(0.55)
    e.color = (0.40, 0.38, 0.37, 1.0)
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = 1.0
    nz = _noise(nt, 30.0, loc=(-700, -300))
    rr = _maprange(nt, nz.outputs["Fac"], 0.32, 0.5, loc=(-350, -300))
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    m.diffuse_color = (0.35, 0.34, 0.33, 1.0)
    m.metallic = 1.0
    _CACHE[key] = m
    return m


def library():
    """Return the dict of all vehicle materials (created once)."""
    L = {}
    L["white"] = principled("paint_white", srgb("#ECEEEF"), rough=0.32, rough_var=0.05,
                            noise_scale=3.0, coat=0.2, coat_rough=0.25,
                            back=(srgb("#A7A9AC"), 1.0, 0.45))
    L["black"] = principled("tps_black", srgb("#111111"), rough=0.72, rough_var=0.06,
                            noise_scale=1.5, bump=0.015, spec=0.35)
    L["carbon"] = principled("leg_black", srgb("#121213"), rough=0.5, rough_var=0.07,
                             noise_scale=10.0, coat=0.15, coat_rough=0.4)
    L["raceway"] = principled("raceway_black", srgb("#161616"), rough=0.55, rough_var=0.05)
    L["titanium"] = principled("titanium", srgb("#5E5B57"), metallic=1.0, rough=0.55,
                               rough_var=0.12, noise_scale=14.0, bump=0.04)
    L["nozzle"] = nozzle_metal()
    L["nozzle_in"] = principled("nozzle_inner", srgb("#1A1716"), metallic=0.6, rough=0.7)
    L["niobium"] = principled("niobium_c103", srgb("#4A4A4C"), metallic=0.85, rough=0.5,
                              rough_var=0.1, noise_scale=20.0)
    L["regen"] = principled("mvac_regen", srgb("#6E6A66"), metallic=1.0, rough=0.38)
    L["engine_dark"] = principled("engine_dark", srgb("#2A2826"), metallic=0.9, rough=0.45)
    L["steel"] = principled("steel", srgb("#9A9A9C"), metallic=1.0, rough=0.3)
    L["alu"] = principled("aluminium", srgb("#B4B6B8"), metallic=1.0, rough=0.35)
    L["boot"] = principled("boot_fabric", srgb("#B9B4AA"), rough=0.92, bump=0.3, noise_scale=40.0)
    L["heatshield"] = principled("heatshield", srgb("#2E2D2B"), rough=0.85, rough_var=0.05, bump=0.1)
    L["pad"] = principled("rest_pad", srgb("#1E1A15"), rough=0.85)
    L["grey"] = principled("panel_grey", srgb("#7D7F81"), rough=0.5)
    L["light_grey"] = principled("light_grey", srgb("#C4C6C8"), rough=0.45)
    L["nosecap"] = principled("fairing_nosecap", srgb("#3A3B3D"), rough=0.5)
    L["vent"] = principled("vent_black", srgb("#050505"), rough=0.9)
    L["seam"] = principled("seam_dark", srgb("#55585B"), rough=0.6)
    L["blue"] = principled("livery_blue", srgb("#21428C"), rough=0.35, coat=0.2, coat_rough=0.25)
    L["red_bird"] = principled("livery_red", srgb("#CF2442"), rough=0.35, coat=0.2, coat_rough=0.25)
    L["ink"] = principled("livery_black", srgb("#0F0F10"), rough=0.4, coat=0.2, coat_rough=0.25)
    L["flag_red"] = principled("flag_red", srgb("#B22234"), rough=0.4, coat=0.2, coat_rough=0.25)
    L["flag_blue"] = principled("flag_blue", srgb("#3C3B6E"), rough=0.4, coat=0.2, coat_rough=0.25)
    L["flag_white"] = principled("flag_white", srgb("#F7F7F7"), rough=0.4, coat=0.2, coat_rough=0.25)
    L["bolt"] = principled("bolt", srgb("#8B8C8E"), metallic=1.0, rough=0.35)
    L["orange"] = principled("marking_orange", srgb("#D0682A"), rough=0.45)
    return L
