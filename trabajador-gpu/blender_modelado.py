"""Construye en Blender los muebles de <escaneo>/modelado/modelado.json y renderiza desde las cámaras reales.

Se ejecuta dentro de Blender (complemento MCP o Texto > Ejecutar):
    exec(open(r"…/trabajador-gpu/blender_modelado.py", encoding="utf-8").read())
    construir(r"C:\\…\\Entorno_…")                       # muebles en la colección «Modelado_verificado»
    comparar(r"C:\\…\\Entorno_…", ["fot_036.jpg"])       # render desde la cámara de cada foto → modelado/render_*.png

Convención local de los muebles (la de modelado_verificado.py): x largo, y fondo (+y contra la pared), z arriba.
"""

import json
import math

import bpy
import mathutils

SUELO = -1.196


def _mat(nombre, color, rugosidad=0.6, metal=0.0):
    m = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs[0].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rugosidad
    b.inputs["Metallic"].default_value = metal
    return m


def _coleccion():
    col = bpy.data.collections.get("Modelado_verificado") or bpy.data.collections.new("Modelado_verificado")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    return col


def _caja(col, nombre, lo, hi, material, padre, bisel=0.008):
    malla = bpy.data.meshes.new(nombre)
    v = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    caras = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    malla.from_pydata(v, [], caras)
    malla.update()
    o = bpy.data.objects.new(nombre, malla)
    col.objects.link(o)
    o.parent = padre
    o.data.materials.append(material)
    if bisel:
        b = o.modifiers.new("Bisel", "BEVEL")
        b.width, b.segments = bisel, 2
    return o


def _padre(col, nombre, p):
    e = bpy.data.objects.new(nombre, None)
    col.objects.link(e)
    e.location = (p["x"], p["y"], SUELO)
    e.rotation_euler = (0, 0, p["giro"])
    return e


def cama(col, nombre, p):
    blanco = _mat("MV_colchon", (0.80, 0.80, 0.78), 0.85)
    granate = _mat("MV_granate", (0.20, 0.02, 0.035), 0.7)
    almohada = _mat("MV_almohada", (0.85, 0.86, 0.88), 0.9)
    L, W, H = p["largo"], p["ancho"], p["alto"]
    e = _padre(col, nombre, p)
    base = min(0.40, H - 0.18)
    _caja(col, nombre + "_base", (-L / 2, -W / 2, 0.02), (L / 2, W / 2, base), granate, e, 0.02)
    _caja(col, nombre + "_colchon", (-L / 2 + 0.01, -W / 2 + 0.01, base), (L / 2 - 0.01, W / 2 - 0.01, H), blanco, e, 0.05)
    for k, (y0, y1) in enumerate(((-W / 2 + 0.06, -0.03), (0.03, W / 2 - 0.06))):
        _caja(col, f"{nombre}_almohada_{k + 1}", (-L / 2 + 0.05, y0, H - 0.02), (-L / 2 + 0.47, y1, H + 0.13), almohada, e, 0.05)


def escritorio(col, nombre, p, lado):
    madera = _mat("MV_madera", (0.045, 0.016, 0.009), 0.45)
    canto = _mat("MV_canto", (0.30, 0.05, 0.04), 0.4)
    laton = _mat("MV_laton", (0.6, 0.45, 0.15), 0.3, 1.0)
    L, D, H = p["largo"], p["fondo"], p["alto"]
    wc = min(max(p["ancho_cajonero"], 0.25), 0.55 * L)
    e = _padre(col, nombre, p)
    t = 0.035
    s = 1 if lado >= 0 else -1                       # el cajonero en el extremo s·x

    def X(a, b):                                       # tramo en x reflejado según el lado
        return (min(s * a, s * b), max(s * a, s * b))

    _caja(col, nombre + "_tablero", (-L / 2, -D / 2, H - t), (L / 2, D / 2, H), madera, e, 0.006)
    _caja(col, nombre + "_canto", (-L / 2 - 0.004, -D / 2 - 0.004, H - t + 0.006), (L / 2 + 0.004, D / 2 + 0.004, H - 0.006), canto, e, 0)
    x0, x1 = X(L / 2 - wc, L / 2)
    _caja(col, nombre + "_cajonero", (x0, -D / 2 + 0.01, 0.03), (x1, D / 2, H - t), madera, e, 0.006)
    x0, x1 = X(-L / 2, -L / 2 + 0.035)
    _caja(col, nombre + "_lateral", (x0, -D / 2 + 0.01, 0.03), (x1, D / 2, H - t), madera, e, 0.006)
    hc = (H - t - 0.06) / 3
    for k in range(3):                                 # frentes de cajón y tiradores en el frente (−y)
        z0 = 0.05 + k * hc
        x0, x1 = X(L / 2 - wc + 0.02, L / 2 - 0.02)
        _caja(col, f"{nombre}_cajon_{k + 1}", (x0, -D / 2 - 0.012, z0 + 0.01), (x1, -D / 2 + 0.01, z0 + hc - 0.01), madera, e, 0.004)
        xc = (x0 + x1) / 2
        _caja(col, f"{nombre}_tirador_{k + 1}", (xc - 0.06, -D / 2 - 0.03, z0 + hc / 2 - 0.012), (xc + 0.06, -D / 2 - 0.012, z0 + hc / 2 + 0.012), laton, e, 0.003)


def colocar_malla(escaneo, nombre, p, objeto):
    """Coloca un modelo detallado (generado por IA) con la pose ajustada: misma convención que MallaRigida
    (vértices de <nombre>_ia.npz relativos a su pose inicial, centrados en planta y apoyados en el suelo)."""
    import numpy as np

    o = bpy.data.objects.get(objeto)
    if o is None:
        return
    d = np.load(escaneo + rf"\modelado\{nombre}_ia.npz")
    v, (x0, y0, g0) = d["v"], d["pose"]
    c = (v.min(0) + v.max(0)) / 2
    c[2] = v[:, 2].min()
    if "matriz_original" not in o:                    # la pose de antes del ajuste, para poder repetirlo
        o["matriz_original"] = [list(f) for f in o.matrix_world]
    M0 = mathutils.Matrix(o["matriz_original"])
    loc0 = mathutils.Vector((x0, y0, SUELO + float(v[:, 2].min()) * 0))
    # Del objeto original a local (sin su giro y posición), centrar, escalar, girar y llevar a la pose ajustada.
    local = mathutils.Matrix.Rotation(-g0, 4, "Z") @ mathutils.Matrix.Translation(-mathutils.Vector((x0, y0, SUELO)))
    centrar = mathutils.Matrix.Translation(-mathutils.Vector(c))
    e = min(max(p["escala"], 0.6), 1.6)
    final = (mathutils.Matrix.Translation((p["x"], p["y"], SUELO)) @ mathutils.Matrix.Rotation(p["giro"], 4, "Z")
             @ mathutils.Matrix.Scale(e, 4) @ centrar @ local)
    o.matrix_world = final @ M0
    o.hide_set(False)
    o.hide_render = False


def construir(escaneo, sufijo="", variantes=None):
    """Crea los muebles del JSON. `variantes`: {nombre: {"lado": ±1, "giro_extra": rad}} para probar alternativas."""
    global SUELO
    datos = json.load(open(escaneo + r"\modelado\modelado.json", encoding="utf-8"))
    SUELO = datos.pop("_escena", {}).get("suelo", SUELO)
    col = _coleccion()
    for o in list(col.objects):
        if o.name.endswith(sufijo) or sufijo == "":
            bpy.data.objects.remove(o)
    for nombre, d in datos.items():
        p = dict(d["parametros"])
        lado = d.get("lado", 1)
        if variantes and nombre in variantes:
            lado = variantes[nombre].get("lado", lado)
            p["giro"] += variantes[nombre].get("giro_extra", 0.0)
        if d["tipo"] == "Cama":
            cama(col, "Cama_verificada" + sufijo, p)
        elif d["tipo"] == "Escritorio":
            escritorio(col, "Escritorio_verificado" + sufijo, p, lado)
        elif d["tipo"] == "MallaRigida":
            colocar_malla(escaneo, nombre, p, {"silla": "Silla_gamer_IA"}.get(nombre, nombre))
    return [o.name for o in col.objects]


def camara(escaneo, foto):
    """Cámara de Blender idéntica a la de la foto (pose refinada y mundo LiDAR)."""
    k = json.load(open(escaneo + r"\modelado\camaras_mundo.json"))[foto]
    A = mathutils.Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))           # ARKit (Y arriba) → Blender (Z arriba)
    R = A @ mathutils.Matrix(k["c2w_R"]) @ mathutils.Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    M = R.to_4x4()
    M.translation = A @ mathutils.Vector(k["centro"])
    datos = bpy.data.cameras.get("cam_" + foto) or bpy.data.cameras.new("cam_" + foto)
    cam = bpy.data.objects.get("Cam_" + foto) or bpy.data.objects.new("Cam_" + foto, datos)
    if cam.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(cam)
    cam.matrix_world = M
    datos.sensor_fit, datos.sensor_width = "HORIZONTAL", 36
    datos.lens = k["fx"] / k["ancho"] * 36
    datos.shift_x = -(k["cx"] + 0.5 - k["ancho"] / 2) / k["ancho"]
    datos.shift_y = (k["cy"] + 0.5 - k["alto"] / 2) / k["ancho"]
    datos.clip_start = 0.05
    return cam, k["ancho"], k["alto"]


def comparar(escaneo, fotos, etiqueta="", ocultar=("Malla_2DGS", "Fotogrametria", "Prueba")):
    sc = bpy.context.scene
    antes = (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage,
             sc.render.filepath, sc.view_settings.view_transform)
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name.startswith(ocultar):
            o.hide_render = True
    salida = []
    try:
        for foto in fotos:
            cam, w, h = camara(escaneo, foto)
            sc.camera = cam
            sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 50
            sc.view_settings.view_transform = "Standard"
            sc.render.filepath = escaneo + rf"\modelado\render_{foto[:-4]}{etiqueta}.png"
            bpy.ops.render.render(write_still=True)
            salida.append(sc.render.filepath)
    finally:
        (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage,
         sc.render.filepath, sc.view_settings.view_transform) = antes
    return salida
