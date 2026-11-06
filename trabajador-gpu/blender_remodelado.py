r"""Remodelado de la habitación, paso 5: montar la escena limpia en Blender, guardarla y renderizar una vista previa.

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b --factory-startup ^
        --python blender_remodelado.py -- "C:\…\Habitacion_…"

Construye con remodelado/envolvente.json las paredes, el suelo y el techo como piezas sólidas con su color real; si
el escaneo trae habitacion.json (RoomPlan), recorta en las paredes sus puertas y ventanas. Coloca cada mueble:
el modelo escaneado en detalle (Object Capture) si existe, y si no la forma ajustada de ajustar_muebles.py con su
color por vértice. Añade luz y una cámara en una esquina, guarda remodelado/habitacion_remodelada.blend y escribe
remodelado/vista_previa.png. Coordenadas: el mundo de ARKit (Y arriba) pasa a Z arriba como en cualquier importador.
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

carpeta = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
remodelado = carpeta / "remodelado"
env = json.loads((remodelado / "envolvente.json").read_text("utf-8"))
GROSOR = 0.10
A_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))   # (x, y, z) ARKit → (x, -z, y)


def b(v):
    return Vector((v[0], -v[2], v[1]))


def lineal(c):
    return [(x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def material(nombre, rgb=None, rugosidad=0.85, por_vertice=False, transmision=0.0):
    m = bpy.data.materials.new(nombre)
    m.use_nodes = True
    nodos = m.node_tree.nodes
    p = next(n for n in nodos if n.type == "BSDF_PRINCIPLED")
    p.inputs["Roughness"].default_value = rugosidad
    if transmision:
        # Cristal: luz de día pálida y emisiva. Con transmisión real EEVEE (y la Vista rápida del iPhone) lo
        # pintaba negro y parecía un hueco oscuro en la pared.
        p.inputs["Emission Color"].default_value = (*lineal(rgb), 1)
        p.inputs["Emission Strength"].default_value = transmision
    if por_vertice:
        attr = nodos.new("ShaderNodeVertexColor")
        m.node_tree.links.new(attr.outputs["Color"], p.inputs["Base Color"])
    elif rgb is not None:
        # El color medido en las fotos ya incluye la luz de la habitación: se aclara un poco como albedo.
        p.inputs["Base Color"].default_value = (*lineal([min(1.0, x * 1.25) for x in rgb]), 1)
    return m


def coleccion(nombre):
    c = bpy.data.collections.new(nombre)
    bpy.context.scene.collection.children.link(c)
    return c


def caja(nombre, matriz, tam, mat, col):
    malla = bpy.data.meshes.new(nombre)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(tam), verts=bm.verts)
    bm.to_mesh(malla)
    bm.free()
    o = bpy.data.objects.new(nombre, malla)
    o.matrix_world = matriz
    o.data.materials.append(mat)
    col.objects.link(o)
    return o


def base(u, v, n, origen):
    return Matrix(((u.x, v.x, n.x, origen.x), (u.y, v.y, n.y, origen.y), (u.z, v.z, n.z, origen.z), (0, 0, 0, 1)))


# ----------------------------------------------------------------------------------------- escena limpia
bpy.ops.wm.read_factory_settings(use_empty=True)
escena = bpy.context.scene
escena.unit_settings.system = "METRIC"
col_env = coleccion("Envolvente")
paredes = {}
def losa(nombre, poli, y, mat, col):
    """Suelo o techo con la forma real de la habitación (contorno de RoomPlan), de GROSOR de espesor."""
    malla = bpy.data.meshes.new(nombre)
    bm = bmesh.new()
    cara = bm.faces.new([bm.verts.new(b((x, y, z))) for x, z in poli])
    ext = bmesh.ops.extrude_face_region(bm, geom=[cara])
    hacia = -1 if nombre == "suelo" else 1
    bmesh.ops.translate(bm, vec=Vector((0, 0, hacia * GROSOR)),
                        verts=[e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(malla)
    bm.free()
    o = bpy.data.objects.new(nombre, malla)
    o.data.materials.append(mat)
    col.objects.link(o)
    return o


poli = env.get("contorno_xz")
for nombre, pl in env["planos"].items():
    c, u, v, n = b(pl["centro"]), b(pl["u"]), b(pl["v"]), b(pl["normal"])
    if poli and nombre in ("suelo", "techo"):
        losa(nombre, poli, env["suelo_y"] if nombre == "suelo" else env["techo_y"],
             material(f"Mat_{nombre}", pl["color"], 0.55 if nombre == "suelo" else 0.9), col_env)
        continue
    extra = 2 * GROSOR if nombre in ("suelo", "techo") else 0
    rug = 0.55 if nombre == "suelo" else 0.9
    o = caja(nombre, base(u, v, n, c - n * GROSOR / 2), (pl["ancho_m"] + extra, pl["alto_m"] + extra, GROSOR),
             material(f"Mat_{nombre}", pl["color"], rug), col_env)
    if nombre.startswith("pared"):
        paredes[nombre] = (o, c, n)

# Puertas y ventanas de RoomPlan: hueco en la pared más cercana y una hoja o un cristal dentro.
sala = carpeta / "habitacion.json"
huecos = json.loads(sala.read_text("utf-8")).get("openings", []) if sala.exists() else []
if huecos:
    col_h = coleccion("Puertas y ventanas")
    mat_puerta, mat_cristal = material("Puerta", [0.55, 0.42, 0.3], 0.6), material("Cristal", [0.9, 0.95, 1.0], 0.05,
                                                                                   transmision=1.0)
    for i, h in enumerate(huecos):
        c, n = b(h["center"]), b(h["normal"]).normalized()
        nombre, (pared, pc, pn) = min(paredes.items(), key=lambda kv: abs((c - kv[1][1]).dot(kv[1][2])))
        c = c - pn * (c - pc).dot(pn)                                 # sobre el plano interior de esa pared
        u = Vector((0, 0, 1)).cross(pn).normalized()
        corte = caja(f"corte_{i}", base(u, Vector((0, 0, 1)), pn, c - pn * GROSOR / 2),
                     (h["width"], h["height"], GROSOR * 3), mat_puerta, col_h)
        mod = pared.modifiers.new(f"hueco_{i}", "BOOLEAN")
        mod.operation, mod.object = "DIFFERENCE", corte
        corte.hide_render = corte.hide_viewport = True
        if h["kind"] == "door":
            caja(f"Puerta {i + 1}", base(u, Vector((0, 0, 1)), pn, c - pn * GROSOR * 0.8),
                 (h["width"] - 0.02, h["height"] - 0.01, 0.04), mat_puerta, col_h)
        elif h["kind"] == "window":
            caja(f"Ventana {i + 1}", base(u, Vector((0, 0, 1)), pn, c - pn * GROSOR / 2),
                 (h["width"], h["height"], 0.01), mat_cristal, col_h)

# ----------------------------------------------------------------------------------------- muebles
objetos = remodelado / "objetos"
info = json.loads((objetos / "objetos.json").read_text("utf-8")) if (objetos / "objetos.json").exists() else {"muebles": []}
ajuste = {a["clave"]: a for a in json.loads((objetos / "ajuste.json").read_text("utf-8"))} \
    if (objetos / "ajuste.json").exists() else {}
col_m = coleccion("Muebles")
mat_v = material("Color escaneado", por_vertice=True, rugosidad=0.7)
colocados = []
for m in info["muebles"]:
    antes = set(bpy.data.objects)
    if m.get("detalle") and (carpeta / m["detalle"]).exists():
        bpy.ops.wm.usd_import(filepath=str(carpeta / m["detalle"]))
        nuevos = [o for o in bpy.data.objects if o not in antes]
        D = Matrix([m["matriz_detalle"][i::4] for i in range(4)])       # modelo → mueble (por columnas)
        c, s = math.cos(m["yaw"]), math.sin(m["yaw"])
        W = Matrix(((c, 0, s, m["posicion"][0]), (0, 1, 0, m["posicion"][1]), (-s, 0, c, m["posicion"][2]), (0, 0, 0, 1)))
        # El importador ya pasó el USDZ (Y arriba) a Z arriba: se deshace, se coloca en ARKit y se vuelve a pasar.
        for o in nuevos:
            if o.parent is None:
                o.matrix_world = A_BLENDER @ W @ D @ A_BLENDER.inverted() @ o.matrix_world
        estado = "escaneo en detalle"
    elif (objetos / f"{m['clave']}.ply").exists():
        bpy.ops.wm.ply_import(filepath=str(objetos / f"{m['clave']}.ply"), forward_axis="NEGATIVE_Z", up_axis="Y")
        nuevos = [o for o in bpy.data.objects if o not in antes]
        for o in nuevos:
            o.data.materials.clear()
            o.data.materials.append(mat_v)
            # Las formas generadas y las cajas pueden traer caras del revés: se orientan todas hacia fuera.
            bm = bmesh.new()
            bm.from_mesh(o.data)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(o.data)
            bm.free()
        estado = ajuste.get(m["clave"], {}).get("estado", "")
    else:
        continue
    for o in nuevos:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col_m.objects.link(o)
        o.name = m["nombre"]
        o["estado_remodelado"] = estado
    colocados.append(f"{m['nombre']} ({estado})")

# ----------------------------------------------------------------------------------------- luz, cámara y render
suelo, techo = env["planos"]["suelo"], env["planos"]["techo"]
centro = (b(suelo["centro"]) + b(techo["centro"])) / 2
luz = bpy.data.objects.new("Plafón", bpy.data.lights.new("Plafón", "AREA"))
luz.data.energy, luz.data.size = 350, 1.0
luz.location = b(techo["centro"]) - Vector((0, 0, 0.05))
escena.collection.objects.link(luz)
relleno = bpy.data.objects.new("Relleno", bpy.data.lights.new("Relleno", "POINT"))
relleno.data.energy = 120
relleno.location = centro
escena.collection.objects.link(relleno)
mundo = bpy.data.worlds.new("Mundo")
mundo.use_nodes = True
next(n for n in mundo.node_tree.nodes if n.type == "BACKGROUND").inputs[1].default_value = 0.4
escena.world = mundo

cam = bpy.data.objects.new("Vista", bpy.data.cameras.new("Vista"))
cam.data.lens = 14
# Esquina alta: desde casi el techo se ve toda la habitación por encima de los muebles (la esquina del contorno,
# o del rectángulo del suelo, más lejana del centro).
if poli:
    esquinas = [b((x, suelo["centro"][1], z)) for x, z in poli]
else:
    su, sv = b(suelo["u"]) * suelo["ancho_m"] / 2, b(suelo["v"]) * suelo["alto_m"] / 2
    esquinas = [b(suelo["centro"]) + su + sv, b(suelo["centro"]) - su - sv]
esquina = max(esquinas, key=lambda p: (p.xy - centro.xy).length)
ojo = centro + (esquina - centro) * 0.85
ojo.z = b(techo["centro"]).z - 0.25
mira = centro.copy()
mira.z = b(suelo["centro"]).z + 0.6
cam.location = ojo
cam.rotation_euler = (mira - ojo).to_track_quat("-Z", "Y").to_euler()
escena.collection.objects.link(cam)
escena.camera = cam
try:
    escena.render.engine = "BLENDER_EEVEE"
except TypeError:
    escena.render.engine = "BLENDER_EEVEE_NEXT"
escena.render.resolution_x, escena.render.resolution_y = 1600, 900
escena.render.filepath = str(remodelado / "vista_previa.png")
bpy.ops.wm.save_as_mainfile(filepath=str(remodelado / "habitacion_remodelada.blend"))
# USDZ para el iPhone (Vista rápida y realidad aumentada); sin luces ni cámara, solo la habitación.
try:
    bpy.ops.wm.usd_export(filepath=str(remodelado / "habitacion_remodelada.usdz"), export_lights=False,
                          export_cameras=False)
except Exception as error:  # noqa: BLE001 — el .blend ya está guardado
    print(f"Sin USDZ: {error}", flush=True)
bpy.ops.render.render(write_still=True)
print(f"Remodelado en Blender: {len(colocados)} muebles ({', '.join(colocados)}), {len(huecos)} puertas y ventanas "
      f"→ {remodelado / 'habitacion_remodelada.blend'}", flush=True)
