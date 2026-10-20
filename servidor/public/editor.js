// Editor web de escenas: equivalente a SceneViewer.swift.
// Lee escena.json + objetos/*.ply + textura.jpg y guarda las ediciones en edicion.json con el mismo formato
// que el iPhone: { "objetos": { "<nombre de nodo>": { dx, dy, dz, yaw, borrado } } }.
// Cada objeto gira alrededor del centro de su caja, en el eje vertical (Y), igual que en la app.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const $ = (id) => document.getElementById(id);
const idEscaneo = location.pathname.split('/').filter(Boolean)[1];

// Token opcional (si el servidor tiene API_TOKEN): ?token=… la primera vez, luego se recuerda.
let token = new URLSearchParams(location.search).get('token');
try {
  if (token) localStorage.setItem('token', token); else token = localStorage.getItem('token');
} catch { /* almacenamiento no disponible */ }
const conToken = (url) => (token ? `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : url);

async function api(ruta, opciones = {}) {
  const r = await fetch(conToken(`/api/escaneos/${idEscaneo}${ruta}`), opciones);
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.error ?? `${r.status} ${r.statusText}`);
  return r.json();
}

/// Igual que `ModelExporter.objectName`: sin espacios ni acentos.
function nombreNodo(nombre) {
  const plano = nombre.normalize('NFD').replace(/\p{M}/gu, '');
  const limpio = [...plano].map((c) => (/^[A-Za-z0-9]$/.test(c) ? c : '_')).join('');
  return /^[0-9]/.test(limpio) ? `_${limpio}` : limpio;
}

// ------------------------------------------------------------------------------------------------ escena

const contenedor = $('lienzo');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
contenedor.appendChild(renderer.domElement);

const escena = new THREE.Scene();
escena.background = new THREE.Color(0x000000);
escena.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sol = new THREE.DirectionalLight(0xffffff, 1.4);
sol.position.set(2, 5, 3);
escena.add(sol);

const camara = new THREE.PerspectiveCamera(55, 1, 0.01, 200);
const orbita = new OrbitControls(camara, renderer.domElement);
orbita.enableDamping = true;

const gizmo = new TransformControls(camara, renderer.domElement);
gizmo.setSpace('world');
escena.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);   // three ≥ r169 separa el ayudante

function redimensionar() {
  const { clientWidth: w, clientHeight: h } = contenedor;
  renderer.setSize(w, h);
  camara.aspect = w / Math.max(h, 1);
  camara.updateProjectionMatrix();
}
new ResizeObserver(redimensionar).observe(contenedor);
renderer.setAnimationLoop(() => { orbita.update(); renderer.render(escena, camara); });

// ------------------------------------------------------------------------------------------------ estado

/** @type {{id, titulo, detalle, estructura, grupo: THREE.Group, malla: THREE.Mesh, centro: THREE.Vector3}[]} */
let items = [];
let ediciones = {};          // id → { dx, dy, dz, yaw, borrado }
let ocultos = new Set();
let seleccion = null;
let modo = null;             // 'translate' | 'rotate' | null
let sombreado = false;
let textura = null;
const pilaDeshacer = [];
let pendiente = null;

const edicionVacia = () => ({ dx: 0, dy: 0, dz: 0, yaw: 0, borrado: false });
const edicionDe = (id) => ediciones[id] ?? edicionVacia();
const iguales = (a, b) => ['dx', 'dy', 'dz', 'yaw', 'borrado'].every((k) => a[k] === b[k]);

function aplicarTransformacion(item) {
  const e = edicionDe(item.id);
  item.grupo.position.set(item.centro.x + e.dx, item.centro.y + e.dy, item.centro.z + e.dz);
  item.grupo.rotation.set(0, e.yaw, 0);
}

function material(item) {
  const conTextura = textura && item.malla.geometry.hasAttribute('uv');
  const opciones = {
    map: conTextura ? textura : null,
    vertexColors: !conTextura && item.malla.geometry.hasAttribute('color'),
    side: THREE.DoubleSide,
  };
  return sombreado ? new THREE.MeshLambertMaterial(opciones) : new THREE.MeshBasicMaterial(opciones);
}

function aplicarAspecto() {
  for (const item of items) {
    aplicarTransformacion(item);
    item.grupo.visible = !ocultos.has(item.id) && !edicionDe(item.id).borrado;
    const opaco = seleccion === null || seleccion === item.id;
    const m = item.malla.material;
    m.transparent = !opaco;
    m.opacity = opaco ? 1 : 0.25;
    m.depthWrite = opaco;
  }
  const item = items.find((i) => i.id === seleccion);
  if (item && modo && !item.estructura) {
    gizmo.attach(item.grupo);
    gizmo.setMode(modo);
    gizmo.showX = modo === 'translate';
    gizmo.showZ = modo === 'translate';
    gizmo.showY = modo === 'rotate';
  } else {
    gizmo.detach();
  }
  pintarInterfaz();
}

// ------------------------------------------------------------------------------------------------ ediciones

let temporizadorGuardado;
function guardar() {
  $('guardado').textContent = 'Guardando…';
  clearTimeout(temporizadorGuardado);
  temporizadorGuardado = setTimeout(async () => {
    try {
      await api('/edicion', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: JSON.stringify({ objetos: ediciones }),
      });
      $('guardado').textContent = 'Guardado';
    } catch (error) {
      $('guardado').textContent = `No se pudo guardar: ${error.message}`;
    }
  }, 300);
}

function cambiar(id, cambio) {
  const anterior = edicionDe(id);
  const nueva = { ...anterior };
  cambio(nueva);
  if (iguales(anterior, nueva)) return;
  pilaDeshacer.push({ id, edicion: anterior });
  ediciones[id] = nueva;
  aplicarAspecto();
  guardar();
}

gizmo.addEventListener('dragging-changed', (evento) => {
  orbita.enabled = !evento.value;
  if (!seleccion) return;
  if (evento.value) {
    pendiente = { id: seleccion, edicion: edicionDe(seleccion) };
  } else if (pendiente) {
    if (!iguales(pendiente.edicion, edicionDe(pendiente.id))) {
      pilaDeshacer.push(pendiente);
      guardar();
    }
    pendiente = null;
    pintarInterfaz();
  }
});

gizmo.addEventListener('objectChange', () => {
  const item = items.find((i) => i.id === seleccion);
  if (!item) return;
  const e = { ...edicionDe(item.id) };
  e.dx = item.grupo.position.x - item.centro.x;
  e.dy = item.grupo.position.y - item.centro.y;
  e.dz = item.grupo.position.z - item.centro.z;
  e.yaw = item.grupo.rotation.y;
  ediciones[item.id] = e;
});

function deshacer() {
  const ultimo = pilaDeshacer.pop();
  if (!ultimo) return;
  ediciones[ultimo.id] = ultimo.edicion;
  aplicarAspecto();
  guardar();
}

function seleccionar(id) {
  seleccion = id === seleccion ? null : id;
  modo = null;
  aplicarAspecto();
}

// ------------------------------------------------------------------------------------------------ interfaz

function fila(item, borrado) {
  const li = document.createElement('li');
  li.className = (item.id === seleccion ? 'seleccionado ' : '') + (borrado ? 'borrado' : '');
  li.innerHTML = '<div class="texto"><div></div><div class="secundario"></div></div><button></button>';
  li.querySelector('.texto div').textContent = item.titulo;
  li.querySelector('.secundario').textContent = item.detalle;
  const boton = li.querySelector('button');
  if (borrado) {
    boton.textContent = '↶ Recuperar';
    boton.onclick = (e) => { e.stopPropagation(); cambiar(item.id, (x) => { x.borrado = false; }); };
  } else {
    boton.textContent = ocultos.has(item.id) ? '◌' : '●';
    boton.title = ocultos.has(item.id) ? 'Mostrar' : 'Ocultar';
    boton.onclick = (e) => {
      e.stopPropagation();
      if (ocultos.has(item.id)) ocultos.delete(item.id); else ocultos.add(item.id);
      if (ocultos.has(item.id) && seleccion === item.id) seleccion = null;
      aplicarAspecto();
    };
    li.onclick = () => seleccionar(item.id);
  }
  return li;
}

function pintarInterfaz() {
  const borrados = items.filter((i) => edicionDe(i.id).borrado);
  $('activos').replaceChildren(...items.filter((i) => !edicionDe(i.id).borrado).map((i) => fila(i, false)));
  $('borrados').replaceChildren(...borrados.map((i) => fila(i, true)));
  $('tituloBorrados').hidden = borrados.length === 0;
  $('deshacer').disabled = pilaDeshacer.length === 0;

  const item = items.find((i) => i.id === seleccion);
  $('seleccion').hidden = !item;
  $('ayuda').hidden = Boolean(item) || items.length === 0;
  if (!item) return;
  $('selTitulo').textContent = item.titulo;
  $('selDetalle').textContent = item.detalle;
  for (const id of ['mover', 'girar', 'subir', 'bajar']) $(id).hidden = item.estructura;
  $('mover').classList.toggle('activo', modo === 'translate');
  $('girar').classList.toggle('activo', modo === 'rotate');
}

$('deshacer').onclick = deshacer;
$('cerrar').onclick = () => seleccionar(null);
$('mover').onclick = () => { modo = modo === 'translate' ? null : 'translate'; aplicarAspecto(); };
$('girar').onclick = () => { modo = modo === 'rotate' ? null : 'rotate'; aplicarAspecto(); };
$('subir').onclick = () => cambiar(seleccion, (e) => { e.dy += 0.02; });
$('bajar').onclick = () => cambiar(seleccion, (e) => { e.dy -= 0.02; });
$('aislar').onclick = () => { ocultos = new Set(items.map((i) => i.id).filter((id) => id !== seleccion)); aplicarAspecto(); };
$('borrar').onclick = () => { const id = seleccion; seleccion = null; modo = null; cambiar(id, (e) => { e.borrado = true; }); };
$('mostrarTodo').onclick = () => { ocultos = new Set(); seleccion = null; modo = null; aplicarAspecto(); };
$('sombreado').onchange = (e) => {
  sombreado = e.target.checked;
  for (const item of items) { item.malla.material.dispose(); item.malla.material = material(item); }
  aplicarAspecto();
};
$('restaurar').onclick = () => {
  if (!confirm('¿Deshacer todos los cambios (movimientos, giros y borrados)?')) return;
  ediciones = {};
  pilaDeshacer.length = 0;
  aplicarAspecto();
  guardar();
};
addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); deshacer(); }
  if (e.key === 'Escape') seleccionar(null);
  if ((e.key === 'Delete' || e.key === 'Backspace') && seleccion && document.activeElement === document.body) $('borrar').click();
});

// Clic (sin arrastrar) sobre la vista: selecciona el objeto que hay debajo.
const rayo = new THREE.Raycaster();
let inicioClic = null;
renderer.domElement.addEventListener('pointerdown', (e) => { inicioClic = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!inicioClic || gizmo.dragging || Math.hypot(e.clientX - inicioClic[0], e.clientY - inicioClic[1]) > 4) return;
  if (modo && seleccion) return;   // mientras se edita, los clics son para el gizmo
  const r = renderer.domElement.getBoundingClientRect();
  escena.updateMatrixWorld();
  camara.updateMatrixWorld();
  rayo.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camara);
  const visibles = items.filter((i) => i.grupo.visible).map((i) => i.malla);
  const golpe = rayo.intersectObjects(visibles, false)[0];
  seleccionar(golpe ? golpe.object.userData.id : null);
});

// ------------------------------------------------------------------------------------------------ carga

function detalle(o) {
  const dims = (o.dimensiones ?? []).map((d) => d.toFixed(2)).join(' × ');
  const partes = [dims ? `${dims} m` : null];
  if (o.confianzaIA != null) partes.push(`IA ${Math.round(o.confianzaIA * 100)} %`);
  if (o.fraccionInferida > 0) partes.push(`${Math.round(o.fraccionInferida * 100)} % inferido`);
  return partes.filter(Boolean).join(' · ');
}

async function cargar() {
  const info = await api('');
  $('nombre').textContent = info.nombre;
  $('estado').textContent = { listo: 'Procesado en la nube', procesando: `Procesando: ${info.etapa ?? ''}`,
                             en_cola: 'En cola', esperando_archivos: 'Subida incompleta', error: `Error: ${info.error}` }[info.estado] ?? info.estado;

  // Resultado de la nube si existe; si no, lo que subió el iPhone.
  const procesado = info.resultados.some((r) => r.ruta === 'escena.json');
  const base = `/api/escaneos/${idEscaneo}/${procesado ? 'resultado' : 'archivos'}`;

  let objetos = [];
  try {
    const r = await fetch(conToken(`${base}/escena.json`));
    if (r.ok) objetos = (await r.json()).objetos ?? [];
  } catch { /* sin escena.json */ }
  if (objetos.length === 0) objetos = [{ nombre: 'Escaneo completo', archivo: 'malla_color.ply', estructura: false }];

  // Solo se pide si existe: tras el procesado lo dice la lista de resultados; antes, se comprueba.
  const hayTextura = procesado
    ? info.resultados.some((r) => r.ruta === 'textura.jpg')
    : (await fetch(conToken(`${base}/textura.jpg`), { method: 'HEAD' })).ok;
  if (hayTextura) {
    textura = await new THREE.TextureLoader().loadAsync(conToken(`${base}/textura.jpg`));
    textura.colorSpace = THREE.SRGBColorSpace;
  }

  const cargador = new PLYLoader();
  const cargados = await Promise.all(objetos.map(async (o) => {
    try {
      return [o, await cargador.loadAsync(conToken(`${base}/${o.archivo.split('/').map(encodeURIComponent).join('/')}`))];
    } catch (error) {
      console.warn(`No se pudo cargar ${o.archivo}`, error);
      return null;
    }
  }));

  const caja = new THREE.Box3();
  for (const [o, geometria] of cargados.filter(Boolean)) {
    geometria.computeBoundingBox();
    if (!geometria.hasAttribute('normal')) geometria.computeVertexNormals();
    const centro = geometria.boundingBox.getCenter(new THREE.Vector3());
    caja.union(geometria.boundingBox);
    const id = nombreNodo(o.nombre);
    const item = { id, titulo: o.nombre, detalle: detalle(o), estructura: Boolean(o.estructura), centro };
    item.malla = new THREE.Mesh(geometria);
    item.malla.position.copy(centro).negate();   // el pivote en el centro de la caja
    item.malla.userData.id = id;
    item.malla.material = material(item);
    item.grupo = new THREE.Group();
    item.grupo.add(item.malla);
    escena.add(item.grupo);
    items.push(item);
  }
  if (items.length === 0) throw new Error('No se encontró ninguna malla en este escaneo.');

  ediciones = (await api('/edicion')).objetos ?? {};
  ocultos = new Set(items.filter((i) => i.titulo === 'Techo').map((i) => i.id));

  // Cámara en diagonal por encima de la escena, como en el iPhone.
  const centro = caja.getCenter(new THREE.Vector3());
  const radio = Math.max(caja.getSize(new THREE.Vector3()).length() / 2, 0.5);
  camara.position.set(centro.x + radio * 1.2, centro.y + radio * 1.4, centro.z + radio * 1.2);
  camara.far = radio * 20;
  camara.updateProjectionMatrix();
  orbita.target.copy(centro);

  aplicarAspecto();
}

cargar().catch((error) => {
  $('error').hidden = false;
  $('error').textContent = error.message;
  $('nombre').textContent = 'Sin modelo';
});
