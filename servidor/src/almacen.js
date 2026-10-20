// Almacenamiento en disco local. En producción, `entrada/` y `salida/` irían a R2/S3 con URLs prefirmadas
// y el estado a Postgres; la interfaz de este módulo es la misma que tendría esa versión.
//
// DATOS/escaneos/<id>/
//   estado.json      escaneo: manifiesto, estado, progreso, etapa, errores
//   entrada/...      archivos subidos por la app (rutas relativas del manifiesto)
//   salida/...       resultados del trabajador
//   edicion.json     ediciones hechas en el editor web (mismo formato que el iPhone)

import { randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

export const DIRECTORIO_DATOS = path.resolve(process.env.DATOS ?? path.join(process.cwd(), 'datos'));
const RAIZ = path.join(DIRECTORIO_DATOS, 'escaneos');

export const ESTADOS = ['esperando_archivos', 'en_cola', 'procesando', 'listo', 'error'];

const ID_VALIDO = /^[a-f0-9-]{36}$/;

export function carpeta(id) {
  if (!ID_VALIDO.test(id)) throw new ErrorHttp(404, 'Escaneo no encontrado');
  return path.join(RAIZ, id);
}

export class ErrorHttp extends Error {
  constructor(codigo, mensaje) {
    super(mensaje);
    this.codigo = codigo;
  }
}

/// Ruta relativa segura: sin `..`, sin rutas absolutas y solo caracteres razonables.
export function rutaSegura(ruta) {
  const limpia = String(ruta ?? '').replace(/\\/g, '/');
  if (!limpia || limpia.length > 200 || limpia.startsWith('/') ||
      limpia.split('/').some((p) => p === '' || p === '.' || p === '..') ||
      !/^[\p{L}\p{N} ._\-/]+$/u.test(limpia)) {
    throw new ErrorHttp(400, `Ruta de archivo no válida: ${ruta}`);
  }
  return limpia;
}

export async function crear(manifiesto) {
  const id = randomUUID();
  const escaneo = {
    id,
    nombre: manifiesto.nombre,
    creado: new Date().toISOString(),
    actualizado: new Date().toISOString(),
    estado: 'esperando_archivos',
    progreso: 0,
    etapa: null,
    error: null,
    opciones: manifiesto.opciones,
    archivos: manifiesto.archivos.map((a) => ({ ruta: a.ruta, bytes: a.bytes, sha256: a.sha256 ?? null, subido: false })),
    resultados: [],
  };
  await mkdir(path.join(carpeta(id), 'entrada'), { recursive: true });
  await guardar(escaneo);
  return escaneo;
}

export async function leer(id) {
  try {
    return JSON.parse(await readFile(path.join(carpeta(id), 'estado.json'), 'utf8'));
  } catch (e) {
    if (e.code === 'ENOENT') throw new ErrorHttp(404, 'Escaneo no encontrado');
    throw e;
  }
}

export async function guardar(escaneo) {
  escaneo.actualizado = new Date().toISOString();
  const destino = path.join(carpeta(escaneo.id), 'estado.json');
  const temporal = `${destino}.${process.pid}.tmp`;
  await writeFile(temporal, JSON.stringify(escaneo, null, 2));
  await rename(temporal, destino);   // escritura atómica
}

/// Cambia el escaneo con `cambio(escaneo)` en exclusión mutua (las peticiones concurrentes se encadenan).
const cerrojos = new Map();
export function actualizar(id, cambio) {
  const anterior = cerrojos.get(id) ?? Promise.resolve();
  const siguiente = anterior.catch(() => {}).then(async () => {
    const escaneo = await leer(id);
    await cambio(escaneo);
    await guardar(escaneo);
    return escaneo;
  });
  cerrojos.set(id, siguiente);
  siguiente.finally(() => { if (cerrojos.get(id) === siguiente) cerrojos.delete(id); }).catch(() => {});
  return siguiente;
}

export async function listar() {
  let ids = [];
  try {
    ids = (await readdir(RAIZ)).filter((n) => ID_VALIDO.test(n));
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  const escaneos = [];
  for (const id of ids) {
    try { escaneos.push(await leer(id)); } catch { /* carpeta a medio crear */ }
  }
  return escaneos.sort((a, b) => b.creado.localeCompare(a.creado));
}

export async function borrar(id) {
  await leer(id);
  await rm(carpeta(id), { recursive: true, force: true });
}

export function rutaArchivo(id, zona, ruta) {
  return path.join(carpeta(id), zona, rutaSegura(ruta));
}

export async function existe(archivo) {
  try { return (await stat(archivo)).isFile(); } catch { return false; }
}

/// Lista recursiva de archivos (rutas relativas con `/`).
export async function archivosEn(directorio, prefijo = '') {
  let entradas = [];
  try { entradas = await readdir(directorio, { withFileTypes: true }); } catch { return []; }
  const out = [];
  for (const e of entradas) {
    const rel = prefijo ? `${prefijo}/${e.name}` : e.name;
    if (e.isDirectory()) out.push(...await archivosEn(path.join(directorio, e.name), rel));
    else out.push(rel);
  }
  return out.sort();
}
