// API de post-proceso con IA en la nube + editor web. Sin dependencias: solo Node 18+.
// Diseño completo en docs/ia-nube.md.

import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { copyFile, mkdir, rename, rm, stat, statfs, writeFile } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import * as almacen from './almacen.js';
import { ErrorHttp } from './almacen.js';
import { firmaValida, textoEntrada, textoSalida } from './firmas.js';

// Quién procesa: el trabajador simulado (por defecto, sin GPU) o la GPU en Modal (trabajador-gpu/).
const GPU_PROVEEDOR = (process.env.GPU_PROVEEDOR || 'simulado').toLowerCase();
const PROVEEDORES = { simulado: './trabajador.js', modal: './modal.js' };
if (!PROVEEDORES[GPU_PROVEEDOR]) {
  console.error(`GPU_PROVEEDOR=${GPU_PROVEEDOR} no es válido (usa: ${Object.keys(PROVEEDORES).join(', ')})`);
  process.exit(1);
}
const trabajador = await import(PROVEEDORES[GPU_PROVEEDOR]);
if (trabajador.comprobarConfiguracion) {
  try { trabajador.comprobarConfiguracion(); } catch (error) { console.error(error.message); process.exit(1); }
}

const PUERTO = Number(process.env.PORT ?? 8080);
const TOKEN = process.env.API_TOKEN || null;              // si se define, toda la API lo exige
const SECRETO_INTERNO = process.env.SECRETO_INTERNO || null;  // firma HMAC de los avisos del trabajador
const MAX_ARCHIVO = 500 * 1024 * 1024;
const MAX_ESCANEO = 1024 * 1024 * 1024;
const MAX_ARCHIVOS = 2000;
const PUBLICO = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'public');

const TIPOS = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.ply': 'application/octet-stream', '.obj': 'text/plain; charset=utf-8', '.mtl': 'text/plain; charset=utf-8',
  '.usdz': 'model/vnd.usdz+zip', '.spz': 'application/octet-stream', '.svg': 'image/svg+xml',
};

// ---------------------------------------------------------------------------------------------- utilidades

function enviarJSON(res, codigo, datos) {
  const cuerpo = JSON.stringify(datos, null, 2);
  res.writeHead(codigo, { 'Content-Type': TIPOS['.json'], 'Content-Length': Buffer.byteLength(cuerpo) });
  res.end(cuerpo);
}

async function leerCuerpo(req, limite = 1024 * 1024) {
  const trozos = [];
  let total = 0;
  for await (const trozo of req) {
    total += trozo.length;
    if (total > limite) throw new ErrorHttp(413, 'Cuerpo demasiado grande');
    trozos.push(trozo);
  }
  return Buffer.concat(trozos);
}

async function leerJSON(req) {
  const cuerpo = await leerCuerpo(req);
  try { return JSON.parse(cuerpo.toString('utf8') || '{}'); } catch { throw new ErrorHttp(400, 'JSON no válido'); }
}

async function enviarArchivo(req, res, archivo) {
  let info;
  try { info = await stat(archivo); } catch { throw new ErrorHttp(404, 'Archivo no encontrado'); }
  if (!info.isFile()) throw new ErrorHttp(404, 'Archivo no encontrado');
  res.writeHead(200, {
    'Content-Type': TIPOS[path.extname(archivo).toLowerCase()] ?? 'application/octet-stream',
    'Content-Length': info.size,
    'Cache-Control': 'no-cache',
  });
  if (req.method === 'HEAD') return res.end();
  try {
    await pipeline(createReadStream(archivo), res);
  } catch (error) {
    // El cliente cerró la conexión (a veces justo después de recibir el último byte): no es un error nuestro.
    if (error.code !== 'ERR_STREAM_PREMATURE_CLOSE') throw error;
  }
}

function autorizado(req, url) {
  if (!TOKEN) return true;
  const cabecera = req.headers.authorization ?? '';
  const dado = cabecera.startsWith('Bearer ') ? cabecera.slice(7) : url.searchParams.get('token');
  if (!dado) return false;
  const a = Buffer.from(dado), b = Buffer.from(TOKEN);
  return a.length === b.length && timingSafeEqual(a, b);
}

function urlDe(id, zona, ruta) {
  return `/api/escaneos/${id}/${zona}/${ruta.split('/').map(encodeURIComponent).join('/')}`;
}

/// Vista pública de un escaneo (lo que devuelve GET /api/escaneos/:id).
function vista(e) {
  const pendientes = e.archivos.filter((a) => !a.subido).map((a) => a.ruta);
  return {
    id: e.id,
    nombre: e.nombre,
    creado: e.creado,
    actualizado: e.actualizado,
    estado: e.estado,
    progreso: e.progreso,
    etapa: e.etapa,
    error: e.error,
    opciones: e.opciones,
    archivos: { total: e.archivos.length, pendientes },
    resultados: e.resultados.map((ruta) => ({ ruta, url: urlDe(e.id, 'resultado', ruta) })),
    editor: `/editor/${e.id}`,
  };
}

// ------------------------------------------------------------------------------------------------ validación

function validarManifiesto(m) {
  if (typeof m !== 'object' || m === null) throw new ErrorHttp(400, 'Falta el manifiesto');
  const nombre = typeof m.nombre === 'string' && m.nombre.trim() ? m.nombre.trim().slice(0, 120) : 'Escaneo';
  if (!Array.isArray(m.archivos) || m.archivos.length === 0) throw new ErrorHttp(400, '`archivos` debe ser una lista no vacía');
  if (m.archivos.length > MAX_ARCHIVOS) throw new ErrorHttp(400, `Como máximo ${MAX_ARCHIVOS} archivos`);

  const vistos = new Set();
  let total = 0;
  const archivos = m.archivos.map((a) => {
    const ruta = almacen.rutaSegura(a?.ruta);
    if (vistos.has(ruta)) throw new ErrorHttp(400, `Archivo repetido: ${ruta}`);
    vistos.add(ruta);
    if (!Number.isInteger(a.bytes) || a.bytes < 0 || a.bytes > MAX_ARCHIVO) {
      throw new ErrorHttp(400, `Tamaño no válido para ${ruta} (máximo ${MAX_ARCHIVO / 1024 / 1024} MB)`);
    }
    if (a.sha256 != null && !/^[a-f0-9]{64}$/.test(a.sha256)) throw new ErrorHttp(400, `SHA-256 no válido para ${ruta}`);
    total += a.bytes;
    return { ruta, bytes: a.bytes, sha256: a.sha256 ?? null };
  });
  if (total > MAX_ESCANEO) throw new ErrorHttp(400, `El escaneo supera ${MAX_ESCANEO / 1024 / 1024} MB`);
  if (!vistos.has('malla_color.ply') && !vistos.has('escena.json')) {
    throw new ErrorHttp(400, 'Hace falta al menos `malla_color.ply` o `escena.json`');
  }

  const o = m.opciones ?? {};
  const opciones = {
    segmentacion: o.segmentacion !== false,
    completado: o.completado === true,
    splats: o.splats === true,
    // De dónde salen las vistas para la IA: fotos clave reales, imágenes sintéticas de la malla, o lo que haya.
    vistas: ['fotos', 'sinteticas'].includes(o.vistas) ? o.vistas : 'auto',
    vocabulario: Array.isArray(o.vocabulario)
      ? o.vocabulario.filter((p) => typeof p === 'string' && p.trim()).map((p) => p.trim().slice(0, 60)).slice(0, 50)
      : [],
  };
  return { nombre, archivos, opciones };
}

/// `edicion.json` con todos los campos: el `Codable` sintetizado de Swift no usa los valores por defecto
/// de `ObjectEdit` al decodificar, así que una clave que falte haría fallar la lectura en el iPhone.
function normalizarEdicion(datos) {
  const objetos = {};
  const entrada = datos?.objetos;
  if (typeof entrada !== 'object' || entrada === null) throw new ErrorHttp(400, 'Falta `objetos`');
  for (const [nombre, e] of Object.entries(entrada)) {
    if (typeof nombre !== 'string' || nombre.length > 200) continue;
    const num = (v) => (Number.isFinite(v) ? v : 0);
    objetos[nombre] = { borrado: e?.borrado === true, dx: num(e?.dx), dy: num(e?.dy), dz: num(e?.dz), yaw: num(e?.yaw) };
  }
  return { objetos };
}

// ------------------------------------------------------------------------------------------------ espacio en disco

const MARGEN_DISCO = 50 * 1024 * 1024;
const CADUCIDAD_SUBIDA_MS = 2 * 3600 * 1000;   // subidas a medias abandonadas

const mb = (bytes) => Math.round(bytes / 1024 / 1024);

async function espacioLibre() {
  const s = await statfs(almacen.DIRECTORIO_DATOS);
  return s.bavail * s.bsize;
}

function errorSinEspacio(libre, necesario) {
  const detalle = libre != null ? ` (quedan ${mb(libre)} MB${necesario ? ` y el escaneo ocupa ${mb(necesario)} MB` : ''})` : '';
  return new ErrorHttp(507, `El servidor no tiene espacio suficiente${detalle}. Amplía el disco del servidor en Railway.`);
}

/// Una vez procesado, la entrada (fotos, malla) ya no hace falta: el editor y la app usan la salida.
/// Las ediciones hechas en el iPhone se conservan junto al escaneo.
async function liberarEntrada(id) {
  const entrada = path.join(almacen.carpeta(id), 'entrada');
  if (!(await almacen.existe(entrada))) return;
  const edicion = path.join(entrada, 'edicion.json');
  const destino = path.join(almacen.carpeta(id), 'edicion.json');
  if ((await almacen.existe(edicion)) && !(await almacen.existe(destino))) await copyFile(edicion, destino);
  await rm(entrada, { recursive: true, force: true });
}

/// Libera la entrada de los escaneos ya procesados y borra las subidas a medias abandonadas.
async function limpiar() {
  try {
    for (const e of await almacen.listar()) {
      if (e.estado === 'listo') await liberarEntrada(e.id);
      if (e.estado === 'esperando_archivos' && Date.now() - Date.parse(e.actualizado) > CADUCIDAD_SUBIDA_MS) {
        await almacen.borrar(e.id);
        console.log(`[limpieza] subida abandonada ${e.id} (${e.nombre}) borrada`);
      }
    }
  } catch (error) {
    console.error('[limpieza]', error);
  }
}

// ------------------------------------------------------------------------------------------------ rutas

async function crearEscaneo(req, res) {
  const manifiesto = validarManifiesto(await leerJSON(req));
  // Antes de aceptar la subida: ¿cabe? Así el iPhone no sube cientos de MB para nada.
  const necesario = manifiesto.archivos.reduce((t, a) => t + a.bytes, 0);
  await limpiar();
  const libre = await espacioLibre();
  if (necesario + MARGEN_DISCO > libre) throw errorSinEspacio(libre, necesario);
  const e = await almacen.crear(manifiesto);
  enviarJSON(res, 201, {
    ...vista(e),
    // En producción: URLs prefirmadas de R2/S3 que caducan en 1 h.
    subidas: e.archivos.map((a) => ({ ruta: a.ruta, metodo: 'PUT', url: urlDe(e.id, 'archivos', a.ruta) })),
  });
}

async function subirArchivo(req, res, id, ruta) {
  const escaneo = await almacen.leer(id);
  const declarado = escaneo.archivos.find((a) => a.ruta === ruta);
  if (!declarado) throw new ErrorHttp(400, `${ruta} no está en el manifiesto`);
  if (escaneo.estado !== 'esperando_archivos') throw new ErrorHttp(409, 'El escaneo ya se está procesando');
  const largo = req.headers['content-length'];
  if (largo !== undefined && Number(largo) !== declarado.bytes) {
    throw new ErrorHttp(400, `${ruta}: se esperaban ${declarado.bytes} bytes y llegan ${largo}`);
  }

  const destino = almacen.rutaArchivo(id, 'entrada', ruta);
  await mkdir(path.dirname(destino), { recursive: true });
  const temporal = `${destino}.subiendo`;
  const hash = createHash('sha256');
  let bytes = 0;
  try {
    await pipeline(req, async function* (origen) {
      for await (const trozo of origen) {
        bytes += trozo.length;
        if (bytes > declarado.bytes) throw new ErrorHttp(400, `${ruta}: más grande de lo declarado`);
        hash.update(trozo);
        yield trozo;
      }
    }, createWriteStream(temporal));
    if (bytes !== declarado.bytes) throw new ErrorHttp(400, `${ruta}: llegaron ${bytes} de ${declarado.bytes} bytes`);
    const suma = hash.digest('hex');
    if (declarado.sha256 && suma !== declarado.sha256) throw new ErrorHttp(400, `${ruta}: el SHA-256 no coincide`);
    await rename(temporal, destino);
  } catch (error) {
    await rm(temporal, { force: true });
    throw error;
  }

  const actualizado = await almacen.actualizar(id, (e) => {
    e.archivos.find((a) => a.ruta === ruta).subido = true;
  });
  enviarJSON(res, 200, { ruta, bytes, pendientes: actualizado.archivos.filter((a) => !a.subido).length });
}

async function procesarEscaneo(req, res, id) {
  const e = await almacen.actualizar(id, (e) => {
    if (e.estado === 'en_cola' || e.estado === 'procesando') return;
    const pendientes = e.archivos.filter((a) => !a.subido).map((a) => a.ruta);
    if (pendientes.length) throw new ErrorHttp(409, `Faltan ${pendientes.length} archivos: ${pendientes.slice(0, 5).join(', ')}`);
    e.estado = 'en_cola';
    e.progreso = 0;
    e.error = null;
    e.etapa = 'En cola';
  });
  trabajador.encolar(id);
  enviarJSON(res, 202, vista(e));
}

/// Aviso de progreso del trabajador con GPU (el simulado no lo usa: escribe directamente en el almacén).
async function progresoInterno(req, res, id) {
  if (!SECRETO_INTERNO) throw new ErrorHttp(503, 'Define SECRETO_INTERNO para aceptar avisos del trabajador');
  const cuerpo = await leerCuerpo(req);
  const firma = Buffer.from(String(req.headers['x-firma'] ?? ''), 'hex');
  const esperada = createHmac('sha256', SECRETO_INTERNO).update(cuerpo).digest();
  if (firma.length !== esperada.length || !timingSafeEqual(firma, esperada)) throw new ErrorHttp(401, 'Firma no válida');
  let aviso;
  try { aviso = JSON.parse(cuerpo.toString('utf8')); } catch { throw new ErrorHttp(400, 'JSON no válido'); }
  const e = await almacen.actualizar(id, async (e) => {
    // Avisos tardíos de un trabajo ya terminado (o de uno anterior al último reinicio) no cambian nada.
    if (e.estado !== 'en_cola' && e.estado !== 'procesando') return;
    if (almacen.ESTADOS.includes(aviso.estado)) e.estado = aviso.estado;
    if (typeof aviso.progreso === 'number') e.progreso = Math.min(1, Math.max(0, aviso.progreso));
    if ('etapa' in aviso) e.etapa = aviso.etapa;
    if ('error' in aviso) e.error = aviso.error;
    if (e.estado === 'listo') e.resultados = await almacen.archivosEn(path.join(almacen.carpeta(id), 'salida'));
  });
  if (e.estado === 'listo') await liberarEntrada(id);
  enviarJSON(res, 200, vista(e));
}

/// La GPU descarga un archivo subido con una URL firmada (firmas.js).
async function entradaInterna(req, res, url, id, ruta) {
  if (!firmaValida(url, (expira) => textoEntrada(id, ruta, expira))) throw new ErrorHttp(403, 'Firma no válida o caducada');
  await enviarArchivo(req, res, almacen.rutaArchivo(id, 'entrada', ruta));
}

/// La GPU sube un resultado con la URL firmada de la salida de su trabajo.
async function salidaInterna(req, res, url, id, ruta) {
  if (!firmaValida(url, (expira) => textoSalida(id, expira))) throw new ErrorHttp(403, 'Firma no válida o caducada');
  const escaneo = await almacen.leer(id);
  if (escaneo.estado !== 'procesando' && escaneo.estado !== 'en_cola') throw new ErrorHttp(409, 'El trabajo no está en marcha');
  const largo = Number(req.headers['content-length'] ?? NaN);
  if (largo > MAX_ARCHIVO) throw new ErrorHttp(413, 'Archivo demasiado grande');
  const destino = almacen.rutaArchivo(id, 'salida', ruta);
  await mkdir(path.dirname(destino), { recursive: true });
  const temporal = `${destino}.subiendo`;
  let bytes = 0;
  try {
    await pipeline(req, async function* (origen) {
      for await (const trozo of origen) {
        bytes += trozo.length;
        if (bytes > MAX_ARCHIVO) throw new ErrorHttp(413, 'Archivo demasiado grande');
        yield trozo;
      }
    }, createWriteStream(temporal));
    await rename(temporal, destino);
  } catch (error) {
    await rm(temporal, { force: true });
    throw error;
  }
  enviarJSON(res, 200, { ruta, bytes });
}

async function servirPublico(req, res, nombre) {
  await enviarArchivo(req, res, path.join(PUBLICO, nombre));
}

async function enrutar(req, res) {
  const url = new URL(req.url, 'http://localhost');
  let partes;
  try { partes = url.pathname.split('/').filter(Boolean).map(decodeURIComponent); } catch { throw new ErrorHttp(400, 'URL no válida'); }
  const m = req.method;

  // Páginas
  if (m === 'GET' || m === 'HEAD') {
    if (partes.length === 0) return servirPublico(req, res, 'index.html');
    if (partes[0] === 'editor' && partes.length === 2) return servirPublico(req, res, 'editor.html');
    if (partes[0] === 'static' && partes.length === 2) return servirPublico(req, res, almacen.rutaSegura(partes[1]));
  }

  if (partes[0] !== 'api') throw new ErrorHttp(404, 'No encontrado');
  if (partes[1] === 'salud' && m === 'GET') {
    return enviarJSON(res, 200, { ok: true, autenticacion: Boolean(TOKEN), gpu: GPU_PROVEEDOR });
  }
  if (partes[1] === 'interno' && partes[2] === 'trabajos') {
    const [, , , idTrabajo, zona, ...rutaInterna] = partes;
    if (zona === 'progreso' && m === 'POST') return progresoInterno(req, res, idTrabajo);
    if (zona === 'entrada' && rutaInterna.length && (m === 'GET' || m === 'HEAD')) {
      return entradaInterna(req, res, url, idTrabajo, almacen.rutaSegura(rutaInterna.join('/')));
    }
    if (zona === 'salida' && rutaInterna.length && m === 'PUT') {
      return salidaInterna(req, res, url, idTrabajo, almacen.rutaSegura(rutaInterna.join('/')));
    }
    throw new ErrorHttp(404, 'No encontrado');
  }
  if (!autorizado(req, url)) throw new ErrorHttp(401, 'Falta el token (Authorization: Bearer … o ?token=…)');
  if (partes[1] !== 'escaneos') throw new ErrorHttp(404, 'No encontrado');

  const [, , id, accion, ...resto] = partes;
  const ruta = resto.join('/');

  if (!id) {
    if (m === 'GET') return enviarJSON(res, 200, (await almacen.listar()).map(vista));
    if (m === 'POST') return crearEscaneo(req, res);
  } else if (!accion) {
    if (m === 'GET') return enviarJSON(res, 200, vista(await almacen.leer(id)));
    if (m === 'DELETE') { await almacen.borrar(id); res.writeHead(204); return res.end(); }
  } else if (accion === 'archivos' && ruta) {
    if (m === 'PUT') return subirArchivo(req, res, id, almacen.rutaSegura(ruta));
    if (m === 'GET' || m === 'HEAD') return enviarArchivo(req, res, almacen.rutaArchivo(id, 'entrada', ruta));
  } else if (accion === 'resultado' && ruta && (m === 'GET' || m === 'HEAD')) {
    return enviarArchivo(req, res, almacen.rutaArchivo(id, 'salida', ruta));
  } else if (accion === 'procesar' && m === 'POST') {
    return procesarEscaneo(req, res, id);
  } else if (accion === 'edicion') {
    const archivo = path.join(almacen.carpeta(id), 'edicion.json');
    if (m === 'GET') {
      await almacen.leer(id);
      if (await almacen.existe(archivo)) return enviarArchivo(req, res, archivo);
      // Si no hay ediciones web todavía, las que vinieron del iPhone.
      const delIphone = almacen.rutaArchivo(id, 'entrada', 'edicion.json');
      if (await almacen.existe(delIphone)) return enviarArchivo(req, res, delIphone);
      return enviarJSON(res, 200, { objetos: {} });
    }
    if (m === 'PUT') {
      await almacen.leer(id);
      const edicion = normalizarEdicion(await leerJSON(req));
      await writeFile(archivo, JSON.stringify(edicion, null, 2));
      return enviarJSON(res, 200, edicion);
    }
  }
  throw new ErrorHttp(405, 'Método o ruta no admitidos');
}

// ------------------------------------------------------------------------------------------------ arranque

const servidor = http.createServer(async (req, res) => {
  const t0 = Date.now();
  try {
    await enrutar(req, res);
  } catch (error) {
    if (error?.code === 'ENOSPC') error = errorSinEspacio(await espacioLibre().catch(() => null));
    const codigo = error instanceof ErrorHttp ? error.codigo : 500;
    if (codigo === 500) console.error(error);
    if (!res.headersSent) enviarJSON(res, codigo, { error: codigo === 500 ? 'Error interno' : error.message });
    else res.destroy();
  } finally {
    if (process.env.REGISTRO !== '0') console.log(`${req.method} ${req.url.split('?')[0]} → ${res.statusCode} (${Date.now() - t0} ms)`);
  }
});

await mkdir(almacen.DIRECTORIO_DATOS, { recursive: true });
await trabajador.reanudarPendientes();
await limpiar();
setInterval(limpiar, 15 * 60 * 1000).unref();
servidor.listen(PUERTO, () => {
  console.log(`Escáner LiDAR · IA en la nube (prototipo) en http://localhost:${PUERTO}`);
  console.log(`Datos en ${almacen.DIRECTORIO_DATOS}${TOKEN ? ' · API protegida con API_TOKEN' : ''}`);
  console.log(GPU_PROVEEDOR === 'modal' ? `IA: GPU en Modal (${process.env.MODAL_URL})` : 'IA: trabajador simulado (GPU_PROVEEDOR=modal para la GPU real)');
});
