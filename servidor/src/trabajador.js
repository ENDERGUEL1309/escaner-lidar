// Trabajador SIMULADO: sustituye a la función con GPU (Modal / RunPod) para probar el flujo en local.
// Recorre las mismas etapas que el trabajador real, informa del progreso y deja en `salida/` un resultado
// con el formato definitivo (escena.json v2 + objetos/*.ply + informe.json), pero sin IA: los objetos
// son los mismos que detectó el iPhone.
//
// En producción, `despachar` haría `POST https://api.runpod.ai/v2/<endpoint>/run` o llamaría a una función
// de Modal con el id y URLs prefirmadas, y el progreso llegaría a POST /api/interno/trabajos/<id>/progreso.

import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import * as almacen from './almacen.js';

const PAUSA_MS = Number(process.env.PAUSA_SIMULADA_MS ?? 800);

export function etapasPara(opciones = {}) {
  const etapas = [['descarga', 'Descargando el escaneo']];
  if (opciones.segmentacion !== false) {
    etapas.push(['vocabulario', 'Buscando qué hay en las fotos'],
                ['mascaras', 'Segmentando objetos en las fotos'],
                ['elevacion3d', 'Proyectando máscaras sobre la malla'],
                ['nombres', 'Nombrando objetos']);
  }
  if (opciones.completado) etapas.push(['completado', 'Completando la geometría oculta']);
  if (opciones.splats) etapas.push(['splats', 'Entrenando Gaussian Splatting']);
  etapas.push(['empaquetado', 'Preparando resultados']);
  return etapas;
}

const cola = [];
let ocupado = false;

export function encolar(id) {
  if (!cola.includes(id)) cola.push(id);
  setImmediate(siguiente);
}

/// Al arrancar, retoma los trabajos que quedaron a medias (como haría el despachador con Postgres).
export async function reanudarPendientes() {
  for (const e of await almacen.listar()) {
    if (e.estado === 'en_cola' || e.estado === 'procesando') encolar(e.id);
  }
}

async function siguiente() {
  if (ocupado || cola.length === 0) return;
  ocupado = true;
  const id = cola.shift();
  try {
    await procesar(id);
  } catch (error) {
    console.error(`[trabajador] ${id}:`, error);
    await almacen.actualizar(id, (e) => {
      e.estado = 'error';
      e.error = String(error.message ?? error);
    }).catch(() => {});
  } finally {
    ocupado = false;
    setImmediate(siguiente);
  }
}

const esperar = (ms) => new Promise((r) => setTimeout(r, ms));

async function procesar(id) {
  const escaneo = await almacen.leer(id);
  const etapas = etapasPara(escaneo.opciones);
  const inicio = Date.now();
  const tiempos = {};

  for (const [i, [clave, texto]] of etapas.entries()) {
    const t0 = Date.now();
    await almacen.actualizar(id, (e) => {
      e.estado = 'procesando';
      e.etapa = texto;
      e.progreso = i / etapas.length;
    });
    if (clave === 'empaquetado') await empaquetar(id, escaneo);
    else await esperar(PAUSA_MS);
    tiempos[clave] = (Date.now() - t0) / 1000;
  }

  const resultados = await almacen.archivosEn(path.join(almacen.carpeta(id), 'salida'));
  await almacen.actualizar(id, (e) => {
    e.estado = 'listo';
    e.etapa = null;
    e.progreso = 1;
    e.resultados = resultados;
  });
  console.log(`[trabajador] ${id} listo en ${((Date.now() - inicio) / 1000).toFixed(1)} s`, tiempos);
}

async function empaquetar(id, escaneo) {
  const entrada = path.join(almacen.carpeta(id), 'entrada');
  const salida = path.join(almacen.carpeta(id), 'salida');
  await mkdir(path.join(salida, 'objetos'), { recursive: true });

  const subidos = await almacen.archivosEn(entrada);
  for (const ruta of subidos) {
    if (ruta.startsWith('objetos/') || ruta === 'textura.jpg') {
      await mkdir(path.dirname(path.join(salida, ruta)), { recursive: true });
      await copyFile(path.join(entrada, ruta), path.join(salida, ruta));
    }
  }

  let escena = { version: 1, unidades: 'metros', ejeVertical: 'Y', objetos: [] };
  if (subidos.includes('escena.json')) {
    escena = JSON.parse(await readFile(path.join(entrada, 'escena.json'), 'utf8'));
  } else if (subidos.includes('malla_color.ply')) {
    // Escaneo sin separar: la malla completa como un único objeto, igual que hace el visor del iPhone.
    await copyFile(path.join(entrada, 'malla_color.ply'), path.join(salida, 'objetos', 'Escaneo.ply'));
    escena.objetos.push({ nombre: 'Escaneo completo', categoria: 'objeto', estructura: false,
                          centro: [0, 0, 0], dimensiones: [0, 0, 0], rotacionY: 0,
                          vertices: 0, triangulos: 0, archivo: 'objetos/Escaneo.ply' });
  }

  const v2 = {
    ...escena,
    version: 2,
    procesado: {
      servicio: 'escaner-lidar-ia',
      fecha: new Date().toISOString(),
      simulado: true,
      modelos: [],
    },
    objetos: escena.objetos.map((o) => ({ ...o, fuente: 'iphone', fraccionInferida: 0 })),
  };
  await writeFile(path.join(salida, 'escena.json'), JSON.stringify(v2, null, 2));

  await writeFile(path.join(salida, 'informe.json'), JSON.stringify({
    simulado: true,
    aviso: 'Resultado del trabajador simulado: no se ha ejecutado ningún modelo de IA.',
    opciones: escaneo.opciones,
    etapas: etapasPara(escaneo.opciones).map(([clave]) => clave),
    archivosEntrada: subidos.length,
  }, null, 2));
}
