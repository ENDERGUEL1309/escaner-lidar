// Despachador a la GPU en Modal (GPU_PROVEEDOR=modal). Sustituye al trabajador simulado: en vez de procesar
// aquí, envía el trabajo a la función `lanzar` de trabajador-gpu/app.py, que arranca la GPU y vuelve enseguida.
// La GPU descarga la entrada y sube la salida con URLs firmadas (firmas.js) y avisa del progreso en
// POST /api/interno/trabajos/<id>/progreso con X-Firma. Instrucciones en trabajador-gpu/README.md.

import { rm } from 'node:fs/promises';
import path from 'node:path';
import * as almacen from './almacen.js';
import { firmar, textoEntrada, textoSalida } from './firmas.js';

const VALIDEZ_S = Number(process.env.VALIDEZ_URLS_S ?? 6 * 3600);   // la GPU puede tardar en arrancar

const config = () => ({
  modalUrl: process.env.MODAL_URL,
  urlPublica: (process.env.URL_PUBLICA ?? '').replace(/\/+$/, ''),
});

/// Lanza un error con lo que falta para poder despachar a Modal.
export function comprobarConfiguracion() {
  const { modalUrl, urlPublica } = config();
  const faltan = [];
  if (!process.env.SECRETO_INTERNO) faltan.push('SECRETO_INTERNO (el mismo que en el secreto «escaner-lidar» de Modal)');
  if (!modalUrl) faltan.push('MODAL_URL (la URL de `lanzar` que muestra `modal deploy app.py`)');
  if (!/^https?:\/\//.test(urlPublica)) faltan.push('URL_PUBLICA (dirección de este servidor accesible desde internet)');
  if (faltan.length) throw new Error(`GPU_PROVEEDOR=modal necesita:\n  - ${faltan.join('\n  - ')}`);
}

const codificar = (ruta) => ruta.split('/').map(encodeURIComponent).join('/');

export function trabajoPara(escaneo) {
  const { urlPublica } = config();
  const { id } = escaneo;
  const expira = Math.floor(Date.now() / 1000) + VALIDEZ_S;
  const base = `${urlPublica}/api/interno/trabajos/${id}`;
  return {
    id,
    emitido: Date.now(),
    opciones: escaneo.opciones,
    entradas: escaneo.archivos.map((a) => ({
      ruta: a.ruta,
      bytes: a.bytes,
      url: `${base}/entrada/${codificar(a.ruta)}?expira=${expira}&firma=${firmar(textoEntrada(id, a.ruta, expira))}`,
    })),
    // `{ruta}` lo sustituye el trabajador por la ruta de cada resultado.
    salida: { url: `${base}/salida/{ruta}?expira=${expira}&firma=${firmar(textoSalida(id, expira))}` },
    progreso: { url: `${base}/progreso` },
  };
}

async function despachar(id) {
  const escaneo = await almacen.leer(id);
  await rm(path.join(almacen.carpeta(id), 'salida'), { recursive: true, force: true });
  const cuerpo = JSON.stringify(trabajoPara(escaneo));
  const respuesta = await fetch(config().modalUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Firma': firmar(cuerpo) },
    body: cuerpo,
    signal: AbortSignal.timeout(90_000),       // arranque en frío del contenedor web de Modal
  });
  if (!respuesta.ok) throw new Error(`Modal respondió ${respuesta.status}: ${(await respuesta.text()).slice(0, 300)}`);
  const { llamada = null } = await respuesta.json().catch(() => ({}));
  await almacen.actualizar(id, (e) => {
    if (e.estado === 'en_cola') e.etapa = 'Esperando a la GPU';
    e.llamadaModal = llamada;
  });
  console.log(`[modal] ${id} enviado (llamada ${llamada})`);
}

export function encolar(id) {
  despachar(id).catch(async (error) => {
    console.error(`[modal] ${id}:`, error);
    await almacen.actualizar(id, (e) => {
      e.estado = 'error';
      e.etapa = null;
      e.error = `No se pudo enviar a la GPU: ${error.message ?? error}`;
    }).catch(() => {});
  });
}

/// Al arrancar, reenvía lo que se quedó en cola. Lo que ya está `procesando` sigue en Modal y avisará solo.
export async function reanudarPendientes() {
  for (const e of await almacen.listar()) {
    if (e.estado === 'en_cola') encolar(e.id);
  }
}
