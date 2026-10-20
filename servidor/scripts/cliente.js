// Cliente mínimo de la API: hace lo mismo que haría la app del iPhone (manifiesto → subidas → procesar → estado).

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { archivosEn } from '../src/almacen.js';

/// Archivos de la carpeta de un escaneo que se suben (ver docs/ia-nube.md §3).
export function seSube(ruta) {
  return ['malla_color.ply', 'escena.json', 'textura.jpg', 'edicion.json', 'manifiesto.json'].includes(ruta) ||
    /^objetos\/[^/]+\.ply$/.test(ruta) ||
    /^fotos_clave\/[^/]+\.(jpg|jpeg|prof|json)$/i.test(ruta);
}

export async function subirCarpeta({ base, token, carpeta, opciones = {}, registro = console.log }) {
  const cabeceras = token ? { Authorization: `Bearer ${token}` } : {};
  const rutas = (await archivosEn(carpeta)).filter(seSube).filter((r) => r !== 'manifiesto.json');
  if (rutas.length === 0) throw new Error(`No hay archivos de escaneo en ${carpeta}`);

  const contenidos = new Map();
  const archivos = [];
  for (const ruta of rutas) {
    const datos = await readFile(path.join(carpeta, ruta));
    contenidos.set(ruta, datos);
    archivos.push({ ruta, bytes: datos.length, sha256: createHash('sha256').update(datos).digest('hex') });
  }

  const respuesta = await pedir(`${base}/api/escaneos`, {
    method: 'POST',
    headers: { ...cabeceras, 'Content-Type': 'application/json' },
    body: JSON.stringify({ version: 1, nombre: path.basename(carpeta), opciones, archivos }),
  });
  registro(`Escaneo ${respuesta.id}: subiendo ${archivos.length} archivos…`);

  for (const subida of respuesta.subidas) {
    await pedir(`${base}${subida.url}`, { method: subida.metodo, headers: cabeceras, body: contenidos.get(subida.ruta) });
  }
  await pedir(`${base}/api/escaneos/${respuesta.id}/procesar`, { method: 'POST', headers: cabeceras });

  let estado;
  let ultimaEtapa;
  do {
    await new Promise((r) => setTimeout(r, 400));
    estado = await pedir(`${base}/api/escaneos/${respuesta.id}`, { headers: cabeceras });
    if (estado.etapa && estado.etapa !== ultimaEtapa) {
      registro(`  ${Math.round(estado.progreso * 100)} % · ${estado.etapa}`);
      ultimaEtapa = estado.etapa;
    }
  } while (estado.estado === 'en_cola' || estado.estado === 'procesando');

  if (estado.estado !== 'listo') throw new Error(`El trabajo terminó en «${estado.estado}»: ${estado.error}`);
  return estado;
}

async function pedir(url, opciones) {
  const r = await fetch(url, opciones);
  const texto = await r.text();
  const datos = texto ? JSON.parse(texto) : null;
  if (!r.ok) throw new Error(`${opciones?.method ?? 'GET'} ${url} → ${r.status}: ${datos?.error ?? texto}`);
  return datos;
}
