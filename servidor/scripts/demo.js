// Crea un escaneo sintético (habitación con mesa, silla y lámpara) con el mismo formato que la app,
// lo sube al servidor en marcha, espera al trabajador simulado y muestra el enlace al editor.
//   npm start        (en otra terminal)
//   npm run demo

import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { DIRECTORIO_DATOS } from '../src/almacen.js';
import { subirCarpeta } from './cliente.js';

/// Caja con 4 vértices por cara (normales planas), como las que salen del escáner tras aplanar.
function caja([cx, cy, cz], [ancho, alto, fondo], color, yaw = 0) {
  const [hx, hy, hz] = [ancho / 2, alto / 2, fondo / 2];
  const caras = [
    [[1, 0, 0], [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]]],
    [[-1, 0, 0], [[-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, -hz]]],
    [[0, 1, 0], [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]]],
    [[0, -1, 0], [[-hx, -hy, hz], [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz]]],
    [[0, 0, 1], [[hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz], [-hx, -hy, hz]]],
    [[0, 0, -1], [[-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz], [hx, -hy, -hz]]],
  ];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const girar = ([x, y, z]) => [c * x + s * z, y, -s * x + c * z];
  const malla = { posiciones: [], normales: [], colores: [], indices: [] };
  for (const [normal, esquinas] of caras) {
    const base = malla.posiciones.length;
    for (const p of esquinas) {
      const [x, y, z] = girar(p);
      malla.posiciones.push([x + cx, y + cy, z + cz]);
      malla.normales.push(girar(normal));
      // Un poco de variación de color para que se note el sombreado por vértice.
      const k = 0.85 + 0.15 * ((y + hy) / (2 * hy || 1));
      malla.colores.push(color.map((v) => Math.min(255, Math.round(v * k))));
    }
    malla.indices.push([base, base + 1, base + 2], [base, base + 2, base + 3]);
  }
  return malla;
}

function unir(...mallas) {
  const out = { posiciones: [], normales: [], colores: [], indices: [] };
  for (const m of mallas) {
    const base = out.posiciones.length;
    out.posiciones.push(...m.posiciones);
    out.normales.push(...m.normales);
    out.colores.push(...m.colores);
    out.indices.push(...m.indices.map((t) => t.map((i) => i + base)));
  }
  return out;
}

/// PLY binario con el mismo encabezado que `ModelExporter.writePLY` (sin coordenadas de textura).
function ply(malla) {
  const cabecera = 'ply\nformat binary_little_endian 1.0\ncomment Escaner LiDAR - demo\n' +
    `element vertex ${malla.posiciones.length}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property float nx\nproperty float ny\nproperty float nz\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    `element face ${malla.indices.length}\nproperty list uchar uint vertex_indices\nend_header\n`;
  const datos = Buffer.alloc(malla.posiciones.length * 27 + malla.indices.length * 13);
  let o = 0;
  malla.posiciones.forEach((p, i) => {
    for (const v of [...p, ...malla.normales[i]]) { datos.writeFloatLE(v, o); o += 4; }
    for (const v of malla.colores[i]) datos[o++] = v;
  });
  for (const t of malla.indices) {
    datos[o++] = 3;
    for (const v of t) { datos.writeUInt32LE(v, o); o += 4; }
  }
  return Buffer.concat([Buffer.from(cabecera, 'ascii'), datos]);
}

const objetos = [
  { nombre: 'Suelo', categoria: 'suelo', estructura: true, archivo: 'Suelo', malla: caja([0, -0.01, 0], [4, 0.02, 4], [150, 140, 130]), centro: [0, -0.01, 0], dim: [4, 0.02, 4] },
  { nombre: 'Paredes', categoria: 'pared', estructura: true, archivo: 'Paredes', malla: unir(caja([0, 1.25, -2], [4, 2.5, 0.05], [225, 220, 210]), caja([-2, 1.25, 0], [0.05, 2.5, 4], [215, 215, 205])), centro: [-1, 1.25, -1], dim: [4, 2.5, 4] },
  { nombre: 'Mesa 1', categoria: 'mesa', estructura: false, archivo: 'Mesa_1', malla: unir(caja([0.3, 0.73, -0.8], [1.2, 0.04, 0.7], [120, 80, 50], 0.3), caja([0.3, 0.36, -0.8], [0.08, 0.7, 0.5], [100, 65, 40], 0.3)), centro: [0.3, 0.375, -0.8], dim: [1.2, 0.75, 0.7], rot: 0.3, ia: ['escritorio', 0.62] },
  { nombre: 'Silla 1', categoria: 'asiento', estructura: false, archivo: 'Silla_1', malla: unir(caja([0.4, 0.23, 0.0], [0.45, 0.46, 0.45], [40, 70, 140]), caja([0.4, 0.7, 0.2], [0.45, 0.5, 0.05], [40, 70, 140])), centro: [0.4, 0.475, 0.1], dim: [0.45, 0.95, 0.45] },
  { nombre: 'Lámpara 1', categoria: 'objeto', estructura: false, archivo: 'Lampara_1', malla: unir(caja([-1.4, 0.8, -1.4], [0.04, 1.6, 0.04], [60, 60, 60]), caja([-1.4, 1.6, -1.4], [0.35, 0.25, 0.35], [240, 210, 120])), centro: [-1.4, 0.86, -1.4], dim: [0.35, 1.72, 0.35], ia: ['lámpara', 0.88] },
];

const carpeta = path.join(DIRECTORIO_DATOS, 'demo', 'Entorno_demo');
await rm(carpeta, { recursive: true, force: true });
await mkdir(path.join(carpeta, 'objetos'), { recursive: true });

for (const o of objetos) await writeFile(path.join(carpeta, 'objetos', `${o.archivo}.ply`), ply(o.malla));
await writeFile(path.join(carpeta, 'malla_color.ply'), ply(unir(...objetos.map((o) => o.malla))));
await writeFile(path.join(carpeta, 'escena.json'), JSON.stringify({
  version: 1,
  unidades: 'metros',
  ejeVertical: 'Y',
  objetos: objetos.map((o) => ({
    nombre: o.nombre,
    categoria: o.categoria,
    estructura: o.estructura,
    centro: o.centro,
    dimensiones: o.dim,
    rotacionY: ((o.rot ?? 0) * 180) / Math.PI,
    vertices: o.malla.posiciones.length,
    triangulos: o.malla.indices.length,
    archivo: `objetos/${o.archivo}.ply`,
    ...(o.ia ? { etiquetaIA: o.ia[0], confianzaIA: o.ia[1] } : {}),
  })),
}, null, 2));
console.log(`Escaneo de demostración creado en ${carpeta}`);

const base = process.env.SERVIDOR ?? `http://localhost:${process.env.PORT ?? 8080}`;
try {
  const estado = await subirCarpeta({ base, token: process.env.API_TOKEN, carpeta, opciones: { completado: true } });
  console.log(`Resultados: ${estado.resultados.map((r) => r.ruta).join(', ')}`);
  console.log(`Editor web: ${base}${estado.editor}`);
} catch (error) {
  console.error(error.cause?.code === 'ECONNREFUSED' ? `No hay servidor en ${base}: arráncalo con «npm start».` : error.message);
  process.exit(1);
}
