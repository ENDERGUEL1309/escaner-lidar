// Lectura y escritura de PLY: el binario del iPhone (x y z nx ny nz [s t] red green blue) y el del trabajador del
// PC (x y z r g b, sin normales). Las mallas viajan en arrays tipados: una habitación tiene millones de vértices.
import { readFileSync, writeFileSync } from 'node:fs';

const TIPOS = {
  char: ['getInt8', 1], uchar: ['getUint8', 1], int8: ['getInt8', 1], uint8: ['getUint8', 1],
  short: ['getInt16', 2], ushort: ['getUint16', 2], int16: ['getInt16', 2], uint16: ['getUint16', 2],
  int: ['getInt32', 4], uint: ['getUint32', 4], int32: ['getInt32', 4], uint32: ['getUint32', 4],
  float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
};

function cabecera(datos) {
  const fin = datos.indexOf('end_header\n');
  if (fin < 0) throw new Error('PLY sin end_header');
  const texto = datos.subarray(0, fin).toString('latin1');
  if (!texto.startsWith('ply') || !texto.includes('format binary_little_endian')) {
    throw new Error('Solo se admite PLY binario little-endian');
  }
  const elementos = [];
  for (const linea of texto.split('\n')) {
    const p = linea.trim().split(/\s+/);
    if (p[0] === 'element') elementos.push({ nombre: p[1], n: Number(p[2]), props: [] });
    else if (p[0] === 'property' && p[1] === 'list') {
      elementos.at(-1).props.push({ lista: true, cuenta: p[2], tipo: p[3], nombre: p[4] });
    } else if (p[0] === 'property') elementos.at(-1).props.push({ tipo: p[1], nombre: p[2] });
  }
  return { elementos, inicio: fin + 'end_header\n'.length };
}

export function leerPLY(ruta) {
  const datos = readFileSync(ruta);
  const vista = new DataView(datos.buffer, datos.byteOffset, datos.byteLength);
  const { elementos, inicio } = cabecera(datos);
  let o = inicio;
  const malla = { posiciones: null, normales: null, colores: null, uvs: null, indices: new Uint32Array(0) };

  for (const el of elementos) {
    if (el.nombre === 'vertex') {
      const nombres = new Map(el.props.map((p) => [p.nombre, p]));
      const tiene = (...n) => n.every((x) => nombres.has(x));
      malla.posiciones = new Float32Array(el.n * 3);
      if (tiene('nx', 'ny', 'nz')) malla.normales = new Float32Array(el.n * 3);
      if (tiene('s', 't')) malla.uvs = new Float32Array(el.n * 2);
      if (tiene('red', 'green', 'blue') || tiene('r', 'g', 'b')) malla.colores = new Uint8Array(el.n * 3);
      const destinos = {
        x: [malla.posiciones, 0, 3], y: [malla.posiciones, 1, 3], z: [malla.posiciones, 2, 3],
        nx: [malla.normales, 0, 3], ny: [malla.normales, 1, 3], nz: [malla.normales, 2, 3],
        s: [malla.uvs, 0, 2], t: [malla.uvs, 1, 2],
        red: [malla.colores, 0, 3], green: [malla.colores, 1, 3], blue: [malla.colores, 2, 3],
        r: [malla.colores, 0, 3], g: [malla.colores, 1, 3], b: [malla.colores, 2, 3],
      };
      const campos = el.props.map((p) => {
        const [lector, bytes] = TIPOS[p.tipo];
        const [arr, k, paso] = destinos[p.nombre] ?? [null, 0, 0];
        return { lector, bytes, arr, k, paso };
      });
      for (let i = 0; i < el.n; i++) {
        for (const c of campos) {
          if (c.arr) c.arr[i * c.paso + c.k] = vista[c.lector](o, true);
          o += c.bytes;
        }
      }
    } else if (el.nombre === 'face') {
      const salida = new Uint32Array(el.n * 3);
      let t = 0;
      const prop = el.props[0];
      const [lc, bc] = TIPOS[prop.cuenta];
      const [li, bi] = TIPOS[prop.tipo];
      for (let i = 0; i < el.n; i++) {
        const n = vista[lc](o, true);
        o += bc;
        if (n === 3) {
          salida[t++] = vista[li](o, true);
          salida[t++] = vista[li](o + bi, true);
          salida[t++] = vista[li](o + 2 * bi, true);
        }
        o += n * bi;
      }
      malla.indices = salida.subarray(0, t);
    } else {
      for (let i = 0; i < el.n; i++) {
        for (const p of el.props) {
          if (p.lista) {
            const [lc, bc] = TIPOS[p.cuenta];
            const n = vista[lc](o, true);
            o += bc + n * TIPOS[p.tipo][1];
          } else o += TIPOS[p.tipo][1];
        }
      }
    }
  }
  if (!malla.posiciones) throw new Error('PLY sin vértices');
  if (!malla.normales) calcularNormales(malla);
  return malla;
}

export function calcularNormales(malla) {
  const p = malla.posiciones, ind = malla.indices;
  const n = new Float32Array(p.length);
  for (let t = 0; t < ind.length; t += 3) {
    const a = ind[t] * 3, b = ind[t + 1] * 3, c = ind[t + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
    for (const i of [a, b, c]) { n[i] += x; n[i + 1] += y; n[i + 2] += z; }
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]);
    if (l > 0) { n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; } else n[i + 1] = 1;
  }
  malla.normales = n;
}

/** PLY con el mismo formato que escribe la app: el visor del iPhone (PLYReader.swift) lo abre tal cual. */
export function escribirPLY(ruta, malla) {
  const nv = malla.posiciones.length / 3, nt = malla.indices.length / 3;
  const uv = !!malla.uvs;
  const encabezado = 'ply\nformat binary_little_endian 1.0\ncomment Escaner LiDAR - PC\n'
    + (malla.textura ? `comment TextureFile ${malla.textura}\n` : '')
    + `element vertex ${nv}\nproperty float x\nproperty float y\nproperty float z\n`
    + 'property float nx\nproperty float ny\nproperty float nz\n'
    + (uv ? 'property float s\nproperty float t\n' : '')
    + 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
    + `element face ${nt}\nproperty list uchar uint vertex_indices\nend_header\n`;
  const paso = 27 + (uv ? 8 : 0);
  const cuerpo = Buffer.alloc(nv * paso + nt * 13);
  let o = 0;
  for (let i = 0; i < nv; i++) {
    for (let k = 0; k < 3; k++, o += 4) cuerpo.writeFloatLE(malla.posiciones[i * 3 + k], o);
    for (let k = 0; k < 3; k++, o += 4) cuerpo.writeFloatLE(malla.normales[i * 3 + k], o);
    if (uv) for (let k = 0; k < 2; k++, o += 4) cuerpo.writeFloatLE(malla.uvs[i * 2 + k], o);
    for (let k = 0; k < 3; k++) cuerpo[o++] = malla.colores ? malla.colores[i * 3 + k] : 180;
  }
  for (let t = 0; t < nt; t++) {
    cuerpo[o++] = 3;
    for (let k = 0; k < 3; k++, o += 4) cuerpo.writeUInt32LE(malla.indices[t * 3 + k], o);
  }
  writeFileSync(ruta, Buffer.concat([Buffer.from(encabezado, 'latin1'), cuerpo]));
}
