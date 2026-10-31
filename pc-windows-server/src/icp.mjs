// Alineación densa: cada mapa de profundidad LiDAR se ajusta a la malla con ICP punto-a-plano, y con esas lecturas
// ya alineadas cada vértice se desplaza por su normal hasta la mediana de lo medido (quita el ruido sin redondear
// aristas, porque solo se mueve en la dirección de la normal).
import { readFileSync } from 'node:fs';

const CELDA = 0.05;
const clave = (x, y, z) => `${Math.floor(x / CELDA)},${Math.floor(y / CELDA)},${Math.floor(z / CELDA)}`;

export function indiceEspacial(p) {
  const celdas = new Map();
  for (let i = 0; i < p.length / 3; i++) {
    const k = clave(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    const lista = celdas.get(k);
    if (lista) lista.push(i); else celdas.set(k, [i]);
  }
  return celdas;
}

function vecino(celdas, pos, x, y, z, maxD2) {
  const cx = Math.floor(x / CELDA), cy = Math.floor(y / CELDA), cz = Math.floor(z / CELDA);
  // Radios mayores que una celda necesitan mirar más celdas: si no, un punto a 8 cm nunca encuentra su vecino.
  const r = Math.ceil(Math.sqrt(maxD2) / CELDA);
  let mejor = -1, d2 = maxD2;
  for (let dz = -r; dz <= r; dz++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const lista = celdas.get(`${cx + dx},${cy + dy},${cz + dz}`);
        if (!lista) continue;
        for (const i of lista) {
          const ex = pos[i * 3] - x, ey = pos[i * 3 + 1] - y, ez = pos[i * 3 + 2] - z;
          const d = ex * ex + ey * ey + ez * ez;
          if (d < d2) { d2 = d; mejor = i; }
        }
      }
    }
  }
  return mejor;
}

/** Profundidad `.prof`: ancho y alto (uint32) + metros en Float16, 0 = sin dato. */
export function leerProf(ruta) {
  const b = readFileSync(ruta);
  const ancho = b.readUInt32LE(0), alto = b.readUInt32LE(4);
  const prof = new Float32Array(ancho * alto);
  for (let i = 0; i < prof.length; i++) prof[i] = media(b.readUInt16LE(8 + i * 2));
  return { ancho, alto, prof };
}

function media(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** Puntos del mapa de profundidad en el mundo. Misma convención que `CameraModel.unproject` de la app. */
export function nubeDeProfundidad(mapa, cam, pose, paso = 3) {
  const { ancho, alto, prof } = mapa;
  const salida = [];
  for (let v = 1; v < alto; v += paso) {
    for (let u = 1; u < ancho; u += paso) {
      const d = prof[v * ancho + u];
      if (!(d > 0.2 && d < 4)) continue;
      const x = (u + 0.5 - cam.cx) / cam.fx * d, y = -((v + 0.5 - cam.cy) / cam.fy * d), z = -d;
      salida.push(pose[0] * x + pose[4] * y + pose[8] * z + pose[12],
        pose[1] * x + pose[5] * y + pose[9] * z + pose[13],
        pose[2] * x + pose[6] * y + pose[10] * z + pose[14]);
    }
  }
  return Float32Array.from(salida);
}

function resolver6(A, b) {
  const n = 6;
  const M = A.map((f, i) => [...f, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let f = c + 1; f < n; f++) if (Math.abs(M[f][c]) > Math.abs(M[p][c])) p = f;
    if (Math.abs(M[p][c]) < 1e-9) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let f = 0; f < n; f++) {
      if (f === c) continue;
      const k = M[f][c] / M[c][c];
      for (let j = c; j <= n; j++) M[f][j] -= k * M[c][j];
    }
  }
  return M.map((f, i) => f[n] / f[i]);
}

/**
 * Devuelve la corrección rígida (R, t), que se aplica por la izquierda a la pose, para llevar `nube` sobre la malla;
 * `null` si no hay datos suficientes o si pide un salto impropio de la deriva de ARKit.
 */
export function icp(nube, malla, celdas, { iteraciones = 14, maxTraslacion = 0.1, maxGiro = 0.12 } = {}) {
  const pos = malla.posiciones, nor = malla.normales;
  const R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const t = [0, 0, 0];
  let residuo = Infinity, usados = 0;

  for (let it = 0; it < iteraciones; it++) {
    const maxD = it < 5 ? 0.1 : 0.05;
    const A = Array.from({ length: 6 }, () => new Array(6).fill(0));
    const b = new Array(6).fill(0);
    let n = 0, suma = 0;
    for (let i = 0; i < nube.length; i += 3) {
      const x0 = nube[i], y0 = nube[i + 1], z0 = nube[i + 2];
      const x = R[0] * x0 + R[1] * y0 + R[2] * z0 + t[0];
      const y = R[3] * x0 + R[4] * y0 + R[5] * z0 + t[1];
      const z = R[6] * x0 + R[7] * y0 + R[8] * z0 + t[2];
      const j = vecino(celdas, pos, x, y, z, maxD * maxD);
      if (j < 0) continue;
      const nx = nor[j * 3], ny = nor[j * 3 + 1], nz = nor[j * 3 + 2];
      const e = nx * (x - pos[j * 3]) + ny * (y - pos[j * 3 + 1]) + nz * (z - pos[j * 3 + 2]);
      const f = [y * nz - z * ny, z * nx - x * nz, x * ny - y * nx, nx, ny, nz];
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) A[r][c] += f[r] * f[c];
        b[r] -= f[r] * e;
      }
      n++; suma += e * e;
    }
    if (n < 150) return null;
    residuo = Math.sqrt(suma / n); usados = n;
    for (let d = 0; d < 6; d++) A[d][d] += 1e-6 * n;
    const dx = resolver6(A, b);
    if (!dx) return null;
    const [a, be, g, tx, ty, tz] = dx;
    const dR = [1, -g, be, g, 1, -a, -be, a, 1];
    const nR = new Array(9).fill(0);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        for (let k = 0; k < 3; k++) nR[r * 3 + c] += dR[r * 3 + k] * R[k * 3 + c];
      }
    }
    const tn = [
      dR[0] * t[0] + dR[1] * t[1] + dR[2] * t[2] + tx,
      dR[3] * t[0] + dR[4] * t[1] + dR[5] * t[2] + ty,
      dR[6] * t[0] + dR[7] * t[1] + dR[8] * t[2] + tz,
    ];
    ortonormalizar(nR);
    R.splice(0, 9, ...nR);
    t.splice(0, 3, ...tn);
    if (Math.hypot(tx, ty, tz) < 1e-5 && Math.hypot(a, be, g) < 1e-5) break;
  }
  const giro = Math.acos(Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2)));
  if (Math.hypot(...t) > maxTraslacion || giro > maxGiro) return null;
  return { R, t, residuo, usados, giro };
}

function ortonormalizar(m) {
  const norm = (o) => { const l = Math.hypot(m[o], m[o + 1], m[o + 2]); m[o] /= l; m[o + 1] /= l; m[o + 2] /= l; };
  norm(0);
  const p = m[0] * m[3] + m[1] * m[4] + m[2] * m[5];
  m[3] -= p * m[0]; m[4] -= p * m[1]; m[5] -= p * m[2];
  norm(3);
  m[6] = m[1] * m[5] - m[2] * m[4];
  m[7] = m[2] * m[3] - m[0] * m[5];
  m[8] = m[0] * m[4] - m[1] * m[3];
}

export function aplicar({ R, t }, nube) {
  const s = new Float32Array(nube.length);
  for (let i = 0; i < nube.length; i += 3) {
    const x = nube[i], y = nube[i + 1], z = nube[i + 2];
    s[i] = R[0] * x + R[1] * y + R[2] * z + t[0];
    s[i + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
    s[i + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
  }
  return s;
}

/** Mueve cada vértice por su normal hasta la mediana de las lecturas cercanas. Devuelve cuántos se movieron. */
export function ajustarVertices(malla, nubes, celdas, { radio = 0.03, maxMovimiento = 0.015, minLecturas = 4 } = {}) {
  const pos = malla.posiciones, nor = malla.normales;
  const lecturas = new Map();
  for (const nube of nubes) {
    for (let i = 0; i < nube.length; i += 3) {
      const j = vecino(celdas, pos, nube[i], nube[i + 1], nube[i + 2], radio * radio);
      if (j < 0) continue;
      const e = nor[j * 3] * (nube[i] - pos[j * 3]) + nor[j * 3 + 1] * (nube[i + 1] - pos[j * 3 + 1])
        + nor[j * 3 + 2] * (nube[i + 2] - pos[j * 3 + 2]);
      const l = lecturas.get(j);
      if (l) l.push(e); else lecturas.set(j, [e]);
    }
  }
  let movidos = 0;
  for (const [j, l] of lecturas) {
    if (l.length < minLecturas) continue;
    l.sort((a, b) => a - b);
    const d = Math.max(-maxMovimiento, Math.min(maxMovimiento, l[l.length >> 1]));
    for (let k = 0; k < 3; k++) pos[j * 3 + k] += nor[j * 3 + k] * d;
    movidos++;
  }
  return movidos;
}
