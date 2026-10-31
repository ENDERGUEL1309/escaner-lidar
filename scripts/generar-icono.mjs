// Genera el icono de la app (PNG de 1024 px sin canal alfa, como exige iOS) sin dependencias:
//   node scripts/generar-icono.mjs
// Dibujo: una habitación en perspectiva de líneas cian barrida por el haz del LiDAR.
import { writeFileSync } from 'node:fs';
import { deflateSync, crc32 } from 'node:zlib';

const N = 1024;
const SS = 3; // sobremuestreo para suavizar los bordes
const W = N * SS;
const rgb = new Float32Array(W * W * 3);

for (let y = 0; y < W; y++) {
  for (let x = 0; x < W; x++) {
    const t = (x + y) / (2 * W);
    const i = (y * W + x) * 3;
    rgb[i] = 10 + 8 * t; rgb[i + 1] = 22 + 22 * t; rgb[i + 2] = 44 + 42 * t;
  }
}

function punto(px, py, r, c, a = 1) {
  const R = r * SS;
  const x0 = Math.max(0, Math.floor(px * SS - R)), x1 = Math.min(W - 1, Math.ceil(px * SS + R));
  const y0 = Math.max(0, Math.floor(py * SS - R)), y1 = Math.min(W - 1, Math.ceil(py * SS + R));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if ((x - px * SS) ** 2 + (y - py * SS) ** 2 > R * R) continue;
      const i = (y * W + x) * 3;
      for (let k = 0; k < 3; k++) rgb[i + k] += (c[k] - rgb[i + k]) * a;
    }
  }
}

function linea(a, b, r, c, alfa = 1) {
  const pasos = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2);
  for (let i = 0; i <= pasos; i++) {
    const t = i / pasos;
    punto(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, r, c, alfa);
  }
}

const CIAN = [64, 224, 255], BLANCO = [235, 250, 255], VERDE = [90, 255, 170];
// Caja isométrica: el modelo CAD limpio.
const c = [512, 545], s = 262, h = 0.58;
const sup = [[c[0], c[1] - s * h * 1.0 - s * 0.5], [c[0] + s, c[1] - s * h], [c[0], c[1] - s * h + s * 0.5], [c[0] - s, c[1] - s * h]];
const inf = sup.map(([x, y]) => [x, y + s * 1.05]);
// Haz de puntos LiDAR sobre la cara frontal izquierda.
for (let v = 0; v <= 12; v++) {
  for (let u = 0; u <= 12; u++) {
    const a = u / 12, b = v / 12;
    const x = sup[3][0] + (sup[2][0] - sup[3][0]) * a;
    const y = sup[3][1] + (sup[2][1] - sup[3][1]) * a + s * 1.05 * b;
    punto(x, y, 6.5, CIAN, 0.32 + 0.5 * (1 - b));
  }
}
for (let i = 0; i < 4; i++) {
  linea(sup[i], sup[(i + 1) % 4], 11, BLANCO);
  linea(sup[i], inf[i], 11, BLANCO);
}
linea(inf[3], inf[2], 11, BLANCO);
linea(inf[2], inf[1], 11, BLANCO);
// Marca de verificado.
linea([640, 590], [690, 640], 20, VERDE);
linea([690, 640], [785, 520], 20, VERDE);

const crudo = Buffer.alloc((N * 3 + 1) * N);
for (let y = 0; y < N; y++) {
  const o = y * (N * 3 + 1);
  crudo[o] = 0;
  for (let x = 0; x < N; x++) {
    for (let k = 0; k < 3; k++) {
      let suma = 0;
      for (let j = 0; j < SS; j++) for (let i = 0; i < SS; i++) suma += rgb[((y * SS + j) * W + x * SS + i) * 3 + k];
      crudo[o + 1 + x * 3 + k] = Math.round(suma / (SS * SS));
    }
  }
}

function trozo(tipo, datos) {
  const t = Buffer.concat([Buffer.from(tipo, 'latin1'), datos]);
  const largo = Buffer.alloc(4); largo.writeUInt32BE(datos.length);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(t));
  return Buffer.concat([largo, t, crc]);
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(N, 0); ihdr.writeUInt32BE(N, 4);
ihdr[8] = 8; ihdr[9] = 2; // 8 bits, RGB sin alfa
writeFileSync(new URL('../LidarScanner/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png', import.meta.url), Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  trozo('IHDR', ihdr), trozo('IDAT', deflateSync(crudo, { level: 9 })), trozo('IEND', Buffer.alloc(0)),
]));
