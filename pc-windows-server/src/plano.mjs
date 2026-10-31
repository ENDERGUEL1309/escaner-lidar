// Plano en planta DXF: corta la malla a 1 m del suelo (la altura a la que se dibujan los muros), endereza los tramos
// a la orientación dominante de la casa y añade la caja de cada mueble de escena.json. Unidades: metros.
import { writeFileSync } from 'node:fs';

export function alturaSuelo(p) {
  const y = new Float32Array(p.length / 3);
  for (let i = 0; i < y.length; i++) y[i] = p[i * 3 + 1];
  y.sort();
  return y[Math.floor(y.length * 0.02)];
}

export function corteHorizontal(malla, h) {
  const p = malla.posiciones, ind = malla.indices;
  const s = [];
  for (let t = 0; t < ind.length; t += 3) {
    const puntos = [];
    for (let k = 0; k < 3; k++) {
      const a = ind[t + k] * 3, b = ind[t + (k + 1) % 3] * 3;
      const ya = p[a + 1] - h, yb = p[b + 1] - h;
      if ((ya < 0) === (yb < 0)) continue;
      const u = ya / (ya - yb);
      puntos.push(p[a] + (p[b] - p[a]) * u, p[a + 2] + (p[b + 2] - p[a + 2]) * u);
    }
    if (puntos.length === 4) s.push(puntos);
  }
  return s;
}

function anguloDominante(segmentos) {
  // Media circular con periodo de 90°: las paredes perpendiculares refuerzan la misma orientación.
  let sx = 0, sy = 0;
  for (const [x0, z0, x1, z1] of segmentos) {
    const l = Math.hypot(x1 - x0, z1 - z0);
    const a = Math.atan2(z1 - z0, x1 - x0) * 4;
    sx += l * Math.cos(a); sy += l * Math.sin(a);
  }
  return Math.atan2(sy, sx) / 4;
}

/** Tramos rectos: agrupa por línea (bandas de 6 cm) y une lo que está a menos de 12 cm. */
export function muros(segmentos, { toleranciaGrados = 12, banda = 0.06, hueco = 0.12, minimo = 0.2 } = {}) {
  if (!segmentos.length) return { lineas: [], giro: 0 };
  const giro = anguloDominante(segmentos);
  const c = Math.cos(-giro), s = Math.sin(-giro);
  const rot = (x, z) => [x * c - z * s, x * s + z * c];
  const lim = Math.tan(toleranciaGrados * Math.PI / 180);

  // Primera pasada: los ejes de los tramos largos fijan dónde hay pared de verdad.
  const crudos = [];
  for (const [x0, z0, x1, z1] of segmentos) {
    const [a0, b0] = rot(x0, z0), [a1, b1] = rot(x1, z1);
    const da = Math.abs(a1 - a0), db = Math.abs(b1 - b0);
    if (da < 1e-6 && db < 1e-6) continue;
    const horizontal = da >= db;
    if ((horizontal ? db / da : da / db) > lim) continue;
    crudos.push({
      horizontal,
      of: horizontal ? (b0 + b1) / 2 : (a0 + a1) / 2,
      ini: horizontal ? Math.min(a0, a1) : Math.min(b0, b1),
      fin: horizontal ? Math.max(a0, a1) : Math.max(b0, b1),
    });
  }
  crudos.sort((p, q) => (p.horizontal === q.horizontal ? p.of - q.of : p.horizontal ? -1 : 1));

  // Segunda pasada: como `crudos` está ordenado, cada grupo es una racha de desplazamientos a menos de `banda`
  // (con bandas fijas, el ruido partiría una misma pared justo en el límite entre dos bandas).
  const grupos = [];
  let g = null;
  for (const r of crudos) {
    if (!g || g.horizontal !== r.horizontal || r.of - g.ultimo > banda) {
      g = { horizontal: r.horizontal, of: 0, peso: 0, ultimo: r.of, tramos: [] };
      grupos.push(g);
    }
    g.ultimo = r.of;
    const largo = r.fin - r.ini;
    g.of += r.of * largo; g.peso += largo; g.tramos.push([r.ini, r.fin]);
  }

  const vuelta = (a, b) => [a * Math.cos(giro) - b * Math.sin(giro), a * Math.sin(giro) + b * Math.cos(giro)];
  const lineas = [];
  for (const grupo of grupos) {
    const of = grupo.of / grupo.peso;
    grupo.tramos.sort((p, q) => p[0] - q[0]);
    let [ini, fin] = grupo.tramos[0];
    const emitir = () => {
      if (fin - ini < minimo) return;
      const p = grupo.horizontal ? vuelta(ini, of) : vuelta(of, ini);
      const q = grupo.horizontal ? vuelta(fin, of) : vuelta(of, fin);
      lineas.push([...p, ...q]);
    };
    for (const [a, b] of grupo.tramos.slice(1)) {
      if (a - fin <= hueco) fin = Math.max(fin, b);
      else { emitir(); ini = a; fin = b; }
    }
    emitir();
  }
  return { lineas, giro };
}

const num = (v) => Number(v).toFixed(4);

function linea(capa, x0, y0, x1, y1) {
  return `0\nLINE\n8\n${capa}\n10\n${num(x0)}\n20\n${num(y0)}\n30\n0\n11\n${num(x1)}\n21\n${num(y1)}\n31\n0\n`;
}

/** DXF R12, el más compatible (AutoCAD, LibreCAD, FreeCAD…). En planta: X = x de ARKit, Y = −z de ARKit. */
export function escribirDXF(ruta, { lineas = [], objetos = [] }) {
  const entidades = [];
  for (const [x0, z0, x1, z1] of lineas) entidades.push(linea('MUROS', x0, -z0, x1, -z1));
  for (const o of objetos) {
    if (o.estructura || !o.centro || !o.dimensiones) continue;
    const [cx, , cz] = o.centro;
    const [w, , d] = o.dimensiones;
    const g = (o.rotacionY ?? 0) * Math.PI / 180;
    const esquina = (a, b) => [cx + a * Math.cos(g) + b * Math.sin(g), -(cz - a * Math.sin(g) + b * Math.cos(g))];
    const c = [esquina(-w / 2, -d / 2), esquina(w / 2, -d / 2), esquina(w / 2, d / 2), esquina(-w / 2, d / 2)];
    for (let i = 0; i < 4; i++) entidades.push(linea('MUEBLES', ...c[i], ...c[(i + 1) % 4]));
    const texto = String(o.nombre ?? '').normalize('NFD').replace(/[^\x20-\x7e]/g, '');
    entidades.push(`0\nTEXT\n8\nTEXTOS\n10\n${num(cx)}\n20\n${num(-cz)}\n30\n0\n40\n0.08\n1\n${texto}\n`);
  }
  const capas = [['MUROS', 7], ['MUEBLES', 3], ['TEXTOS', 2]];
  const tablas = `0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n${capas.length}\n`
    + capas.map(([n, c]) => `0\nLAYER\n2\n${n}\n70\n0\n62\n${c}\n6\nCONTINUOUS\n`).join('')
    + '0\nENDTAB\n0\nENDSEC\n';
  writeFileSync(ruta, `0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC\n${tablas}`
    + `0\nSECTION\n2\nENTITIES\n${entidades.join('')}0\nENDSEC\n0\nEOF\n`);
}
