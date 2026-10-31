// Prueba de extremo a extremo con un escaneo sintético: una habitación de 4 × 3 m con paredes ruidosas y una vista
// LiDAR desplazada (deriva), subida por HTTP como lo hace el iPhone.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { ajustarVertices, aplicar, icp, indiceEspacial, leerProf, nubeDeProfundidad } from '../src/icp.mjs';
import { alturaSuelo, corteHorizontal, muros } from '../src/plano.mjs';
import { escribirPLY, leerPLY } from '../src/ply.mjs';
import { crearServidor } from '../src/servidor.mjs';

const tmp = mkdtempSync(join(tmpdir(), 'lidar-'));
after(() => rmSync(tmp, { recursive: true, force: true }));

function aleatorio(semilla) { let s = semilla; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32; }

/** Caja interior: cada pared es una rejilla de vértices con ruido a lo largo de la normal. */
function habitacion(ruido = 0, paso = 0.04) {
  const azar = aleatorio(7);
  const P = [], N = [], C = [], I = [];
  const pared = (origen, u, v, nu, nv, n) => {
    const base = P.length / 3;
    for (let j = 0; j <= nv; j++) {
      for (let i = 0; i <= nu; i++) {
        const e = ruido * (azar() - 0.5) * 2;
        for (let k = 0; k < 3; k++) {
          P.push(origen[k] + u[k] * i * paso + v[k] * j * paso + n[k] * e);
          N.push(n[k]);
        }
        C.push(200, 190, 180);
      }
    }
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const a = base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
        I.push(a, b, d, a, d, c);
      }
    }
  };
  const nx = Math.round(4 / paso), nz = Math.round(3 / paso), ny = Math.round(2.5 / paso);
  pared([0, 0, 0], [1, 0, 0], [0, 0, 1], nx, nz, [0, 1, 0]);    // suelo
  pared([0, 2.5, 0], [1, 0, 0], [0, 0, 1], nx, nz, [0, -1, 0]); // techo
  pared([0, 0, 0], [1, 0, 0], [0, 1, 0], nx, ny, [0, 0, 1]);    // z = 0
  pared([0, 0, 3], [1, 0, 0], [0, 1, 0], nx, ny, [0, 0, -1]);   // z = 3
  pared([0, 0, 0], [0, 0, 1], [0, 1, 0], nz, ny, [1, 0, 0]);    // x = 0
  pared([4, 0, 0], [0, 0, 1], [0, 1, 0], nz, ny, [-1, 0, 0]);   // x = 4
  return {
    posiciones: Float32Array.from(P), normales: Float32Array.from(N),
    colores: Uint8Array.from(C), uvs: null, indices: Uint32Array.from(I),
  };
}

/** Mapa de profundidad exacto de la caja, visto desde `pose` (traza los 6 planos). */
function mapaDeCaja(pose, cam, ancho, alto) {
  const prof = new Float32Array(ancho * alto);
  for (let v = 0; v < alto; v++) {
    for (let u = 0; u < ancho; u++) {
      const dx = (u + 0.5 - cam.cx) / cam.fx, dy = -((v + 0.5 - cam.cy) / cam.fy), dz = -1;
      const w = [0, 1, 2].map((k) => pose[k] * dx + pose[4 + k] * dy + pose[8 + k] * dz);
      const o = [pose[12], pose[13], pose[14]];
      let t = Infinity;
      for (const [k, min, max] of [[0, 0, 4], [1, 0, 2.5], [2, 0, 3]]) {
        if (w[k] > 1e-6) t = Math.min(t, (max - o[k]) / w[k]);
        if (w[k] < -1e-6) t = Math.min(t, (min - o[k]) / w[k]);
      }
      prof[v * ancho + u] = t;
    }
  }
  return { ancho, alto, prof };
}

/** Cámara cerca del centro, girada hacia una esquina para ver tres planos a la vez. */
const POSE = (() => {
  const a = 40 * Math.PI / 180, b = -25 * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  // Columnas de Ry(a)·Rx(b).
  return [c, 0, -s, 0, s * sb, cb, c * sb, 0, s * cb, -sb, c * cb, 0, 2, 1.4, 1.5, 1];
})();
const CAM = { fx: 210, fy: 210, cx: 128, cy: 96 };

function medioFloat(v) {
  if (v === 0) return 0;
  const b = new Float32Array([v]);
  const i = new Uint32Array(b.buffer)[0];
  return ((i >> 16) & 0x8000) | (((i >> 23) & 0xff) - 112 << 10) | ((i >> 13) & 0x3ff);
}

function guardarProf(ruta, mapa) {
  const b = Buffer.alloc(8 + mapa.prof.length * 2);
  b.writeUInt32LE(mapa.ancho, 0); b.writeUInt32LE(mapa.alto, 4);
  mapa.prof.forEach((d, i) => b.writeUInt16LE(medioFloat(d), 8 + i * 2));
  writeFileSync(ruta, b);
}

test('PLY: ida y vuelta conserva la malla', () => {
  const ruta = join(tmp, 'ida.ply');
  const m = habitacion(0.002);
  escribirPLY(ruta, m);
  const l = leerPLY(ruta);
  assert.deepEqual(l.posiciones, m.posiciones);
  assert.deepEqual(Uint32Array.from(l.indices), m.indices);
  assert.deepEqual(l.colores, m.colores);
});

test('.prof: Float16 se lee con precisión de milímetros', () => {
  const ruta = join(tmp, 'p.prof');
  const mapa = mapaDeCaja(POSE, CAM, 256, 192);
  guardarProf(ruta, mapa);
  const l = leerProf(ruta);
  for (let i = 0; i < l.prof.length; i += 97) assert.ok(Math.abs(l.prof[i] - mapa.prof[i]) < 0.002);
});

test('ICP recupera una deriva de 4 cm y 1,5° hasta menos de 3 mm', () => {
  const malla = habitacion();
  const celdas = indiceEspacial(malla.posiciones);
  const mapa = mapaDeCaja(POSE, CAM, 256, 192);
  const buena = nubeDeProfundidad(mapa, CAM, POSE);
  // Deriva conocida: giro de 1,5° alrededor de Y + traslación.
  const g = 1.5 * Math.PI / 180, c = Math.cos(g), s = Math.sin(g);
  const deriva = { R: [c, 0, s, 0, 1, 0, -s, 0, c], t: [0.03, -0.02, 0.018] };
  const r = icp(aplicar(deriva, buena), malla, celdas);
  assert.ok(r, 'ICP encontró alineación');
  const corregida = aplicar(r, aplicar(deriva, buena));
  let max = 0;
  for (let i = 0; i < buena.length; i++) max = Math.max(max, Math.abs(corregida[i] - buena[i]));
  assert.ok(max < 0.003, `error máximo ${(max * 1000).toFixed(2)} mm`);
});

function errorSuelo(m) {
  let suma = 0, n = 0;
  for (let i = 0; i < m.posiciones.length; i += 3) {
    if (m.normales[i + 1] > 0.9 && m.posiciones[i + 1] < 0.1) { suma += m.posiciones[i + 1] ** 2; n++; }
  }
  return Math.sqrt(suma / n);
}

test('el ajuste de vértices reduce el ruido de la malla', () => {
  const malla = habitacion(0.008);
  const antes = errorSuelo(malla);
  const celdas = indiceEspacial(malla.posiciones);
  const nube = [];
  const azar = aleatorio(3);
  for (let i = 0; i < 400000; i++) nube.push(azar() * 4, (azar() - 0.5) * 0.001, azar() * 3);
  assert.ok(ajustarVertices(malla, [Float32Array.from(nube)], celdas) > 1000);
  const despues = errorSuelo(malla);
  assert.ok(despues < antes * 0.4, `${(antes * 1000).toFixed(2)} → ${(despues * 1000).toFixed(2)} mm`);
});

test('el plano recupera las cuatro paredes con sus medidas, aunque la casa esté girada', () => {
  const malla = habitacion(0.003);
  const g = 23 * Math.PI / 180, c = Math.cos(g), s = Math.sin(g);
  const p = malla.posiciones;
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i], z = p[i + 2];
    p[i] = c * x - s * z; p[i + 2] = s * x + c * z;
  }
  const suelo = alturaSuelo(p);
  assert.ok(Math.abs(suelo) < 0.01);
  const { lineas } = muros(corteHorizontal(malla, suelo + 1));
  const largos = lineas.map(([a, b, d, e]) => Math.hypot(d - a, e - b)).sort((a, b) => b - a);
  assert.equal(largos.length, 4, `tramos: ${largos}`);
  assert.ok(Math.abs(largos[0] - 4) < 0.03 && Math.abs(largos[1] - 4) < 0.03, String(largos));
  assert.ok(Math.abs(largos[2] - 3) < 0.03 && Math.abs(largos[3] - 3) < 0.03, String(largos));
});

test('API: subida, refinado y descarga como lo hace el iPhone', async () => {
  const { servidor, token } = crearServidor({ datos: join(tmp, 'datos'), token: 'prueba' });
  await new Promise((r) => servidor.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${servidor.address().port}`;
  const cab = { Authorization: `Bearer ${token}` };
  try {
    assert.equal((await fetch(`${base}/api/escaneos`)).status, 401);

    const origen = join(tmp, 'origen');
    mkdirSync(join(origen, 'fotos_clave'), { recursive: true });
    escribirPLY(join(origen, 'malla_color.ply'), habitacion(0.004));
    guardarProf(join(origen, 'fotos_clave', '000.prof'), mapaDeCaja(POSE, CAM, 256, 192));
    writeFileSync(join(origen, 'fotos_clave', 'camaras.json'), JSON.stringify({
      version: 1, fotos: [{ archivo: '000.jpg', cameraToWorld: POSE, profundidad: { archivo: '000.prof', ancho: 256, alto: 192, ...CAM } }],
    }));
    writeFileSync(join(origen, 'escena.json'), JSON.stringify({ objetos: [
      { nombre: 'Mesa 1', estructura: false, centro: [2, 0.4, 1.5], dimensiones: [1.2, 0.8, 0.6], rotacionY: 30 },
    ] }));

    const rutas = ['malla_color.ply', 'escena.json', 'fotos_clave/000.prof', 'fotos_clave/camaras.json'];
    const archivos = rutas.map((ruta) => {
      const d = readFileSync(join(origen, ruta));
      return { ruta, bytes: d.length, sha256: createHash('sha256').update(d).digest('hex') };
    });

    const malo = await fetch(`${base}/api/escaneos`, { method: 'POST', headers: cab, body: JSON.stringify({ archivos: [{ ruta: '../x', bytes: 1, sha256: 'a'.repeat(64) }] }) });
    assert.equal(malo.status, 400, 'rechaza rutas que salen de la carpeta');

    const creado = await (await fetch(`${base}/api/escaneos`, { method: 'POST', headers: cab, body: JSON.stringify({ nombre: 'Prueba', opciones: { tipo: 'refinado' }, archivos }) })).json();
    assert.equal((await fetch(`${base}/api/escaneos/${creado.id}/procesar`, { method: 'POST', headers: cab })).status, 409, 'no procesa sin archivos');

    const danado = await fetch(base + creado.subidas[1].url, { method: 'PUT', headers: cab, body: Buffer.alloc(archivos[1].bytes) });
    assert.equal(danado.status, 422, 'detecta un archivo dañado');

    for (const s of creado.subidas) {
      const r = await fetch(base + s.url, { method: s.metodo, headers: cab, body: readFileSync(join(origen, s.ruta)) });
      assert.equal(r.status, 200);
    }
    assert.equal((await fetch(`${base}/api/escaneos/${creado.id}/procesar`, { method: 'POST', headers: cab })).status, 202);

    let estado;
    for (let i = 0; i < 600; i++) {
      estado = await (await fetch(`${base}/api/escaneos/${creado.id}`, { headers: cab })).json();
      if (estado.estado === 'listo' || estado.estado === 'error') break;
      await new Promise((r) => setTimeout(r, 200));
    }
    assert.equal(estado.estado, 'listo', estado.error ?? '');
    const nombres = estado.resultados.map((r) => r.ruta).sort();
    assert.deepEqual(nombres, ['informe.json', 'malla_alta.ply', 'modelo.glb', 'modelo.obj', 'modelo.usdz', 'plano.dxf']);

    const bajar = async (n) => Buffer.from(await (await fetch(base + estado.resultados.find((r) => r.ruta === n).url, { headers: cab })).arrayBuffer());
    const informe = JSON.parse(await bajar('informe.json'));
    assert.equal(informe.vistasAlineadas, 1);
    assert.ok(informe.residuoMedianoMm < 5, `residuo ${informe.residuoMedianoMm} mm`);

    const glb = await bajar('modelo.glb');
    assert.equal(glb.readUInt32LE(0), 0x46546c67);
    assert.equal(glb.readUInt32LE(8), glb.length);
    const gltf = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)));
    assert.equal(gltf.accessors[0].count, habitacion().posiciones.length / 3);

    const usdz = await bajar('modelo.usdz');
    assert.equal(usdz.readUInt32LE(0), 0x04034b50);
    const inicio = 30 + usdz.readUInt16LE(26) + usdz.readUInt16LE(28);
    assert.equal(inicio % 64, 0, 'datos del USDZ alineados a 64 bytes');
    assert.equal(usdz.subarray(inicio, inicio + 9).toString(), '#usda 1.0');

    const dxf = (await bajar('plano.dxf')).toString();
    assert.ok(dxf.includes('MUROS') && dxf.includes('Mesa 1') && dxf.trimEnd().endsWith('EOF'));
    assert.equal(dxf.match(/\nLINE\n8\nMUEBLES/g).length, 4);

    assert.equal((await fetch(`${base}/api/escaneos/${creado.id}`, { method: 'DELETE', headers: cab })).status, 200);
    assert.equal((await fetch(`${base}/api/escaneos/${creado.id}`, { headers: cab })).status, 404);
  } finally {
    servidor.close();
  }
});
