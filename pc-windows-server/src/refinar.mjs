// Refinado de un escaneo. Se ejecuta en su propio proceso (`node src/refinar.mjs <carpeta> [completo]`) para que el
// servidor siga respondiendo: cada línea JSON de stdout es un aviso de progreso para la app.
//   Siempre: ICP denso de cada mapa LiDAR contra la malla, ajuste de vértices, OBJ/GLB/USDZ y plano DXF.
//   «completo»: además, la reconstrucción de trabajador-gpu/procesar_escaneo.py (ajuste de haces COLMAP + 2DGS).
import { spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ajustarVertices, aplicar, icp, indiceEspacial, leerProf, nubeDeProfundidad } from './icp.mjs';
import { escribirGLB, escribirOBJ, escribirUSDZ } from './exportar.mjs';
import { alturaSuelo, corteHorizontal, escribirDXF, muros } from './plano.mjs';
import { calcularNormales, escribirPLY, leerPLY } from './ply.mjs';

const AQUI = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(AQUI, '..', '..');

const avisar = (progreso, etapa) => console.log(JSON.stringify({ progreso, etapa }));

/** Python del entorno que deja preparado splat2d_local.bat, o el indicado en ESCANER_PYTHON. */
export function pythonCompleto() {
  const candidatos = [process.env.ESCANER_PYTHON, join(REPO, 'trabajador-gpu', '.venv-splat', 'Scripts', 'python.exe')];
  return candidatos.find((c) => c && existsSync(c)) ?? null;
}

function reconstruccionCompleta(carpeta, python) {
  return new Promise((ok, mal) => {
    const hijo = spawn(python, [join(REPO, 'trabajador-gpu', 'procesar_escaneo.py'), carpeta], { windowsHide: true });
    let ultimo = '';
    const leer = (d) => {
      for (const l of String(d).split(/\r?\n/)) {
        if (!l.trim()) continue;
        ultimo = l.trim();
        if (/^\d(-\d)?\/6/.test(ultimo)) avisar(0.7, ultimo.slice(0, 80));
      }
    };
    hijo.stdout.on('data', leer);
    hijo.stderr.on('data', leer);
    hijo.on('error', mal);
    hijo.on('close', (c) => (c === 0 ? ok() : mal(new Error(ultimo || `procesar_escaneo salió con ${c}`))));
  });
}

export function alinearVistas(carpeta, malla, celdas, alProgreso = () => {}) {
  const ruta = join(carpeta, 'fotos_clave', 'camaras.json');
  const fotos = existsSync(ruta) ? JSON.parse(readFileSync(ruta, 'utf8')).fotos ?? [] : [];
  const nubes = [], vistas = [];
  const conProf = fotos.filter((f) => f.profundidad);
  for (const [i, foto] of conProf.entries()) {
    const archivo = join(carpeta, 'fotos_clave', foto.profundidad.archivo);
    alProgreso(i / Math.max(1, conProf.length), i);
    if (!existsSync(archivo)) continue;
    const nube = nubeDeProfundidad(leerProf(archivo), foto.profundidad, foto.cameraToWorld);
    const r = icp(nube, malla, celdas);
    const medida = r ? { residuoMm: r.residuo * 1000, puntos: r.usados, traslacionMm: Math.hypot(...r.t) * 1000, giroGrados: r.giro * 180 / Math.PI } : {};
    vistas.push({ foto: foto.archivo, alineada: !!r, ...medida });
    if (r) nubes.push(aplicar(r, nube));
  }
  return { nubes, vistas };
}

export async function refinar(carpeta, { completo = false } = {}) {
  const salida = join(carpeta, 'refinado');
  mkdirSync(salida, { recursive: true });
  const informe = { fecha: new Date().toISOString(), avisos: [] };

  const lidar = leerPLY(join(carpeta, 'malla_color.ply'));
  let textura = null;
  const jpg = join(carpeta, 'textura.jpg');
  if (lidar.uvs && existsSync(jpg)) textura = { ruta: jpg, nombre: 'textura.jpg' };

  // Las medidas siempre salen de la malla LiDAR: es la que tiene la escala métrica de partida.
  avisar(0.1, 'Alineando los mapas LiDAR (ICP denso)');
  const celdas = indiceEspacial(lidar.posiciones);
  const { nubes, vistas } = alinearVistas(carpeta, lidar, celdas, (f, i) => {
    if (i % 3 === 0) avisar(0.1 + 0.45 * f, `Alineando vista ${i + 1}`);
  });
  informe.vistas = vistas;
  informe.vistasAlineadas = nubes.length;
  if (!nubes.length) informe.avisos.push('Sin mapas de profundidad alineados: no se ajustó la malla.');
  const residuos = vistas.filter((v) => v.alineada).map((v) => v.residuoMm).sort((a, b) => a - b);
  if (residuos.length) informe.residuoMedianoMm = residuos[residuos.length >> 1];

  avisar(0.6, 'Ajustando la superficie a las lecturas');
  informe.verticesAjustados = nubes.length ? ajustarVertices(lidar, nubes, celdas) : 0;
  if (informe.verticesAjustados) calcularNormales(lidar);

  avisar(0.68, 'Generando el plano DXF');
  const suelo = alturaSuelo(lidar.posiciones);
  const trazado = muros(corteHorizontal(lidar, suelo + 1));
  const rutaEscena = join(carpeta, 'escena.json');
  const objetos = existsSync(rutaEscena) ? JSON.parse(readFileSync(rutaEscena, 'utf8')).objetos ?? [] : [];
  escribirDXF(join(salida, 'plano.dxf'), { lineas: trazado.lineas, objetos });
  informe.muros = trazado.lineas.length;
  informe.giroPlanoGrados = trazado.giro * 180 / Math.PI;

  let malla = lidar;
  informe.reconstruccion = 'lidar';
  if (completo) {
    const python = pythonCompleto();
    if (!python) {
      informe.avisos.push('Reconstrucción completa no instalada (falta trabajador-gpu/.venv-splat): se entrega el refinado LiDAR.');
    } else {
      try {
        avisar(0.7, 'Fotogrametría y ajuste de haces (30–60 min)');
        await reconstruccionCompleta(carpeta, python);
        malla = leerPLY(join(carpeta, 'splat2d', 'malla_movil.ply'));
        textura = null;
        informe.reconstruccion = 'colmap_2dgs';
        const modelada = join(carpeta, 'splat2d', 'escena_modelada.ply');
        if (existsSync(modelada)) copyFileSync(modelada, join(salida, 'escena_modelada.ply'));
      } catch (e) {
        informe.avisos.push(`Falló la reconstrucción completa (${e.message.slice(0, 200)}): se entrega el refinado LiDAR.`);
      }
    }
  }

  avisar(0.85, 'Exportando OBJ, GLB y USDZ');
  if (textura) {
    copyFileSync(textura.ruta, join(salida, textura.nombre));
    malla.textura = textura.nombre;
  }
  escribirPLY(join(salida, 'malla_alta.ply'), malla);
  await escribirOBJ(join(salida, 'modelo.obj'), malla, { mtl: 'modelo.mtl', textura });
  escribirGLB(join(salida, 'modelo.glb'), malla, { textura });
  escribirUSDZ(join(salida, 'modelo.usdz'), malla, { textura });
  writeFileSync(join(salida, 'informe.json'), JSON.stringify(informe, null, 2));
  avisar(1, 'Listo');
  return informe;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  refinar(resolve(process.argv[2]), { completo: process.argv[3] === 'completo' })
    .catch((e) => { console.error(e.stack ?? e); process.exit(1); });
}
