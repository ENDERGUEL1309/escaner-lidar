// Servidor de refinado en la Wi-Fi de casa. Habla la misma API que servidor/ (la que usa CloudClient.swift), así que
// para el iPhone es «otro servidor»: crear escaneo → subir archivos → procesar → estado → descargar resultados.
// Solo responde con el token de datos/.token. Escaneos y resultados quedan en datos/ hasta que el iPhone los borra.
import { spawn } from 'node:child_process';
import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { networkInterfaces } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const AQUI = dirname(fileURLToPath(import.meta.url));

export function crearServidor({ datos = process.env.ESCANER_DATOS ?? join(AQUI, '..', 'datos'), token } = {}) {
  mkdirSync(datos, { recursive: true });
  const rutaToken = join(datos, '.token');
  const clave = token ?? (existsSync(rutaToken) ? readFileSync(rutaToken, 'utf8').trim() : randomBytes(18).toString('base64url'));
  if (!token) writeFileSync(rutaToken, clave);

  const escaneos = new Map();
  const cola = [];
  let ocupado = false;

  const carpetaDe = (id) => join(datos, id);
  const guardar = (e) => writeFileSync(join(carpetaDe(e.id), 'estado.json'), JSON.stringify(e));

  for (const id of readdirSync(datos)) {
    const ruta = join(carpetaDe(id), 'estado.json');
    if (!existsSync(ruta)) continue;
    const e = JSON.parse(readFileSync(ruta, 'utf8'));
    if (e.estado === 'procesando' || e.estado === 'en_cola') {
      Object.assign(e, { estado: 'error', error: 'El PC se reinició durante el refinado. Vuelve a enviarlo.' });
      guardar(e);
    }
    escaneos.set(id, e);
  }

  const rutaSegura = (r) => typeof r === 'string' && r.length < 200 && /^[\w.-]+(\/[\w.-]+)*$/.test(r) && !r.split('/').includes('..');

  function dentro(base, relativa) {
    const destino = resolve(base, relativa);
    if (!destino.startsWith(resolve(base) + sep)) throw new Error('ruta fuera del escaneo');
    return destino;
  }

  function resultados(e) {
    const dir = join(carpetaDe(e.id), 'refinado');
    return readdirSync(dir).filter((f) => statSync(join(dir, f)).isFile())
      .map((ruta) => ({ ruta, url: `/api/escaneos/${e.id}/resultados/${ruta}` }));
  }

  function vista(e) {
    return {
      id: e.id, nombre: e.nombre, estado: e.estado, progreso: e.progreso, etapa: e.etapa, error: e.error,
      resultados: e.estado === 'listo' ? resultados(e) : [],
    };
  }

  function siguiente() {
    if (ocupado || !cola.length) return;
    ocupado = true;
    const e = cola.shift();
    Object.assign(e, { estado: 'procesando', etapa: 'Empezando', progreso: 0 });
    guardar(e);
    const args = [join(AQUI, 'refinar.mjs'), carpetaDe(e.id), ...(e.opciones?.tipo === 'refinado_completo' ? ['completo'] : [])];
    const hijo = spawn(process.execPath, args, { windowsHide: true });
    let resto = '', errores = '';
    hijo.stdout.on('data', (d) => {
      const lineas = (resto + d).split('\n');
      resto = lineas.pop();
      for (const l of lineas) {
        try {
          const { progreso, etapa } = JSON.parse(l);
          Object.assign(e, { progreso, etapa });
        } catch { /* una línea que no es un aviso */ }
      }
    });
    hijo.stderr.on('data', (d) => { errores = (errores + d).slice(-2000); });
    let terminado = false;
    const terminar = (error) => {
      if (terminado) return;
      terminado = true;
      Object.assign(e, error ? { estado: 'error', error, etapa: null } : { estado: 'listo', progreso: 1, etapa: 'Listo', error: null });
      guardar(e);
      console.log(`${error ? 'ERROR' : 'LISTO'} ${e.nombre} (${e.id})${error ? `: ${error}` : ''}`);
      ocupado = false;
      siguiente();
    };
    hijo.on('error', (err) => terminar(`No se pudo arrancar el refinado: ${err.message}`));
    hijo.on('close', (c) => terminar(c === 0 ? null : `Falló en el PC: ${errores.trim().split('\n').slice(0, 2).join(' ').slice(0, 300)}`));
  }

  function leerJSON(req) {
    return new Promise((ok, mal) => {
      let t = '';
      req.on('data', (d) => { t += d; if (t.length > 8 << 20) { mal(new Error('manifiesto demasiado grande')); req.destroy(); } });
      req.on('end', () => { try { ok(JSON.parse(t || '{}')); } catch { mal(new Error('JSON no válido')); } });
      req.on('error', mal);
    });
  }

  function json(res, codigo, cuerpo) {
    if (res.headersSent) return;
    res.writeHead(codigo, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(cuerpo));
  }

  function subir(req, res, e, ruta) {
    const archivo = e.archivos.find((a) => a.ruta === ruta);
    if (!archivo) return json(res, 404, { error: 'Archivo fuera del manifiesto' });
    const destino = dentro(carpetaDe(e.id), ruta);
    mkdirSync(dirname(destino), { recursive: true });
    const temporal = `${destino}.parte`;
    const f = createWriteStream(temporal);
    const hash = createHash('sha256');
    let bytes = 0;
    req.on('data', (d) => {
      bytes += d.length;
      hash.update(d);
      if (bytes > archivo.bytes) {
        json(res, 413, { error: `${ruta} es mayor de lo anunciado` });
        req.destroy();
        f.destroy();
        rmSync(temporal, { force: true });
      }
    });
    req.pipe(f);
    f.on('finish', () => {
      const firma = hash.digest('hex');
      if (bytes !== archivo.bytes || firma !== archivo.sha256) {
        rmSync(temporal, { force: true });
        return json(res, 422, { error: `${ruta} llegó dañado: vuelve a enviarlo` });
      }
      renameSync(temporal, destino);
      archivo.subido = true;
      guardar(e);
      json(res, 200, { ok: true });
    });
    f.on('error', (err) => json(res, 500, { error: err.message }));
  }

  function autorizado(req) {
    const a = Buffer.from(req.headers.authorization ?? '');
    const b = Buffer.from(`Bearer ${clave}`);
    return a.length === b.length && timingSafeEqual(a, b);
  }

  const servidor = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://local');
      const p = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
      if (url.pathname === '/salud') return json(res, 200, { ok: true, servicio: 'escaner-lidar-pc' });
      if (!autorizado(req)) return json(res, 401, { error: 'Token no válido' });
      if (p[0] !== 'api' || p[1] !== 'escaneos') return json(res, 404, { error: 'No existe' });

      if (p.length === 2 && req.method === 'GET') return json(res, 200, { escaneos: [...escaneos.values()].map(vista) });
      if (p.length === 2 && req.method === 'POST') {
        const m = await leerJSON(req);
        const archivos = m.archivos;
        if (!Array.isArray(archivos) || !archivos.length) return json(res, 400, { error: 'Manifiesto sin archivos' });
        if (!archivos.every((a) => rutaSegura(a.ruta) && Number.isSafeInteger(a.bytes) && a.bytes >= 0 && /^[0-9a-f]{64}$/.test(a.sha256))) {
          return json(res, 400, { error: 'Manifiesto no válido' });
        }
        const id = randomUUID();
        const e = {
          id, nombre: String(m.nombre ?? 'Escaneo').slice(0, 120), opciones: m.opciones ?? {}, estado: 'subiendo',
          progreso: 0, etapa: null, error: null, archivos: archivos.map(({ ruta, bytes, sha256 }) => ({ ruta, bytes, sha256 })),
        };
        mkdirSync(carpetaDe(id), { recursive: true });
        guardar(e);
        escaneos.set(id, e);
        console.log(`Nuevo escaneo: ${e.nombre} (${archivos.length} archivos)`);
        return json(res, 201, {
          id,
          subidas: e.archivos.map((a) => ({ ruta: a.ruta, metodo: 'PUT', url: `/api/escaneos/${id}/archivos/${a.ruta}` })),
        });
      }

      const e = escaneos.get(p[2]);
      if (!e) return json(res, 404, { error: 'Escaneo no encontrado' });

      if (p.length === 3 && req.method === 'GET') return json(res, 200, vista(e));
      if (p.length === 3 && req.method === 'DELETE') {
        if (e.estado === 'procesando') return json(res, 409, { error: 'Se está refinando' });
        const i = cola.indexOf(e);
        if (i >= 0) cola.splice(i, 1);
        escaneos.delete(e.id);
        rmSync(carpetaDe(e.id), { recursive: true, force: true });
        return json(res, 200, { ok: true });
      }
      if (p.length === 4 && p[3] === 'procesar' && req.method === 'POST') {
        if (e.estado !== 'subiendo') return json(res, 409, { error: 'Ya se envió a procesar' });
        const faltan = e.archivos.filter((a) => !a.subido).length;
        if (faltan) return json(res, 409, { error: `Faltan ${faltan} archivos por subir` });
        if (!e.archivos.some((a) => a.ruta === 'malla_color.ply')) return json(res, 422, { error: 'El escaneo no trae malla_color.ply' });
        Object.assign(e, { estado: 'en_cola', etapa: 'Esperando turno en tu PC', progreso: 0 });
        guardar(e);
        cola.push(e);
        siguiente();
        return json(res, 202, { ok: true });
      }
      if (p[3] === 'archivos' && p.length > 4 && req.method === 'PUT') {
        const ruta = p.slice(4).join('/');
        if (!rutaSegura(ruta)) return json(res, 400, { error: 'Ruta no válida' });
        return subir(req, res, e, ruta);
      }
      if (p[3] === 'resultados' && p.length === 5 && req.method === 'GET') {
        if (!rutaSegura(p[4])) return json(res, 400, { error: 'Ruta no válida' });
        const archivo = dentro(join(carpetaDe(e.id), 'refinado'), p[4]);
        if (!existsSync(archivo)) return json(res, 404, { error: 'Sin ese resultado' });
        res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': statSync(archivo).size });
        return createReadStream(archivo).pipe(res);
      }
      json(res, 404, { error: 'No existe' });
    } catch (error) {
      json(res, 400, { error: error.message });
    }
  });
  // El límite por defecto de Node cortaría las subidas largas de fotos de 12 MP por Wi-Fi.
  servidor.requestTimeout = 0;
  return { servidor, token: clave };
}

export function direccionesLocales() {
  return Object.values(networkInterfaces()).flat()
    .filter((i) => i && i.family === 'IPv4' && !i.internal).map((i) => i.address);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const puerto = Number(process.env.PORT ?? 8787);
  const { servidor, token } = crearServidor();
  servidor.listen(puerto, '0.0.0.0', () => {
    console.log('\nServidor de refinado en marcha. En el iPhone, botón del PC en la pantalla principal:');
    for (const ip of direccionesLocales()) console.log(`  Dirección: http://${ip}:${puerto}`);
    console.log(`  Token:     ${token}\n`);
  });
}
