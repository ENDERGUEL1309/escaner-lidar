// URLs firmadas con HMAC (SECRETO_INTERNO) para que el trabajador con GPU descargue la entrada y suba la
// salida sin el token de la API. Hacen el papel de las URLs prefirmadas de R2/S3 (docs/ia-nube.md §2):
// cuando los archivos vayan a R2, el despachador dará URLs de R2 y el trabajador no cambiará.
//
//   GET /api/interno/trabajos/<id>/entrada/<ruta>?expira=<s>&firma=<hex>   un archivo concreto
//   PUT /api/interno/trabajos/<id>/salida/<ruta>?expira=<s>&firma=<hex>    cualquier ruta de la salida de ese trabajo

import { createHmac, timingSafeEqual } from 'node:crypto';

const secreto = () => process.env.SECRETO_INTERNO || null;

export function firmar(texto) {
  if (!secreto()) throw new Error('Falta SECRETO_INTERNO');
  return createHmac('sha256', secreto()).update(texto).digest('hex');
}

export const textoEntrada = (id, ruta, expira) => `GET\n${id}\nentrada\n${ruta}\n${expira}`;
export const textoSalida = (id, expira) => `PUT\n${id}\nsalida\n${expira}`;

/// ¿La URL lleva una firma válida y sin caducar para `texto(expira)`?
export function firmaValida(url, texto) {
  if (!secreto()) return false;
  const expira = Number(url.searchParams.get('expira'));
  if (!Number.isInteger(expira) || expira < Date.now() / 1000) return false;
  const dada = Buffer.from(url.searchParams.get('firma') ?? '', 'hex');
  const esperada = Buffer.from(firmar(texto(expira)), 'hex');
  return dada.length === esperada.length && timingSafeEqual(dada, esperada);
}
