// Sube la carpeta de un escaneo real (copiada del iPhone desde la app Archivos) y la procesa.
//   npm run subir -- "C:\ruta\Entorno_2026-10-20_18-04-11" [--completado] [--splats]

import path from 'node:path';
import { subirCarpeta } from './cliente.js';

const argumentos = process.argv.slice(2);
const carpeta = argumentos.find((a) => !a.startsWith('--'));
if (!carpeta) {
  console.error('Uso: npm run subir -- <carpeta del escaneo> [--completado] [--splats]');
  process.exit(1);
}

const base = process.env.SERVIDOR ?? `http://localhost:${process.env.PORT ?? 8080}`;
try {
  const estado = await subirCarpeta({
    base,
    token: process.env.API_TOKEN,
    carpeta: path.resolve(carpeta),
    opciones: { completado: argumentos.includes('--completado'), splats: argumentos.includes('--splats') },
  });
  console.log(`Listo. Abre el editor: ${base}${estado.editor}`);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
