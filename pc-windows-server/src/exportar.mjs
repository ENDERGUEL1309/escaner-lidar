// Exportadores sin dependencias: OBJ (+MTL), GLB y USDZ. Unidades: metros, Y hacia arriba (como ARKit).
// `malla.uvs` usa el origen abajo a la izquierda, como el PLY de la app.
import { createWriteStream, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { finished } from 'node:stream/promises';
import { crc32 } from 'node:zlib';

export async function escribirOBJ(ruta, malla, { mtl, textura } = {}) {
  const f = createWriteStream(ruta);
  const p = malla.posiciones, n = malla.normales, c = malla.colores, uv = malla.uvs, ind = malla.indices;
  const usarUV = !!(uv && textura);
  let b = `# Escaner LiDAR - refinado en el PC\n# Unidades: metros, eje Y hacia arriba\n${usarUV ? `mtllib ${mtl}\n` : ''}o Refinado\n`;
  const vaciar = async () => {
    if (b.length > 1 << 20) {
      if (!f.write(b)) await new Promise((r) => f.once('drain', r));
      b = '';
    }
  };
  for (let i = 0; i < p.length / 3; i++) {
    b += `v ${p[i * 3].toFixed(5)} ${p[i * 3 + 1].toFixed(5)} ${p[i * 3 + 2].toFixed(5)}`;
    b += c ? ` ${(c[i * 3] / 255).toFixed(4)} ${(c[i * 3 + 1] / 255).toFixed(4)} ${(c[i * 3 + 2] / 255).toFixed(4)}\n` : '\n';
    await vaciar();
  }
  if (usarUV) {
    b += 'usemtl escaneo\n';
    for (let i = 0; i < uv.length / 2; i++) { b += `vt ${uv[i * 2].toFixed(6)} ${uv[i * 2 + 1].toFixed(6)}\n`; await vaciar(); }
  }
  for (let i = 0; i < n.length / 3; i++) {
    b += `vn ${n[i * 3].toFixed(4)} ${n[i * 3 + 1].toFixed(4)} ${n[i * 3 + 2].toFixed(4)}\n`;
    await vaciar();
  }
  for (let t = 0; t < ind.length; t += 3) {
    const [x, y, z] = [ind[t] + 1, ind[t + 1] + 1, ind[t + 2] + 1];
    b += usarUV ? `f ${x}/${x}/${x} ${y}/${y}/${y} ${z}/${z}/${z}\n` : `f ${x}//${x} ${y}//${y} ${z}//${z}\n`;
    await vaciar();
  }
  f.end(b);
  await finished(f);
  if (usarUV) {
    writeFileSync(join(dirname(ruta), mtl), `newmtl escaneo\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd ${textura.nombre}\n`);
  }
}

export function escribirGLB(ruta, malla, { textura } = {}) {
  const nv = malla.posiciones.length / 3;
  const piezas = [];
  const vistas = [], accesores = [];
  let o = 0;
  const anadir = (buf, target) => {
    const bytes = Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength);
    vistas.push({ buffer: 0, byteOffset: o, byteLength: bytes.length, ...(target ? { target } : {}) });
    const relleno = (4 - bytes.length % 4) % 4;
    piezas.push(bytes, Buffer.alloc(relleno));
    o += bytes.length + relleno;
    return vistas.length - 1;
  };
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv * 3; i++) {
    min[i % 3] = Math.min(min[i % 3], malla.posiciones[i]);
    max[i % 3] = Math.max(max[i % 3], malla.posiciones[i]);
  }
  const atributos = {};
  const acc = (a) => { accesores.push(a); return accesores.length - 1; };
  atributos.POSITION = acc({ bufferView: anadir(malla.posiciones, 34962), componentType: 5126, count: nv, type: 'VEC3', min, max });
  atributos.NORMAL = acc({ bufferView: anadir(malla.normales, 34962), componentType: 5126, count: nv, type: 'VEC3' });
  const conTextura = !!(malla.uvs && textura);
  if (malla.colores && !conTextura) {
    const rgba = new Uint8Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      rgba[i * 4] = malla.colores[i * 3]; rgba[i * 4 + 1] = malla.colores[i * 3 + 1];
      rgba[i * 4 + 2] = malla.colores[i * 3 + 2]; rgba[i * 4 + 3] = 255;
    }
    atributos.COLOR_0 = acc({ bufferView: anadir(rgba, 34962), componentType: 5121, normalized: true, count: nv, type: 'VEC4' });
  }
  const material = { pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 0.9 }, doubleSided: true };
  const gltf = { asset: { version: '2.0', generator: 'Escaner LiDAR PC' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, name: 'Refinado' }] };
  if (conTextura) {
    const uv = new Float32Array(nv * 2);
    for (let i = 0; i < nv; i++) { uv[i * 2] = malla.uvs[i * 2]; uv[i * 2 + 1] = 1 - malla.uvs[i * 2 + 1]; } // glTF: origen arriba
    atributos.TEXCOORD_0 = acc({ bufferView: anadir(uv, 34962), componentType: 5126, count: nv, type: 'VEC2' });
    const img = readFileSync(textura.ruta);
    gltf.images = [{ bufferView: anadir(new Uint8Array(img), null), mimeType: 'image/jpeg' }];
    gltf.textures = [{ source: 0, sampler: 0 }];
    gltf.samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
    material.pbrMetallicRoughness.baseColorTexture = { index: 0 };
  }
  const indices = acc({ bufferView: anadir(Uint32Array.from(malla.indices), 34963), componentType: 5125, count: malla.indices.length, type: 'SCALAR' });
  Object.assign(gltf, {
    meshes: [{ primitives: [{ attributes: atributos, indices, material: 0 }] }],
    materials: [material], accessors: accesores, bufferViews: vistas, buffers: [{ byteLength: o }],
  });
  let json = Buffer.from(JSON.stringify(gltf));
  json = Buffer.concat([json, Buffer.alloc((4 - json.length % 4) % 4, 0x20)]);
  const bin = Buffer.concat(piezas);
  const cab = Buffer.alloc(20);
  cab.writeUInt32LE(0x46546c67, 0); cab.writeUInt32LE(2, 4); cab.writeUInt32LE(28 + json.length + bin.length, 8);
  cab.writeUInt32LE(json.length, 12); cab.writeUInt32LE(0x4e4f534a, 16);
  const cabBin = Buffer.alloc(8);
  cabBin.writeUInt32LE(bin.length, 0); cabBin.writeUInt32LE(0x004e4942, 4);
  writeFileSync(ruta, Buffer.concat([cab, json, cabBin, bin]));
}

function usda(malla, textura) {
  const p = malla.posiciones, n = malla.normales, c = malla.colores, uv = malla.uvs, ind = malla.indices;
  const trios = (a, f = 5) => {
    const s = [];
    for (let i = 0; i < a.length; i += 3) s.push(`(${a[i].toFixed(f)},${a[i + 1].toFixed(f)},${a[i + 2].toFixed(f)})`);
    return s.join(',');
  };
  let colores = '', st = '', material = '', enlace = '';
  if (uv && textura) {
    const s = [];
    for (let i = 0; i < uv.length; i += 2) s.push(`(${uv[i].toFixed(6)},${uv[i + 1].toFixed(6)})`);
    st = `        texCoord2f[] primvars:st = [${s.join(',')}] (interpolation = "vertex")\n`;
    enlace = '        rel material:binding = </Refinado/Material>\n';
    material = `
    def Material "Material"
    {
        token outputs:surface.connect = </Refinado/Material/Superficie.outputs:surface>
        def Shader "Superficie"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Refinado/Material/Textura.outputs:rgb>
            float inputs:roughness = 0.9
            token outputs:surface
        }
        def Shader "Lector"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
        def Shader "Textura"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @${textura.nombre}@
            float2 inputs:st.connect = </Refinado/Material/Lector.outputs:result>
            float3 outputs:rgb
        }
    }
`;
  } else if (c) {
    const s = [];
    for (let i = 0; i < c.length; i += 3) s.push(`(${(c[i] / 255).toFixed(3)},${(c[i + 1] / 255).toFixed(3)},${(c[i + 2] / 255).toFixed(3)})`);
    colores = `        color3f[] primvars:displayColor = [${s.join(',')}] (interpolation = "vertex")\n`;
  }
  return `#usda 1.0
(
    defaultPrim = "Refinado"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Refinado"
{
    def Mesh "Malla"
    {
        uniform bool doubleSided = 1
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [${new Array(ind.length / 3).fill(3).join(',')}]
        int[] faceVertexIndices = [${Array.prototype.join.call(ind, ',')}]
        point3f[] points = [${trios(p)}]
        normal3f[] normals = [${trios(n, 4)}] (interpolation = "vertex")
${colores}${st}${enlace}    }
${material}}
`;
}

/** USDZ: ZIP sin comprimir con cada archivo alineado a 64 bytes, como exige la especificación. */
export function escribirUSDZ(ruta, malla, { textura } = {}) {
  const archivos = [['modelo.usda', Buffer.from(usda(malla, textura))]];
  if (malla.uvs && textura) archivos.push([textura.nombre, readFileSync(textura.ruta)]);
  const partes = [], central = [];
  let o = 0;
  for (const [nombre, datos] of archivos) {
    const nom = Buffer.from(nombre);
    const crc = crc32(datos);
    const base = o + 30 + nom.length;
    const relleno = (64 - (base + 4) % 64) % 64;
    const extra = Buffer.alloc(4 + relleno);
    extra.writeUInt16LE(0x1986, 0); extra.writeUInt16LE(relleno, 2);
    const loc = Buffer.alloc(30);
    loc.writeUInt32LE(0x04034b50, 0); loc.writeUInt16LE(20, 4);
    loc.writeUInt32LE(crc, 14); loc.writeUInt32LE(datos.length, 18); loc.writeUInt32LE(datos.length, 22);
    loc.writeUInt16LE(nom.length, 26); loc.writeUInt16LE(extra.length, 28);
    const cen = Buffer.alloc(46);
    cen.writeUInt32LE(0x02014b50, 0); cen.writeUInt16LE(20, 4); cen.writeUInt16LE(20, 6);
    cen.writeUInt32LE(crc, 16); cen.writeUInt32LE(datos.length, 20); cen.writeUInt32LE(datos.length, 24);
    cen.writeUInt16LE(nom.length, 28); cen.writeUInt32LE(o, 42);
    central.push(cen, nom);
    partes.push(loc, nom, extra, datos);
    o += 30 + nom.length + extra.length + datos.length;
  }
  const cd = Buffer.concat(central);
  const fin = Buffer.alloc(22);
  fin.writeUInt32LE(0x06054b50, 0); fin.writeUInt16LE(archivos.length, 8); fin.writeUInt16LE(archivos.length, 10);
  fin.writeUInt32LE(cd.length, 12); fin.writeUInt32LE(o, 16);
  writeFileSync(ruta, Buffer.concat([...partes, cd, fin]));
}
