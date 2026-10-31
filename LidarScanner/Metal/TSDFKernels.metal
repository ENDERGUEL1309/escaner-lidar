#include <metal_stdlib>
using namespace metal;

// Debe coincidir byte a byte con TSDFVolume.Voxel (8 bytes).
struct Voxel {
    half sdf;
    half weight;
    uchar r;
    uchar g;
    uchar b;
    uchar colorSamples;
};

// Debe coincidir con MetalFusion.IntegrationParams.
struct IntegrationParams {
    float4x4 worldToCamera;
    float fx, fy, cx, cy;
    float voxelSize, truncation, maxWeight, maxDepth;
    uint minConfidence, depthWidth, depthHeight, fullRange;
};

// Misma fusión que TSDFVolume.integrate en CPU: un hilo por vóxel de los bloques reservados.
kernel void integrateTSDFKernel(device Voxel *voxels [[buffer(0)]],
                                constant int4 *origins [[buffer(1)]],
                                constant IntegrationParams &p [[buffer(2)]],
                                texture2d<float, access::read> depthMap [[texture(0)]],
                                texture2d<uint, access::read> confidenceMap [[texture(1)]],
                                texture2d<float, access::sample> luma [[texture(2)]],
                                texture2d<float, access::sample> chroma [[texture(3)]],
                                uint3 id [[thread_position_in_grid]]) {
    constexpr sampler linearSampler(address::clamp_to_edge, filter::linear);

    const uint block = id.x / 512, idx = id.x % 512;
    const int3 base = origins[block].xyz * 8;
    const int3 local = int3(int(idx & 7), int((idx >> 3) & 7), int(idx >> 6));
    const float3 center = (float3(base + local) + 0.5) * p.voxelSize;

    const float4 c = p.worldToCamera * float4(center, 1);
    const float z = -c.z;
    if (z <= 0.05) return;
    const float u = p.fx * c.x / z + p.cx;
    const float v = p.fy * (-c.y) / z + p.cy;
    if (u < 1 || v < 1) return;
    const uint ui = uint(u), vi = uint(v);
    if (ui + 1 >= p.depthWidth || vi + 1 >= p.depthHeight) return;

    const float d = depthMap.read(uint2(ui, vi)).r;
    const uint confidence = confidenceMap.read(uint2(ui, vi)).r;
    if (!(d > 0.15 && d <= p.maxDepth) || confidence < p.minConfidence) return;

    // Píxeles volantes: en los bordes de los objetos el LiDAR devuelve distancias intermedias que no existen.
    const float jump = max(max(abs(d - depthMap.read(uint2(ui - 1, vi)).r), abs(d - depthMap.read(uint2(ui + 1, vi)).r)),
                           max(abs(d - depthMap.read(uint2(ui, vi - 1)).r), abs(d - depthMap.read(uint2(ui, vi + 1)).r)));
    if (jump >= 0.02 + 0.03 * d) return;

    const float sdf = d - z;
    if (sdf <= -p.truncation) return;   // muy detrás de la superficie: no observado

    const float w = (confidence >= 2 ? 1.0 : 0.4) * max(0.15, min(1.0, 1.0 / (d * d)));
    Voxel out = voxels[id.x];
    const float oldW = float(out.weight);
    const float total = oldW + w;
    out.sdf = half((float(out.sdf) * oldW + min(1.0, sdf / p.truncation) * w) / total);
    out.weight = half(min(total, p.maxWeight));

    if (abs(sdf) < p.truncation * 0.5) {
        const float2 uv = float2(u / float(p.depthWidth), v / float(p.depthHeight));
        float y = luma.sample(linearSampler, uv).r * 255;
        float2 cbcr = chroma.sample(linearSampler, uv).rg * 255 - 128;
        if (p.fullRange == 0) {
            y = (y - 16) * 1.164;
            cbcr *= 1.138;
        }
        const float3 color = float3(y + 1.402 * cbcr.y,
                                    y - 0.344136 * cbcr.x - 0.714136 * cbcr.y,
                                    y + 1.772 * cbcr.x);
        const float a = 1.0 / (float(out.colorSamples) + 1);
        const float3 old = float3(out.r, out.g, out.b);
        const float3 mixed = clamp(rint(old + (color - old) * a), float3(0.0), float3(255.0));
        out.r = uchar(mixed.x);
        out.g = uchar(mixed.y);
        out.b = uchar(mixed.z);
        if (out.colorSamples < 32) out.colorSamples += 1;
    }
    voxels[id.x] = out;
}

// Debe coincidir con MetalFusion.PartBox y MetalFusion.FreeSpaceParams.
struct PartBox {
    float4 minCorner;
    float4 maxCorner;
};

struct FreeSpaceParams {
    float4x4 cameraToLocal;
    float fx, fy, cx, cy;
    float margin;
    uint partCount, width, height;
};

// Prueba de espacio libre: el LiDAR vio más allá de la entrada a una pieza sólida. Si la lectura está antes de la
// pieza o sobre su cara, no hay contradicción; lo que no se midió (profundidad 0) no cuenta.
kernel void checkFreeSpaceViolationsKernel(texture2d<float, access::read> depthMap [[texture(0)]],
                                           constant PartBox *parts [[buffer(0)]],
                                           constant FreeSpaceParams &p [[buffer(1)]],
                                           device atomic_uint &violations [[buffer(2)]],
                                           uint2 id [[thread_position_in_grid]]) {
    const float d = depthMap.read(id).r;
    if (!(d > 0.15)) return;

    const float3 ray = float3((float(id.x) + 0.5 - p.cx) / p.fx, -(float(id.y) + 0.5 - p.cy) / p.fy, -1.0);
    const float3 origin = (p.cameraToLocal * float4(0, 0, 0, 1)).xyz;
    const float3 direction = (p.cameraToLocal * float4(ray, 0)).xyz;   // t = profundidad z en metros
    const float3 inverse = 1.0 / direction;

    for (uint i = 0; i < p.partCount; i++) {
        const float3 a = (parts[i].minCorner.xyz - origin) * inverse;
        const float3 b = (parts[i].maxCorner.xyz - origin) * inverse;
        const float3 near = min(a, b), far = max(a, b);
        const float enter = max(max(near.x, near.y), near.z);
        const float exit = min(min(far.x, far.y), far.z);
        if (enter > exit || enter < 0.05) continue;
        if (min(d, exit) - enter > p.margin) {
            atomic_fetch_add_explicit(&violations, 1u, memory_order_relaxed);
            return;
        }
    }
}
