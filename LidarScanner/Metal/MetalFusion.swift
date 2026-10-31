import CoreVideo
import Foundation
import Metal
import simd

/// Enlace de los kernels de TSDFKernels.metal. `init?` falla si el dispositivo no puede crear los pipelines; quien lo
/// use debe recurrir entonces a la CPU. No es thread-safe: llámalo siempre desde la misma cola serie.
final class MetalFusion {
    struct IntegrationParams {
        var worldToCamera: simd_float4x4
        var fx, fy, cx, cy: Float
        var voxelSize, truncation, maxWeight, maxDepth: Float
        var minConfidence, depthWidth, depthHeight, fullRange: UInt32
    }

    struct PartBox {
        var minCorner: SIMD4<Float>
        var maxCorner: SIMD4<Float>
    }

    struct FreeSpaceParams {
        var cameraToLocal: simd_float4x4
        var fx, fy, cx, cy: Float
        var margin: Float
        var partCount, width, height: UInt32
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let integratePipeline: MTLComputePipelineState
    private let freeSpacePipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private var voxelBuffer: MTLBuffer?
    private var originBuffer: MTLBuffer?

    init?() {
        guard MemoryLayout<TSDFVolume.Voxel>.stride == 8,
              let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let integrate = library.makeFunction(name: "integrateTSDFKernel"),
              let freeSpace = library.makeFunction(name: "checkFreeSpaceViolationsKernel"),
              let integratePipeline = try? device.makeComputePipelineState(function: integrate),
              let freeSpacePipeline = try? device.makeComputePipelineState(function: freeSpace) else { return nil }
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess,
              let cache else { return nil }
        self.device = device
        self.queue = queue
        self.integratePipeline = integratePipeline
        self.freeSpacePipeline = freeSpacePipeline
        self.textureCache = cache
    }

    /// Fusiona un fotograma en los bloques ya reservados. Devuelve `false` si no se pudo (el volumen queda intacto).
    func integrate(into volume: TSDFVolume, blocks: [TSDFVolume.Block], camera: CameraModel,
                   depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer, capturedImage: CVPixelBuffer,
                   maxDepth: Float, minConfidence: UInt8) -> Bool {
        guard !blocks.isEmpty else { return true }
        guard let depth = texture(depthMap, format: .r32Float, plane: 0),
              let confidence = texture(confidenceMap, format: .r8Uint, plane: 0),
              let luma = texture(capturedImage, format: .r8Unorm, plane: 0),
              let chroma = texture(capturedImage, format: .rg8Unorm, plane: 1) else { return false }

        let voxelCount = blocks.count * 512
        let voxelBytes = voxelCount * MemoryLayout<TSDFVolume.Voxel>.stride
        let originBytes = blocks.count * MemoryLayout<SIMD4<Int32>>.stride
        if (voxelBuffer?.length ?? 0) < voxelBytes {
            voxelBuffer = device.makeBuffer(length: voxelBytes * 2, options: .storageModeShared)
        }
        if (originBuffer?.length ?? 0) < originBytes {
            originBuffer = device.makeBuffer(length: originBytes * 2, options: .storageModeShared)
        }
        guard let voxelBuffer, let originBuffer,
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else { return false }

        let voxels = voxelBuffer.contents().bindMemory(to: TSDFVolume.Voxel.self, capacity: voxelCount)
        let origins = originBuffer.contents().bindMemory(to: SIMD4<Int32>.self, capacity: blocks.count)
        for (i, block) in blocks.enumerated() {
            origins[i] = SIMD4(block.origin.x, block.origin.y, block.origin.z, 0)
            block.voxels.withUnsafeBufferPointer { source in
                (voxels + i * 512).update(from: source.baseAddress!, count: 512)
            }
        }

        var params = IntegrationParams(
            worldToCamera: camera.worldToCamera,
            fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy,
            voxelSize: volume.voxelSize, truncation: volume.truncation, maxWeight: TSDFVolume.maxWeight, maxDepth: maxDepth,
            minConfidence: UInt32(minConfidence),
            depthWidth: UInt32(depth.texture.width), depthHeight: UInt32(depth.texture.height),
            fullRange: CVPixelBufferGetPixelFormatType(capturedImage) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1 : 0)

        encoder.setComputePipelineState(integratePipeline)
        encoder.setBuffer(voxelBuffer, offset: 0, index: 0)
        encoder.setBuffer(originBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<IntegrationParams>.stride, index: 2)
        encoder.setTexture(depth.texture, index: 0)
        encoder.setTexture(confidence.texture, index: 1)
        encoder.setTexture(luma.texture, index: 2)
        encoder.setTexture(chroma.texture, index: 3)
        let width = min(integratePipeline.maxTotalThreadsPerThreadgroup, integratePipeline.threadExecutionWidth)
        encoder.dispatchThreads(MTLSize(width: voxelCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()   // los CVMetalTexture deben seguir vivos hasta aquí
        guard command.status == .completed else { return false }

        withExtendedLifetime((depth.holder, confidence.holder, luma.holder, chroma.holder)) {}
        for (i, block) in blocks.enumerated() {
            block.voxels.withUnsafeMutableBufferPointer { target in
                target.baseAddress!.update(from: voxels + i * 512, count: 512)
            }
        }
        return true
    }

    /// Píxeles de profundidad que atraviesan una pieza sólida; `nil` si no se pudo medir.
    func freeSpaceViolations(depth: [Float], width: Int, height: Int, camera: CameraModel, worldToLocal: simd_float4x4,
                             parts: [FurniturePart], margin: Float = 0.03) -> Int? {
        guard !parts.isEmpty, depth.count == width * height, width > 0 else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor),
              let counter = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared),
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else { return nil }
        depth.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: bytes.baseAddress!, bytesPerRow: width * MemoryLayout<Float>.size)
        }
        counter.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        var boxes = parts.map { part in
            let half = part.size / 2
            return PartBox(minCorner: SIMD4(part.center - half, 0), maxCorner: SIMD4(part.center + half, 0))
        }
        var params = FreeSpaceParams(cameraToLocal: worldToLocal * camera.cameraToWorld,
                                     fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy, margin: margin,
                                     partCount: UInt32(boxes.count), width: UInt32(width), height: UInt32(height))
        encoder.setComputePipelineState(freeSpacePipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&boxes, length: boxes.count * MemoryLayout<PartBox>.stride, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<FreeSpaceParams>.stride, index: 1)
        encoder.setBuffer(counter, offset: 0, index: 2)
        let side = max(1, Int(Double(freeSpacePipeline.maxTotalThreadsPerThreadgroup).squareRoot()))
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { return nil }
        return Int(counter.contents().load(as: UInt32.self))
    }

    /// Textura sin copia de un `CVPixelBuffer`. `holder` debe sobrevivir hasta que termine la GPU.
    private func texture(_ buffer: CVPixelBuffer, format: MTLPixelFormat, plane: Int) -> (texture: MTLTexture, holder: CVMetalTexture)? {
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, plane) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, plane) : CVPixelBufferGetHeight(buffer)
        var holder: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, buffer, nil, format,
                                                               width, height, plane, &holder)
        guard status == kCVReturnSuccess, let holder, let texture = CVMetalTextureGetTexture(holder) else { return nil }
        return (texture, holder)
    }
}
