import CoreGraphics
import Foundation
import Vision
import simd

/// Identifica qué es cada objeto escaneado con la IA de visión de Apple (`VNClassifyImageRequest`,
/// más de 1000 categorías, se ejecuta en el propio iPhone, sin internet).
///
/// 1. Busca la foto clave donde el objeto se ve mejor: casi entero, sin oclusiones y lo más grande posible.
/// 2. Recorta esa zona de la foto.
/// 3. La clasifica y devuelve la etiqueta concreta más probable (descarta "interior", "madera", etc.).
enum ObjectRecognizer {
    struct Result {
        let label: String
        let confidence: Float
    }

    static func recognize(_ mesh: ColoredMesh, keyframes: [Keyframe]) -> Result? {
        guard !mesh.positions.isEmpty, !keyframes.isEmpty else { return nil }
        let step = max(1, mesh.positions.count / 1500)
        let samples = stride(from: 0, to: mesh.positions.count, by: step).map { mesh.positions[$0] }

        // 1. Mejor vista.
        var best: (keyframe: Keyframe, rect: CGRect, score: Float)?
        for kf in keyframes {
            let w = Float(kf.image.width), h = Float(kf.image.height)
            var minP = SIMD2<Float>(w, h), maxP = SIMD2<Float>(0, 0)
            var visible = 0
            for p in samples {
                guard let pr = kf.imageCamera.project(p), pr.u >= 0, pr.v >= 0, pr.u < w, pr.v < h else { continue }
                if let dp = kf.depthCamera.project(p), let sensed = kf.depth(atU: dp.u, v: dp.v),
                   dp.z > sensed + 0.05 + 0.02 * sensed { continue }   // tapado por otra cosa
                visible += 1
                minP = simd_min(minP, SIMD2(pr.u, pr.v))
                maxP = simd_max(maxP, SIMD2(pr.u, pr.v))
            }
            let fraction = Float(visible) / Float(samples.count)
            guard fraction > 0.6 else { continue }
            let size = maxP - minP
            let score = fraction * size.x * size.y
            if score > (best?.score ?? 0) {
                best = (kf, CGRect(x: CGFloat(minP.x), y: CGFloat(minP.y), width: CGFloat(size.x), height: CGFloat(size.y)), score)
            }
        }
        guard let best else { return nil }

        // 2. Recorte con margen (el contexto ayuda a la clasificación).
        let image = best.keyframe.image
        let rect = best.rect.insetBy(dx: -best.rect.width * 0.15, dy: -best.rect.height * 0.15)
            .intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            .integral
        guard rect.width >= 24, rect.height >= 24, let crop = cgImage(from: image, rect: rect) else { return nil }

        // 3. Clasificación. Las fotos de ARKit están en la orientación del sensor (apaisada):
        //    con el iPhone en vertical hay que girarlas 90°.
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: crop, orientation: .right)
        guard (try? handler.perform([request])) != nil, let observations = request.results else { return nil }

        for observation in observations.prefix(20) where observation.confidence >= 0.12 {
            let id = observation.identifier
            guard !ignored.contains(id) else { continue }
            return Result(label: spanishName(for: id), confidence: observation.confidence)
        }
        return nil
    }

    private static func cgImage(from image: RGBImage, rect: CGRect) -> CGImage? {
        let x0 = Int(rect.minX), y0 = Int(rect.minY), w = Int(rect.width), h = Int(rect.height)
        var bytes = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let src = ((y0 + y) * image.width + x0) * 3
            for i in 0..<(w * 3) { bytes[y * w * 3 + i] = image.pixels[src + i] }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(width: w, height: h,
                       bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true,
                       intent: .defaultIntent)
    }

    /// Etiquetas demasiado genéricas para nombrar un objeto.
    private static let ignored: Set<String> = [
        "structure", "indoor", "interior_room", "room", "wood_processed", "wood_natural", "textile",
        "material", "floor", "wall", "ceiling", "housewares", "consumer_electronics", "machine",
        "building", "architecture", "tile", "hardwood", "carpet_rug", "office", "living_room",
        "bedroom", "kitchen_room", "bathroom_room", "home_decor", "blue_sky", "sky", "night_sky",
        "outdoor", "land", "plant_material", "liquid", "water_body", "screenshot", "document",
        "art", "illustrations", "decoration", "paper", "plastic", "metal", "glass", "light",
        "shadow", "dark", "black", "white", "people", "adult", "child",
    ]

    private static let spanishNames: [String: String] = [
        "chair": "Silla", "armchair": "Sillón", "sofa": "Sofá", "couch": "Sofá", "stool": "Taburete",
        "bench": "Banco", "table": "Mesa", "desk": "Escritorio", "coffee_table": "Mesa baja",
        "bed": "Cama", "crib": "Cuna", "pillow": "Cojín", "cushion": "Cojín", "blanket": "Manta",
        "lamp": "Lámpara", "lampshade": "Lámpara", "chandelier": "Lámpara de techo", "candle": "Vela",
        "plant": "Planta", "houseplant": "Planta", "potted_plant": "Planta", "flower": "Flor",
        "flowerpot": "Maceta", "vase": "Jarrón", "tree": "Árbol",
        "television": "Televisor", "tv": "Televisor", "monitor": "Monitor", "computer_monitor": "Monitor",
        "computer": "Ordenador", "laptop": "Portátil", "keyboard": "Teclado", "computer_keyboard": "Teclado",
        "computer_mouse": "Ratón", "printer": "Impresora", "speaker": "Altavoz", "loudspeaker": "Altavoz",
        "headphones": "Auriculares", "phone": "Teléfono", "cellphone": "Móvil", "tablet": "Tableta",
        "camera": "Cámara", "video_game_console": "Consola",
        "bookshelf": "Estantería", "bookcase": "Estantería", "shelf": "Estante", "shelving": "Estantería",
        "cabinet": "Mueble", "cupboard": "Armario", "wardrobe": "Armario", "drawer": "Cajonera",
        "dresser": "Cómoda", "nightstand": "Mesilla", "furniture": "Mueble",
        "refrigerator": "Nevera", "fridge": "Nevera", "oven": "Horno", "stove": "Cocina",
        "microwave": "Microondas", "dishwasher": "Lavavajillas", "washing_machine": "Lavadora",
        "sink": "Fregadero", "toilet": "Inodoro", "bathtub": "Bañera", "shower": "Ducha",
        "kettle": "Hervidor", "coffee_maker": "Cafetera", "toaster": "Tostadora",
        "door": "Puerta", "window": "Ventana", "curtain": "Cortina", "blinds": "Persiana",
        "mirror": "Espejo", "clock": "Reloj", "painting": "Cuadro", "picture_frame": "Cuadro",
        "poster": "Póster", "rug": "Alfombra", "carpet": "Alfombra", "fan": "Ventilador",
        "radiator": "Radiador", "air_conditioner": "Aire acondicionado", "fireplace": "Chimenea",
        "stairs": "Escalera", "staircase": "Escalera", "ladder": "Escalera de mano",
        "box": "Caja", "cardboard_box": "Caja", "basket": "Cesta", "bag": "Bolsa", "backpack": "Mochila",
        "suitcase": "Maleta", "trash_can": "Papelera", "bucket": "Cubo",
        "bottle": "Botella", "cup": "Taza", "mug": "Taza", "glass_cup": "Vaso", "bowl": "Cuenco",
        "plate": "Plato", "book": "Libro", "books": "Libros", "toy": "Juguete", "teddy_bear": "Peluche",
        "ball": "Pelota", "guitar": "Guitarra", "piano": "Piano", "drum": "Batería",
        "shoes": "Zapatos", "footwear": "Calzado", "clothing": "Ropa", "hat": "Sombrero",
        "bicycle": "Bicicleta", "motorcycle": "Moto", "car": "Coche", "vehicle": "Vehículo",
        "dog": "Perro", "cat": "Gato", "person": "Persona", "sculpture": "Escultura", "statue": "Estatua",
        "tool": "Herramienta", "toolbox": "Caja de herramientas", "vacuum_cleaner": "Aspiradora",
    ]

    private static func spanishName(for identifier: String) -> String {
        if let name = spanishNames[identifier] { return name }
        let words = identifier.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}
