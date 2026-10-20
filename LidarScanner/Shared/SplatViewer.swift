import Observation
import SwiftUI
import WebKit
import simd

/// Vista fotorrealista (Gaussian Splatting) editable.
///
/// El `.splat` que devuelve la nube es una sola nube de gaussianas. Al abrir la vista, el iPhone reparte cada
/// gaussiana al objeto de la escena más cercano (los objetos que se estén usando: los del iPhone o los de la
/// nube), y cada objeto se muestra como una pieza independiente. Las ediciones se guardan en `edicion.json`,
/// el mismo archivo que el editor de malla: mover una silla aquí la mueve también allí, y al revés.
///
/// RealityKit no dibuja splats, así que se usa el visor Spark (three.js) en un WKWebView; los datos se sirven
/// desde la memoria con el esquema `escaneo://`.
@MainActor
@Observable
final class SplatSceneModel {
    struct Part: Identifiable {
        let id: String          // = nombre de objeto en edicion.json
        let title: String
        let isStructure: Bool
        let center: SIMD3<Float>
        let data: Data          // registros .splat de 32 bytes
        var count: Int { data.count / 32 }
    }

    private struct EditFile: Codable {
        var objetos: [String: ObjectEdit]
    }

    let folder: URL
    var parts: [Part] = []
    var edits: [String: ObjectEdit] = [:]
    var selectedID: String? { didSet { push() } }
    var status = "Preparando la vista fotorrealista…"
    var isReady = false
    private(set) var undoStack: [(id: String, edit: ObjectEdit)] = []

    @ObservationIgnored weak var webView: WKWebView?
    /// Dirección horizontal hacia la que mira la cámara del visor (para mover con las flechas en pantalla).
    @ObservationIgnored var viewForward = SIMD2<Float>(0, -1)

    init(folder: URL) {
        self.folder = folder
    }

    var selectedPart: Part? { parts.first { $0.id == selectedID } }
    var canUndo: Bool { !undoStack.isEmpty }
    var activeParts: [Part] { parts.filter { edits[$0.id]?.borrado != true } }
    var deletedParts: [Part] { parts.filter { edits[$0.id]?.borrado == true } }
    private var editsURL: URL { folder.appendingPathComponent("edicion.json") }

    static func file(in folder: URL) -> URL? {
        let url = folder.appendingPathComponent("ia/splats/escena.splat")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    // MARK: - Carga y reparto por objetos

    func load() async {
        guard parts.isEmpty, let file = Self.file(in: folder) else { return }
        if let data = try? Data(contentsOf: editsURL), let saved = try? JSONDecoder().decode(EditFile.self, from: data) {
            edits = saved.objetos
        }
        status = "Separando la vista fotorrealista por objetos…"
        let folder = self.folder
        parts = await Task.detached(priority: .userInitiated) { Self.split(splatFile: file, folder: folder) }.value
        status = "Cargando la vista fotorrealista…"
        pageReadyIfNeeded()
    }

    /// Asigna cada gaussiana al vértice de objeto más cercano (a menos de 8 cm); el resto va a «Fondo».
    nonisolated private static func split(splatFile: URL, folder: URL) -> [Part] {
        guard let splats = try? Data(contentsOf: splatFile, options: .mappedIfSafe) else { return [] }
        let count = splats.count / 32

        // Objetos de la escena con su malla (submuestreada para que la búsqueda sea rápida).
        struct Source { let id: String; let title: String; let structure: Bool; let center: SIMD3<Float> }
        var sources: [Source] = []
        let cell: Float = 0.08
        var grid: [Int64: [(SIMD3<Float>, Int32)]] = [:]
        func key(_ p: SIMD3<Float>) -> SIMD3<Int32> {
            let q = (p / cell).rounded(.down)
            return SIMD3(Int32(q.x), Int32(q.y), Int32(q.z))
        }
        func pack(_ c: SIMD3<Int32>) -> Int64 {
            let m: Int64 = 0x1FFFFF
            return (Int64(c.x) & m) | ((Int64(c.y) & m) << 21) | ((Int64(c.z) & m) << 42)
        }
        for object in ScanStorage.sceneObjects(in: folder) {
            guard let mesh = try? PLYReader.readMesh(from: folder.appendingPathComponent(object.archivo)),
                  !mesh.positions.isEmpty else { continue }
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude), hi = -lo
            for p in mesh.positions { lo = simd_min(lo, p); hi = simd_max(hi, p) }
            let index = Int32(sources.count)
            sources.append(Source(id: ModelExporter.objectName(object.nombre), title: object.nombre,
                                  structure: object.estructura, center: (lo + hi) / 2))
            let step = max(1, mesh.positions.count / 40_000)
            for i in stride(from: 0, to: mesh.positions.count, by: step) {
                let p = mesh.positions[i]
                grid[pack(key(p)), default: []].append((p, index))
            }
        }

        // Etiqueta de cada gaussiana (en paralelo).
        var labels = [Int32](repeating: -1, count: count)
        let maxDistance2 = cell * cell
        splats.withUnsafeBytes { raw in
            labels.withUnsafeMutableBufferPointer { out in
                let chunk = 8192
                DispatchQueue.concurrentPerform(iterations: (count + chunk - 1) / chunk) { c in
                    for i in (c * chunk)..<min(count, (c + 1) * chunk) {
                        let o = i * 32
                        let p = SIMD3<Float>(raw.loadUnaligned(fromByteOffset: o, as: Float.self),
                                             raw.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                                             raw.loadUnaligned(fromByteOffset: o + 8, as: Float.self))
                        let k = key(p)
                        var best = maxDistance2, label: Int32 = -1
                        for dz in -1...1 {
                            for dy in -1...1 {
                                for dx in -1...1 {
                                    guard let bucket = grid[pack(k &+ SIMD3(Int32(dx), Int32(dy), Int32(dz)))] else { continue }
                                    for (q, index) in bucket {
                                        let d = simd_distance_squared(p, q)
                                        if d < best { best = d; label = index }
                                    }
                                }
                            }
                        }
                        out[i] = label
                    }
                }
            }
        }

        // Un bloque de datos por objeto (+ «Fondo»), en el mismo formato .splat.
        var buckets = [Data](repeating: Data(), count: sources.count + 1)
        splats.withUnsafeBytes { raw in
            for i in 0..<count {
                let b = labels[i] >= 0 ? Int(labels[i]) : sources.count
                buckets[b].append(raw.baseAddress!.advanced(by: i * 32).assumingMemoryBound(to: UInt8.self), count: 32)
            }
        }
        var parts: [Part] = []
        for (i, source) in sources.enumerated() where !buckets[i].isEmpty {
            parts.append(Part(id: source.id, title: source.title, isStructure: source.structure,
                              center: source.center, data: buckets[i]))
        }
        if !buckets[sources.count].isEmpty {
            parts.append(Part(id: "Fondo_fotorrealista", title: "Resto de la escena", isStructure: true,
                              center: .zero, data: buckets[sources.count]))
        }
        return parts
    }

    // MARK: - Comunicación con el visor

    @ObservationIgnored private var pageLoaded = false

    func pageDidLoad() {
        pageLoaded = true
        pageReadyIfNeeded()
    }

    private func pageReadyIfNeeded() {
        guard pageLoaded, !parts.isEmpty else { return }
        let index = parts.enumerated().map { i, part in
            ["id": part.id, "centro": [part.center.x, part.center.y, part.center.z],
             "url": "escaneo://visor/obj/\(i).splat"] as [String: Any]
        }
        guard let json = try? JSONSerialization.data(withJSONObject: index),
              let text = String(data: json, encoding: .utf8) else { return }
        webView?.evaluateJavaScript("cargar(\(text), \(editsJSON()))")
    }

    func data(forPart index: Int) -> Data? {
        parts.indices.contains(index) ? parts[index].data : nil
    }

    private func editsJSON() -> String {
        let encoder = JSONEncoder()
        guard let data = try? encoder.encode(edits), let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    /// Envía las ediciones y la selección al visor.
    private func push() {
        let selection = selectedID.map { "\"\($0)\"" } ?? "null"
        webView?.evaluateJavaScript("aplicar(\(editsJSON()), \(selection))")
    }

    // MARK: - Edición (mismas reglas que el editor de malla)

    /// Mueve el objeto seleccionado en el plano del suelo, relativo a hacia dónde mira la cámara.
    func moveSelected(forward: Float, right: Float) {
        let f = simd_normalize(viewForward)
        let r = SIMD2<Float>(-f.y, f.x)
        let delta = f * forward + r * right
        change { $0.dx += delta.x; $0.dz += delta.y }
    }

    func liftSelected(_ meters: Float) { change { $0.dy += meters } }
    func rotateSelected(_ radians: Float) { change { $0.yaw += radians } }

    func deleteSelected() {
        change { $0.borrado = true }
        selectedID = nil
    }

    func restore(_ id: String) {
        guard let previous = edits[id] else { return }
        undoStack.append((id, previous))
        var edit = previous
        edit.borrado = false
        edits[id] = edit
        save()
        push()
    }

    func undo() {
        guard let last = undoStack.popLast() else { return }
        edits[last.id] = last.edit
        save()
        push()
    }

    private func change(_ apply: (inout ObjectEdit) -> Void) {
        guard let id = selectedID, let part = selectedPart, !part.isStructure else { return }
        let previous = edits[id] ?? ObjectEdit()
        undoStack.append((id, previous))
        var edit = previous
        apply(&edit)
        edits[id] = edit
        save()
        push()
    }

    private func save() {
        // Se conservan las ediciones de objetos que no están en esta vista (p. ej. de otra segmentación).
        var all = (try? Data(contentsOf: editsURL)).flatMap { try? JSONDecoder().decode(EditFile.self, from: $0) }?.objetos ?? [:]
        for (id, edit) in edits { all[id] = edit }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(EditFile(objetos: all)).write(to: editsURL, options: .atomic)
    }
}

struct SplatViewerView: View {
    @State private var model: SplatSceneModel
    @State private var showList = false

    init(folder: URL) {
        _model = State(initialValue: SplatSceneModel(folder: folder))
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            SplatWebView(model: model)
                .ignoresSafeArea()
                .background(Color.black)

            if let part = model.selectedPart {
                editPanel(part)
            } else {
                Text(model.isReady ? "Toca un objeto para editarlo · arrastra para girar · pellizca para acercar" : model.status)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
        .navigationTitle("Vista fotorrealista")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button { model.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(!model.canUndo)
                Button { showList = true } label: { Image(systemName: "list.bullet") }
                    .disabled(model.parts.count < 2)
            }
        }
        .sheet(isPresented: $showList) { objectList }
        .task { await model.load() }
    }

    private func editPanel(_ part: SplatSceneModel.Part) -> some View {
        VStack(spacing: 10) {
            HStack {
                Text(part.title).font(.headline)
                Spacer()
                Button { model.selectedID = nil } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
                    .foregroundStyle(.secondary)
            }
            if part.isStructure {
                Text("Las paredes, el suelo y el techo no se mueven.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                HStack(spacing: 14) {
                    // Cruceta: mueve 5 cm en la dirección de la pantalla.
                    VStack(spacing: 4) {
                        padButton("arrow.up") { model.moveSelected(forward: 0.05, right: 0) }
                        HStack(spacing: 4) {
                            padButton("arrow.left") { model.moveSelected(forward: 0, right: -0.05) }
                            padButton("arrow.down") { model.moveSelected(forward: -0.05, right: 0) }
                            padButton("arrow.right") { model.moveSelected(forward: 0, right: 0.05) }
                        }
                    }
                    VStack(spacing: 4) {
                        HStack(spacing: 4) {
                            padButton("rotate.left") { model.rotateSelected(.pi / 12) }
                            padButton("rotate.right") { model.rotateSelected(-.pi / 12) }
                        }
                        HStack(spacing: 4) {
                            padButton("arrow.up.to.line") { model.liftSelected(0.02) }
                            padButton("arrow.down.to.line") { model.liftSelected(-0.02) }
                        }
                    }
                    Spacer()
                    Button(role: .destructive) { model.deleteSelected() } label: {
                        Label("Borrar", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .padding()
    }

    private func padButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.body.weight(.semibold)).frame(width: 38, height: 34)
        }
        .buttonStyle(.bordered)
    }

    private var objectList: some View {
        NavigationStack {
            List {
                Section("En la escena") {
                    ForEach(model.activeParts) { part in
                        Button {
                            model.selectedID = part.id
                            showList = false
                        } label: {
                            HStack {
                                Text(part.title).foregroundStyle(.primary)
                                Spacer()
                                Text("\(part.count.formatted(.number.notation(.compactName))) splats")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                if !model.deletedParts.isEmpty {
                    Section("Borrados") {
                        ForEach(model.deletedParts) { part in
                            HStack {
                                Text(part.title).strikethrough().foregroundStyle(.secondary)
                                Spacer()
                                Button("Recuperar", systemImage: "arrow.uturn.backward") { model.restore(part.id) }
                                    .buttonStyle(.borderless)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Objetos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Listo") { showList = false } }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

private struct SplatWebView: UIViewRepresentable {
    let model: SplatSceneModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(context.coordinator, forURLScheme: "escaneo")
        config.userContentController.add(context.coordinator, name: "estado")
        let view = WKWebView(frame: .zero, configuration: config)
        view.isOpaque = false
        view.backgroundColor = .black
        view.scrollView.isScrollEnabled = false
        model.webView = view
        view.loadHTMLString(Self.html, baseURL: URL(string: "escaneo://visor/"))
        return view
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, WKURLSchemeHandler, WKScriptMessageHandler {
        let model: SplatSceneModel

        init(model: SplatSceneModel) { self.model = model }

        /// Sirve `escaneo://visor/obj/<n>.splat` desde la memoria.
        func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
            guard let url = task.request.url, url.pathComponents.contains("obj"),
                  let index = Int(url.deletingPathExtension().lastPathComponent),
                  let data = model.data(forPart: index) else {
                task.didFailWithError(URLError(.fileDoesNotExist))
                return
            }
            let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Type": "application/octet-stream",
                                                          "Content-Length": "\(data.count)",
                                                          "Access-Control-Allow-Origin": "*"])!
            task.didReceive(response)
            task.didReceive(data)
            task.didFinish()
        }

        func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let text = message.body as? String else { return }
            if text == "pagina" {
                model.pageDidLoad()
            } else if text == "listo" {
                model.isReady = true
            } else if text.hasPrefix("sel:") {
                let id = String(text.dropFirst(4))
                model.selectedID = id.isEmpty ? nil : id
            } else if text.hasPrefix("dir:") {
                let values = text.dropFirst(4).split(separator: ",").compactMap { Float($0) }
                if values.count == 2, values[0] != 0 || values[1] != 0 { model.viewForward = SIMD2(values[0], values[1]) }
            } else {
                model.status = text
            }
        }
    }

    /// Visor: Spark 2.4 sobre three.js 0.180. Cada objeto es un grupo con el pivote en su centro, igual que
    /// en el editor de malla: posición = centro + desplazamiento, giro alrededor del eje vertical.
    static let html = """
    <!doctype html>
    <html><head>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <style>html,body{margin:0;height:100%;background:#000;overflow:hidden;touch-action:none}</style>
    <script type="importmap">
    {"imports":{
      "three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
      "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/",
      "@sparkjsdev/spark":"https://cdn.jsdelivr.net/npm/@sparkjsdev/spark@2.4.0/dist/spark.module.js"
    }}
    </script>
    </head><body>
    <script type="module">
    const avisar = (t) => window.webkit.messageHandlers.estado.postMessage(t);
    try {
      const THREE = await import("three");
      const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
      const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200);
      const renderer = new THREE.WebGLRenderer({ antialias: false });
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.setSize(innerWidth, innerHeight);
      document.body.appendChild(renderer.domElement);
      scene.add(new SparkRenderer({ renderer }));
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      const grupos = {};

      window.cargar = async (indice, ediciones) => {
        avisar("Cargando " + indice.length + " objetos…");
        for (const p of indice) {
          const grupo = new THREE.Group();
          const malla = new SplatMesh({ url: p.url });
          malla.position.set(-p.centro[0], -p.centro[1], -p.centro[2]);
          grupo.add(malla);
          scene.add(grupo);
          grupos[p.id] = { grupo, malla, centro: p.centro };
        }
        await Promise.all(Object.values(grupos).map((g) => g.malla.initialized));
        const caja = new THREE.Box3();
        for (const g of Object.values(grupos)) caja.union(g.malla.getBoundingBox(true));
        const centro = caja.getCenter(new THREE.Vector3());
        const tam = caja.getSize(new THREE.Vector3()).length() || 4;
        camera.position.copy(centro).add(new THREE.Vector3(tam * 0.35, tam * 0.3, tam * 0.35));
        controls.target.copy(centro);
        controls.update();
        aplicar(ediciones, null);
        avisar("listo");
      };

      window.aplicar = (ediciones, seleccion) => {
        for (const [id, g] of Object.entries(grupos)) {
          const e = ediciones[id] || {};
          g.grupo.position.set(g.centro[0] + (e.dx || 0), g.centro[1] + (e.dy || 0), g.centro[2] + (e.dz || 0));
          g.grupo.rotation.set(0, e.yaw || 0, 0);
          g.grupo.visible = !e.borrado;
          g.malla.opacity = (!seleccion || seleccion === id) ? 1 : 0.3;
        }
      };

      // Toque (sin arrastrar) = seleccionar el objeto bajo el dedo.
      const raycaster = new THREE.Raycaster();
      let inicio = null;
      renderer.domElement.addEventListener("pointerdown", (e) => { inicio = [e.clientX, e.clientY]; });
      renderer.domElement.addEventListener("pointerup", (e) => {
        if (!inicio || Math.hypot(e.clientX - inicio[0], e.clientY - inicio[1]) > 6) return;
        const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
        raycaster.setFromCamera(ndc, camera);
        const visibles = Object.values(grupos).filter((g) => g.grupo.visible).map((g) => g.malla);
        let id = "";
        try {
          const golpes = raycaster.intersectObjects(visibles, false);
          if (golpes.length) id = Object.keys(grupos).find((k) => grupos[k].malla === golpes[0].object) || "";
        } catch (err) {}
        avisar("sel:" + id);
      });

      // Hacia dónde mira la cámara (en el suelo), para mover con las flechas en el sentido de la pantalla.
      const dir = new THREE.Vector3();
      controls.addEventListener("change", () => {
        camera.getWorldDirection(dir);
        avisar("dir:" + dir.x.toFixed(3) + "," + dir.z.toFixed(3));
      });

      addEventListener("resize", () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
      });
      renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
      avisar("pagina");
    } catch (e) {
      avisar("No se pudo abrir la vista fotorrealista: " + (e && e.message ? e.message : e) + " (hace falta conexión a internet para cargar el visor)");
    }
    </script>
    </body></html>
    """
}
