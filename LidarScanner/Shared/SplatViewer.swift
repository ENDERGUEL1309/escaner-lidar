import SwiftUI
import WebKit

/// Vista fotorrealista (Gaussian Splatting) de un escaneo, con el visor web Spark (three.js) dentro de la app.
/// RealityKit no dibuja splats, así que se usa un WKWebView. El archivo `.splat` se sirve desde la carpeta
/// del escaneo con un esquema propio (`escaneo://`), sin copiarlo ni subirlo a ningún sitio.
struct SplatViewerView: View {
    let file: URL
    @State private var status = "Cargando vista fotorrealista…"
    @State private var loaded = false

    var body: some View {
        ZStack(alignment: .bottom) {
            SplatWebView(file: file, status: $status, loaded: $loaded)
                .ignoresSafeArea()
                .background(Color.black)
            Text(loaded ? "Arrastra para girar · pellizca para acercar · dos dedos para desplazar" : status)
                .font(.caption)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.bottom, 8)
        }
        .navigationTitle("Vista fotorrealista")
        .navigationBarTitleDisplayMode(.inline)
    }

    static func file(in folder: URL) -> URL? {
        let url = folder.appendingPathComponent("ia/splats/escena.splat")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

private struct SplatWebView: UIViewRepresentable {
    let file: URL
    @Binding var status: String
    @Binding var loaded: Bool

    func makeCoordinator() -> Coordinator { Coordinator(file: file, status: $status, loaded: $loaded) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(context.coordinator, forURLScheme: "escaneo")
        config.userContentController.add(context.coordinator, name: "estado")
        let view = WKWebView(frame: .zero, configuration: config)
        view.isOpaque = false
        view.backgroundColor = .black
        view.scrollView.isScrollEnabled = false
        view.loadHTMLString(Self.html, baseURL: URL(string: "escaneo://visor/"))
        return view
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKURLSchemeHandler, WKScriptMessageHandler {
        let file: URL
        var status: Binding<String>
        var loaded: Binding<Bool>

        init(file: URL, status: Binding<String>, loaded: Binding<Bool>) {
            self.file = file
            self.status = status
            self.loaded = loaded
        }

        /// Sirve `escaneo://visor/escena.splat` desde el disco.
        func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
            guard task.request.url?.lastPathComponent == "escena.splat", let data = try? Data(contentsOf: file) else {
                task.didFailWithError(URLError(.fileDoesNotExist))
                return
            }
            let response = HTTPURLResponse(url: task.request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
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
            DispatchQueue.main.async {
                if text == "listo" { self.loaded.wrappedValue = true } else { self.status.wrappedValue = text }
            }
        }
    }

    /// Visor: Spark 2.4 sobre three.js 0.180 (mismas versiones que su documentación).
    /// La escena se encuadra con la caja de los splats; la cámara orbita alrededor de su centro.
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

      avisar("Descargando el modelo fotorrealista…");
      const splats = new SplatMesh({ url: "escaneo://visor/escena.splat" });
      scene.add(splats);
      await splats.initialized;

      const caja = splats.getBoundingBox ? splats.getBoundingBox() : new THREE.Box3().setFromObject(splats);
      const centro = caja.getCenter(new THREE.Vector3());
      const tam = caja.getSize(new THREE.Vector3()).length() || 4;
      camera.position.copy(centro).add(new THREE.Vector3(tam * 0.35, tam * 0.3, tam * 0.35));
      controls.target.copy(centro);
      controls.update();
      avisar("listo");

      addEventListener("resize", () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
      });
      renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
    } catch (e) {
      avisar("No se pudo abrir la vista fotorrealista: " + (e && e.message ? e.message : e) + " (hace falta conexión a internet para cargar el visor)");
    }
    </script>
    </body></html>
    """
}
