import SwiftUI
import QuickLook

/// Lista de archivos de un escaneo con vista previa 3D/AR (USDZ) y botón de compartir.
struct ExportedFilesView: View {
    let title: String
    let files: [URL]
    var objects: [SceneObjectInfo] = []
    @State private var preview: URL?

    /// Carpeta del escaneo, si contiene una malla con color que se pueda abrir en el visor.
    private var viewerFolder: URL? {
        guard let folder = files.first?.deletingLastPathComponent() else { return nil }
        let mesh = folder.appendingPathComponent("malla_color.ply")
        return FileManager.default.fileExists(atPath: mesh.path) ? folder : nil
    }

    var body: some View {
        List {
            if let viewerFolder {
                Section {
                    NavigationLink {
                        SceneViewerView(folder: viewerFolder)
                    } label: {
                        Label("Ver en 3D a color", systemImage: "rotate.3d")
                            .font(.headline)
                            .foregroundStyle(.cyan)
                            .padding(.vertical, 6)
                    }
                } footer: {
                    Text("Toca cada objeto para seleccionarlo, aislarlo, ocultarlo o exportarlo por separado.")
                }
            }

            if !objects.isEmpty {
                Section {
                    ForEach(objects) { object in
                        HStack(spacing: 12) {
                            Image(systemName: icon(for: object))
                                .foregroundStyle(object.estructura ? Color.secondary : Color.orange)
                                .frame(width: 28)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(object.nombre).font(.body.weight(.medium))
                                Text(object.detailText)
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } header: {
                    Text("Objetos detectados")
                } footer: {
                    Text("Cada uno es un objeto independiente en escena_objetos.obj / .usdz: en Blender puedes moverlo, borrarlo o editarlo por separado.")
                }
            }

            Section {
                ForEach(files, id: \.self) { url in
                    HStack(spacing: 12) {
                        Image(systemName: icon(for: url))
                            .font(.title2)
                            .foregroundStyle(.cyan)
                            .frame(width: 32)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(url.lastPathComponent).font(.body.weight(.medium))
                            Text("\(description(for: url)) · \(ScanStorage.fileSizeText(url))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if url.pathExtension.lowercased() == "usdz" {
                            Button { preview = url } label: {
                                Image(systemName: "arkit").font(.title3)
                            }
                            .buttonStyle(.borderless)
                        }
                        ShareLink(item: url) {
                            Image(systemName: "square.and.arrow.up").font(.title3)
                        }
                        .buttonStyle(.borderless)
                    }
                    .padding(.vertical, 4)
                }
            } footer: {
                Text("Los archivos también están en la app Archivos › En mi iPhone › Escáner LiDAR › Escaneos.")
            }
        }
        .navigationTitle(title)
        .quickLookPreview($preview)
    }

    private func icon(for object: SceneObjectInfo) -> String {
        switch object.categoria {
        case "suelo": "square.bottomhalf.filled"
        case "pared": "square.split.1x2"
        case "techo": "square.tophalf.filled"
        case "puerta": "door.left.hand.closed"
        case "ventana": "window.vertical.closed"
        case "mesa": "table.furniture"
        case "asiento": "chair.lounge"
        default: "cube"
        }
    }

    private func icon(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "usdz": "cube.fill"
        case "obj": "cube.transparent"
        case "json": "list.bullet.rectangle"
        case "ply": url.lastPathComponent.contains("nube") ? "circle.grid.3x3.fill" : "square.grid.3x3.square"
        case "jpg": "photo"
        case "mtl": "paintpalette"
        default: "doc"
        }
    }

    private func description(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "usdz": url.lastPathComponent.hasPrefix("malla_color") ? "Geometría para AR (sin color: usa «Ver en 3D a color»)" : "Con textura fotográfica · 3D / AR"
        case "obj": url.lastPathComponent.contains("objetos") ? "Escena separada por objetos (Blender)" : "Modelo completo con textura (Blender, ZBrush…)"
        case "jpg": "Textura fotográfica de alta resolución"
        case "mtl": "Material del OBJ (enlaza la textura)"
        case "json": "Inventario de objetos y medidas"
        case "ply": url.lastPathComponent.contains("nube") ? "Nube de puntos a color" : "Malla con color y normales"
        default: "Archivo"
        }
    }
}

struct ScansListView: View {
    @State private var folders: [URL] = []

    var body: some View {
        List {
            ForEach(folders, id: \.self) { folder in
                NavigationLink {
                    ExportedFilesView(title: folder.lastPathComponent,
                                      files: ScanStorage.files(in: folder),
                                      objects: ScanStorage.sceneObjects(in: folder))
                } label: {
                    Label(folder.lastPathComponent.replacingOccurrences(of: "_", with: " "),
                          systemImage: folder.lastPathComponent.hasPrefix("Objeto") ? "cube.fill" : "house.fill")
                }
            }
            .onDelete { offsets in
                for i in offsets { try? FileManager.default.removeItem(at: folders[i]) }
                folders.remove(atOffsets: offsets)
            }
        }
        .overlay {
            if folders.isEmpty {
                ContentUnavailableView("Sin escaneos todavía",
                                       systemImage: "cube.transparent",
                                       description: Text("Tus modelos aparecerán aquí."))
            }
        }
        .navigationTitle("Mis escaneos")
        .onAppear { folders = ScanStorage.allScanFolders() }
    }
}
