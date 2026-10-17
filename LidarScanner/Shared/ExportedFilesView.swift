import SwiftUI
import QuickLook

/// Lista de archivos de un escaneo con vista previa 3D/AR (USDZ) y botón de compartir.
struct ExportedFilesView: View {
    let title: String
    let files: [URL]
    @State private var preview: URL?

    var body: some View {
        List {
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

    private func icon(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "usdz": "cube.fill"
        case "obj": "cube.transparent"
        case "ply": url.lastPathComponent.contains("nube") ? "circle.grid.3x3.fill" : "square.grid.3x3.square"
        default: "doc"
        }
    }

    private func description(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "usdz": "Vista 3D / AR en iPhone"
        case "obj": "Malla con color (Blender, ZBrush…)"
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
                    ExportedFilesView(title: folder.lastPathComponent, files: ScanStorage.files(in: folder))
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
