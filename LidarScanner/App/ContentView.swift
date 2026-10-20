import SwiftUI

struct ContentView: View {
    @State private var showCloudSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    header

                    NavigationLink {
                        SceneScanView()
                    } label: {
                        ModeCard(title: "Escanear entorno",
                                 subtitle: "Habitaciones, casas, exteriores y objetos grandes. Malla LiDAR y nube de puntos a color en tiempo real.",
                                 systemImage: "house.and.flag.fill",
                                 tint: .cyan,
                                 available: SceneScanner.isSupported)
                    }
                    .disabled(!SceneScanner.isSupported)

                    NavigationLink {
                        ObjectScanView()
                    } label: {
                        ModeCard(title: "Escanear objeto",
                                 subtitle: "Objetos y personas quietas. Captura guiada con LiDAR + fotogrametría: modelo USDZ con textura fotorrealista.",
                                 systemImage: "cube.transparent.fill",
                                 tint: .orange,
                                 available: ObjectScanModel.isSupported)
                    }
                    .disabled(!ObjectScanModel.isSupported)

                    NavigationLink {
                        ScansListView()
                    } label: {
                        ModeCard(title: "Mis escaneos",
                                 subtitle: "Ver, previsualizar en 3D/AR y compartir archivos PLY, OBJ y USDZ.",
                                 systemImage: "folder.fill",
                                 tint: .green,
                                 available: true)
                    }

                    if !SceneScanner.isSupported {
                        Text("Este dispositivo no tiene sensor LiDAR. Se necesita un iPhone Pro o iPad Pro con LiDAR.")
                            .font(.footnote)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding()
            }
            .navigationTitle("Escáner LiDAR")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showCloudSettings = true } label: {
                        Label("Ajustes de la nube", systemImage: "icloud.and.arrow.up")
                    }
                }
            }
            .sheet(isPresented: $showCloudSettings) { CloudSettingsView() }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "dot.radiowaves.left.and.right")
                .font(.system(size: 34))
                .foregroundStyle(.cyan)
            Text("LiDAR + cámara + seguimiento de movimiento para crear modelos 3D a color y a escala real.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 4)
    }
}

private struct ModeCard: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: Color
    let available: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: systemImage)
                .font(.title)
                .foregroundStyle(tint)
                .frame(width: 44)
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.title3.bold()).foregroundStyle(.primary)
                Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
                if !available {
                    Text("No disponible en este dispositivo").font(.caption).foregroundStyle(.red)
                }
            }
            Spacer(minLength: 0)
            Image(systemName: "chevron.right").foregroundStyle(.tertiary)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20))
        .opacity(available ? 1 : 0.5)
    }
}
