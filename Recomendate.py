## Creación das 500 cancións recomendadas para cada playlsit sen repeticións
import json

# Rutas
TOP_TRACKS_PATH = "top_500_tracks.json"     
TEST_PATH = "../spotify_test_playlists/test_input_playlists.json"             # test 
OUTPUT_PATH = "recommendations_per_playlist.json"

TOP_K = 500

# Cargamos top500+ cancións máis populares
with open(TOP_TRACKS_PATH, "r", encoding="utf-8") as f:
    top_tracks = json.load(f)

# Lista ordenada de URIs por popularidade
popular_track_uris = [t["track_uri"] for t in top_tracks]

print("Top de canciones cargado:", len(popular_track_uris))

# Cargamos as playlists a recomendar
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

playlists = test_data["playlists"]

print("Playlists de test:", len(playlists))

# Xeramos recomendacións
recommendations = []

for pl in playlists:
    pid = pl["pid"]

    # Canciones xa presentes naa playlist (evitar repeticións)
    present_tracks = {
        track["track_uri"] for track in pl.get("tracks", [])
    }

    recs = []
    for uri in popular_track_uris:
        if uri not in present_tracks:
            recs.append(uri)
            if len(recs) == TOP_K:
                break

    # Seguridade extra (no debería pasar)
    assert len(recs) == TOP_K

    recommendations.append({
        "pid": pid,
        "recommendations": recs
    })

# Gardamos en .json pra realizar a evaluación
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(recommendations, f, indent=2)

print(f"Recomendaciones guardadas en {OUTPUT_PATH}")
