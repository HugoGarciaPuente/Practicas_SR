import json
from collections import defaultdict
from scipy.sparse import load_npz

NEIGHBORHOOD_PATH      = "neighborhood_user.json"
TRAIN_MATRIX_PATH      = "playlist_track_matrix.npz"
TEST_MATRIX_PATH       = "test_matrix.npz"
TRACK_TO_COL_PATH      = "track_to_col.json"
TOP500_PATH            = "top_500_tracks.json"
OUTPUT_PATH            = "recommendations_user.json"

N_RECOMMENDATIONS = 500


#Cargar vecindario user-based

print("Cargando vecindario user...")
with open(NEIGHBORHOOD_PATH, "r", encoding="utf-8") as f:
    user_neighbors = json.load(f)
print(f"  Vecindarios cargados: {len(user_neighbors)}")


with open(TRACK_TO_COL_PATH, "r", encoding="utf-8") as f:
    track_to_col = json.load(f)

#Convierte los indices en uris 
col_to_track      = {v: k for k, v in track_to_col.items()}

col_to_track_test = col_to_track   # mismo espacio de columnas
print(f"  Tracks (espacio compartido): {len(track_to_col)}")
print(f"  Tracks training: {len(track_to_col)}")
print(f"  Tracks test:     {len(col_to_track_test)}")


#Cargamos las populares 
print("Cargando tracks populares (fallback)...")
with open(TOP500_PATH, "r", encoding="utf-8") as f:
    top500_raw = json.load(f)

popular_uris_ordered = [entry["track_uri"] for entry in top500_raw]
print(f"  Tracks en pool de popularidad: {len(popular_uris_ordered)}")


#cargamos tanto matriz de train como shape 
print("Cargando matrices...")
train_matrix = load_npz(TRAIN_MATRIX_PATH).tocsr()
test_matrix  = load_npz(TEST_MATRIX_PATH).tocsr()
print(f"  Train shape: {train_matrix.shape}")
print(f"  Test shape:  {test_matrix.shape}")

n_playlists = test_matrix.shape[0]


#Generación de recomendaciones 
print(f"\nGenerando recomendaciones ({n_playlists} playlists)...\n")

K_VALUES = [15, 30, 60, 100, 150] #Probamos distintos valores de k, (numero de vecinos)

for k in K_VALUES:
    print(f"\n=== K = {k} ===")
    recommendations      = []
    total_seed_tracks    = 0    
    playlists_with_neighbors = 0  
    playlists_with_fallback  = 0  

    for pid in range(n_playlists):

        if pid % 1000 == 0:
            print(f"  Playlist {pid}/{n_playlists}...")

        try:
            #tracks ya en la playlist test
            row = test_matrix.getrow(pid)
            present_uris = set()
            for test_col_id in row.indices:
                total_seed_tracks += 1
                uri = col_to_track_test.get(test_col_id)
                if uri is not None:
                    present_uris.add(uri)

        #Veicnos de esta playlist test 
            neighbors = user_neighbors.get(str(pid), [])

            scores = defaultdict(float)

            if neighbors:
                playlists_with_neighbors += 1

                for train_pid, similarity in neighbors[:k]:

                #Tracks de la playlist vecina (training)
                    train_row = train_matrix.getrow(train_pid)

                    for train_col_id in train_row.indices:
                        uri = col_to_track.get(train_col_id)
                        if uri is None:
                            continue
                        if uri in present_uris:
                            continue
                        scores[uri] += similarity #Cuanto mas similiar es la playlist mas peso tiene 
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rec_uris = [uri for uri, _ in ranked[:N_RECOMMENDATIONS]]

        #Rellenamos hasta las 500 con las populares sin duplicados ni canciones ya presentes 
            if len(rec_uris) < N_RECOMMENDATIONS:
                playlists_with_fallback += 1
                rec_set = set(rec_uris)
                for pop_uri in popular_uris_ordered:
                    if len(rec_uris) >= N_RECOMMENDATIONS:
                        break
                    if pop_uri not in present_uris and pop_uri not in rec_set:
                        rec_uris.append(pop_uri)
                        rec_set.add(pop_uri)

        except Exception as e:
            print(f"  Error en playlist {pid}: {e}")
            rec_uris = []
            present_uris = set()
            rec_set = set()
            for pop_uri in popular_uris_ordered:
                if len(rec_uris) >= N_RECOMMENDATIONS:
                    break
                if pop_uri not in present_uris and pop_uri not in rec_set:
                    rec_uris.append(pop_uri)
                    rec_set.add(pop_uri)

        recommendations.append({
            "pid": pid,
            "recommendations": rec_uris
        })

    print("\n--- DIAGNÓSTICO ---")
    print(f"Total playlists procesadas:             {n_playlists}")
    print(f"Total tracks en seeds:                  {total_seed_tracks}")
    print(f"Playlists con al menos 1 vecino:        {playlists_with_neighbors}")    
    print(f"Playlists con fallback de popularidad:  {playlists_with_fallback}")
    print(f"  ({100*playlists_with_fallback/n_playlists:.1f}% del total)")

    lengths = [len(r["recommendations"]) for r in recommendations]
    print(f"Longitud mín/máx de listas:             {min(lengths)} / {max(lengths)}")
    assert min(lengths) == N_RECOMMENDATIONS, "Alguna playlist tiene menos de 500 recomendaciones"
    print("OK: todas las playlists tienen exactamente 500 recomendaciones.")
    output_path = f"recommendations_user_k{k}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f)
    print(f"Guardado: {output_path}")


