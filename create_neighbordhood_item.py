
import json
import numpy as np
from scipy.sparse import load_npz

TRAIN_MATRIX_PATH = "playlist_track_matrix.npz"
TEST_MATRIX_PATH = "test_matrix.npz"
OUTPUT_PATH = "neighborhood_item.json"

def build_item_neighborhood(
    train_matrix,
    test_matrix,
    k=50,
    verbose=True
):
    """
    Item-based vecindario pra test.

    Pra cada playlist do test:
        - toma as suas cancions
        - calcula similitud coseno contra canciones del train
        - devolve top-k cancions mas similares
          (excluyendo as presentes)

    Devolve:
        { pid: [(track_index, score), ...] }
    """

    neighborhoods = {}

    n_train_playlists, n_tracks = train_matrix.shape

    if verbose:
        print("=== ITEM-BASED NEIGHBORHOOD ===")
        print("Train matrix shape:", train_matrix.shape)
        print("Test matrix shape:", test_matrix.shape)
        print("Total tracks:", n_tracks)
        print()

    # Precomputamos norma L2 de cada track (columna)
    if verbose:
        print("Calculando normas de tracks (train)...")

    track_popularity = train_matrix.getnnz(axis=0)
    track_norms = np.sqrt(track_popularity)

    if verbose:
        print("Normas calculadas.")
        print()

    # Iteramos sobre playlists de test no vacías
    test_nnz = test_matrix.getnnz(axis=1)
    non_empty_rows = np.where(test_nnz > 0)[0]

    if verbose:
        print("Playlists test no vacías:", len(non_empty_rows))
        print()

    for count, pid in enumerate(non_empty_rows):

        if verbose and count % 100 == 0:
            print(f"Procesando playlist {count}/{len(non_empty_rows)}")

        test_row = test_matrix.getrow(pid)
        test_tracks = test_row.indices

        # Conjunto para exclusión
        test_track_set = set(test_tracks)

        # Scores acumulados para tracks candidatos
        scores = {}

        # Para cada track presente en la playlist test
        for track_j in test_tracks:

            # Playlists del train donde aparece track_j
            train_playlists = train_matrix[:, track_j].nonzero()[0]

            # Submatriz restringida a esas playlists
            submatrix = train_matrix[train_playlists]

            # Intersección con todas las canciones
            cooccurrence = submatrix.sum(axis=0).A1

            # Similitud coseno binaria
            denom = track_norms * track_norms[track_j]
            mask = denom > 0
            sims = np.zeros_like(cooccurrence, dtype=float)
            sims[mask] = cooccurrence[mask] / denom[mask]

            # Acumulamos scores
            for track_i in np.where(sims > 0)[0]:
                if track_i not in test_track_set:
                    scores[track_i] = scores.get(track_i, 0) + sims[track_i]

        if len(scores) == 0:
            neighborhoods[str(pid)] = []
            continue

        # Top-k por score acumulado
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        neighborhoods[str(pid)] = [
            (int(track), float(score))
            for track, score in sorted_items
        ]

    if verbose:
        print()
        print("Vecindario item-based completado.")

    return neighborhoods


# EJECUCIÓN

train_matrix = load_npz(TRAIN_MATRIX_PATH)
test_matrix = load_npz(TEST_MATRIX_PATH)

item_neighborhoods = build_item_neighborhood(
    train_matrix,
    test_matrix,
    k=50,
    verbose=True
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(item_neighborhoods, f)

print("Guardado en:", OUTPUT_PATH)
