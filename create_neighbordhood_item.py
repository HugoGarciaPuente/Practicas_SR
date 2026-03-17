import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz

TRAIN_MATRIX_PATH = "playlist_track_matrix.npz"
TRACK_TO_COL_PATH = "track_to_col.json"
TRACK_TO_COL_TEST_PATH = "track_to_col_test.json"
OUTPUT_PATH = "neighborhood_item.json"

# Número de vecinos a guardar por canción
K_NEIGHBORS = 150


def build_item_neighborhood(train_matrix, track_to_col, track_to_col_test, k=K_NEIGHBORS):
    """
    Construye un vecindario item-based usando similitud coseno.

    Para cada canción del conjunto test que también aparece en train,
    calcula las k canciones más similares del training usando:

        sim(i, j) = (r_i · r_j) / (||r_i|| * ||r_j||)

    donde r_i es el vector de co-ocurrencia de la canción i en playlists.

    Parámetros
    ----------
    train_matrix : sp.csr_matrix, shape (n_playlists, n_tracks_train)
        Matriz binaria playlist × canción del training.
    track_to_col : dict
        Mapeo URI → índice de columna en train_matrix.
    track_to_col_test : dict
        Mapeo URI → índice de columna en test_matrix.
    k : int
        Tamaño del vecindario.

    Devuelve
    --------
    neighborhoods : dict
        { track_uri: [ (neighbor_uri, similitud), ... ] }
    """

    print("=== BUILD ITEM NEIGHBORHOOD ===")

    # Invertir mapping para recuperar URI a partir del índice
    col_to_track = {v: k_uri for k_uri, v in track_to_col.items()}

    # Canciones del test
    test_uris = set(track_to_col_test.keys())

    # Sólo calculamos vecindario para canciones test que existen en train
    common_tracks = [
        track_to_col[uri]
        for uri in test_uris
        if uri in track_to_col
    ]

    print(f"Tracks en test:                {len(test_uris)}")
    print(f"Tracks test presentes en train: {len(common_tracks)}")

    # ------------------------------------------------------------------
    # Normalización para similitud coseno
    #
    # Como la matriz es binaria (0/1), la norma L2 de la columna j es:
    #   ||r_j|| = sqrt( sum_u r_{u,j}^2 ) = sqrt( getnnz(col=j) )
    #
    # Guardamos también las normas para poder dividir por ||r_i||
    # (la canción query) — esto es el fix del denominador.
    # ------------------------------------------------------------------
    track_popularity = train_matrix.getnnz(axis=0)   # shape (n_tracks,)
    norms = np.sqrt(track_popularity.astype(float))
    norms[norms == 0] = 1.0                           # evitar división por cero

    inv_norms = 1.0 / norms
    Inv_D = sp.diags(inv_norms)   # matriz diagonal (n_tracks × n_tracks)

    # Traspuesta en CSR para acceso eficiente a filas (track → playlists)
    M_T = train_matrix.T.tocsr()   # shape (n_tracks, n_playlists)

    print(f"Calculando vecindarios (k={k})...")

    neighborhoods = {}

    for idx, track_id in enumerate(common_tracks):

        if idx % 5000 == 0:
            print(f"  Procesadas {idx}/{len(common_tracks)} canciones...")

        # Vector de la canción query: shape (1, n_playlists)
        vec = M_T[track_id]

        # --- NUMERADOR ---
        # scores[j] = r_i · r_j  para toda canción j
        # shape: (1, n_tracks)
        scores = vec.dot(train_matrix)

        # --- DENOMINADOR: dividir por ||r_j|| para todos los j ---
        scores = scores.dot(Inv_D)

        # --- FIX: dividir también por ||r_i|| (norma de la canción query) ---
        # Sin esto la similitud coseno queda sin normalizar por un lado
        scores = scores / norms[track_id]

        scores = scores.toarray().ravel()

        # La similitud de una canción consigo misma no es útil
        scores[track_id] = 0.0

        if scores.max() == 0:
            continue

        # Top-k vecinos (argpartition es O(n) en vez de O(n log n))
        n_non_zero = np.count_nonzero(scores)
        current_k = min(k, n_non_zero)

        if current_k == 0:
            continue

        top_idx = np.argpartition(-scores, current_k)[:current_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        track_uri = col_to_track[track_id]

        neighbors = []
        for t in top_idx:
            uri = col_to_track.get(int(t))
            if uri is None:
                continue
            neighbors.append((uri, float(scores[t])))

        neighborhoods[track_uri] = neighbors

    print(f"Vecindarios calculados: {len(neighborhoods)}")
    return neighborhoods


if __name__ == "__main__":

    print("Cargando training matrix...")
    train_matrix = load_npz(TRAIN_MATRIX_PATH).tocsr()
    print(f"  Shape: {train_matrix.shape}")

    with open(TRACK_TO_COL_PATH) as f:
        track_to_col = json.load(f)

    with open(TRACK_TO_COL_TEST_PATH) as f:
        track_to_col_test = json.load(f)

    item_neighborhoods = build_item_neighborhood(
        train_matrix,
        track_to_col,
        track_to_col_test,
        k=K_NEIGHBORS
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(item_neighborhoods, f)

    print(f"Vecindarios guardados en: {OUTPUT_PATH}")
