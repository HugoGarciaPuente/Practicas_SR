import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz

TRAIN_MATRIX_PATH = "playlist_track_matrix.npz"
TEST_MATRIX_PATH = "test_matrix.npz"
OUTPUT_PATH = "neighborhood_user.json"

# Número de vecinos a guardar por playlist
K_NEIGHBORS = 150


def build_user_neighborhood(train_matrix, test_matrix, k=K_NEIGHBORS):
    """
    Construye un vecindario user-based usando similitud coseno.

    Para cada playlist del conjunto test calcula las k playlists más
    similares del training usando:

        sim(u, v) = (r_u · r_v) / (||r_u|| * ||r_v||)

    donde r_u es el vector de canciones de la playlist u.

    Ambas matrices comparten el mismo espacio de columnas (track_to_col del train),
    por lo que no es necesario ningún reindexado.

    Parámetros
    ----------
    train_matrix : sp.csr_matrix, shape (n_playlists_train, n_tracks)
        Matriz binaria playlist × canción del training.
    test_matrix : sp.csr_matrix, shape (n_playlists_test, n_tracks)
        Matriz binaria playlist × canción del test.
    k : int
        Tamaño del vecindario.

    Devuelve
    --------
    neighborhoods : dict
        { fila_test (int): [ (fila_train (int), similitud), ... ] }
    """

    print("=== BUILD USER NEIGHBORHOOD ===")

    n_test  = test_matrix.shape[0]
    n_train = train_matrix.shape[0]

    print(f"Playlists en test:  {n_test}")
    print(f"Playlists en train: {n_train}")

    # ------------------------------------------------------------------
    # Normalización para similitud coseno
    # ------------------------------------------------------------------
    train_popularity = train_matrix.getnnz(axis=1)   # shape (n_train,)
    norms_train = np.sqrt(train_popularity.astype(float))
    norms_train[norms_train == 0] = 1.0

    test_popularity = test_matrix.getnnz(axis=1)     # shape (n_test,)
    norms_test = np.sqrt(test_popularity.astype(float))
    norms_test[norms_test == 0] = 1.0

    M_train   = train_matrix.tocsr()
    M_test    = test_matrix.tocsr()
    M_train_T = M_train.T.tocsr()   # (n_tracks, n_train) — precomputado

    # ------------------------------------------------------------------
    # Procesamiento por lotes (batches)
    #
    #   S_batch = M_test[start:end] @ M_train.T   shape (B, n_train)
    #
    # El tamaño de batch controla el uso de RAM:
    #   B x n_train x 4 bytes  ->  500 x 1M x 4 ~ 2 GB (denso)
    # Ajusta BATCH_SIZE según la RAM disponible.
    # ------------------------------------------------------------------
    BATCH_SIZE = 500

    print(f"Calculando vecindarios (k={k}, batch={BATCH_SIZE})...")

    neighborhoods = {}

    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)

        if start % 1000 == 0:
            print(f"  Procesadas {start}/{n_test} playlists...")

        # --- NUMERADOR: shape (B, n_train) ---
        scores_batch = M_test[start:end].dot(M_train_T)
        scores_batch = scores_batch.toarray().astype(np.float32)

        # --- DENOMINADOR: / ||r_v|| para cada vecino en train ---
        scores_batch /= norms_train[np.newaxis, :]

        # --- / ||r_u|| para cada playlist query ---
        scores_batch /= norms_test[start:end, np.newaxis]

        # --- Top-k por fila ---
        actual_k = min(k, scores_batch.shape[1])

        for i, scores in enumerate(scores_batch):
            if scores.max() == 0:
                continue

            n_non_zero = np.count_nonzero(scores)
            current_k = min(actual_k, n_non_zero)
            if current_k == 0:
                continue

            top_idx = np.argpartition(-scores, current_k)[:current_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            neighborhoods[start + i] = [(int(v), float(scores[v])) for v in top_idx]

    print(f"Vecindarios calculados: {len(neighborhoods)}")
    return neighborhoods


if __name__ == "__main__":

    print("Cargando training matrix...")
    train_matrix = load_npz(TRAIN_MATRIX_PATH).tocsr()
    print(f"  Shape: {train_matrix.shape}")

    print("Cargando test matrix...")
    test_matrix = load_npz(TEST_MATRIX_PATH).tocsr()
    print(f"  Shape: {test_matrix.shape}")

    user_neighborhoods = build_user_neighborhood(
        train_matrix,
        test_matrix,
        k=K_NEIGHBORS
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(user_neighborhoods, f)

    print(f"Vecindarios guardados en: {OUTPUT_PATH}")
