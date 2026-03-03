
import json
import numpy as np

from scipy.sparse import load_npz
from scipy.sparse import csr_matrix



TRAIN_MATRIX_PATH = "playlist_track_matrix.npz"
TEST_MATRIX_PATH = "test_matrix.npz"
TRACK_MAP_PATH = "track_to_col.json"
TRACK_MAP_TEST_PATH = "track_to_col_test.json"
OUTPUT_PATH = "neighborhood_user.json"



matrix = load_npz(TRAIN_MATRIX_PATH)
test_matrix = load_npz(TEST_MATRIX_PATH)

with open(TRACK_MAP_PATH, "r") as f:
    track_to_col = json.load(f)

with open(TRACK_MAP_TEST_PATH, "r") as f:
    track_to_col_test = json.load(f)



def build_neighborhood(
    train_matrix,
    test_matrix,
    k,
    return_scores=False,
    verbose=False
):
    """
    k-NN playlist–playlist con coseno binario correctamente alineado.
    Cada fila es una playlist.
    No reconstruye matrices ni proyecta manualmente vectores.
    Funciona con test_matrix dispersa (muchas filas vacías).
    """

    neighborhoods = {}

    # Normas L2 del train (binario)
    train_norms = np.sqrt(train_matrix.getnnz(axis=1))

    # Filas no vacías del test
    test_nnz = test_matrix.getnnz(axis=1)
    non_empty_rows = np.where(test_nnz > 0)[0]

    if verbose:
        print(f"Playlists test no vacías: {len(non_empty_rows)}")

    for pid in non_empty_rows:
        test_row = test_matrix.getrow(pid)
        test_cols = test_row.indices        # tracks de la playlist
        test_norm = np.sqrt(len(test_cols))

        # Submatriz: playlists de train restringidas a tracks del test
        train_sub = train_matrix[:, test_cols]

        # Producto escalar playlist–playlist
        dots = train_sub.sum(axis=1).A1     # |A ∩ B|

        # Coseno binario correctamente definido
        denom = train_norms * test_norm
        sims = np.zeros_like(dots, dtype=float)
        mask = denom > 0
        sims[mask] = dots[mask] / denom[mask]

        # Caso límite: no hay solapamiento real
        if sims.max() == 0:
            neighborhoods[str(pid)] = []
            continue

        # Top-k vecinos reales
        kk = min(k, (sims > 0).sum())
        topk = np.argpartition(-sims, kk - 1)[:kk]
        topk = topk[np.argsort(-sims[topk])]

        if return_scores:
            neighborhoods[str(pid)] = [
                (int(i), float(sims[i])) for i in topk if sims[i] > 0
            ]
        else:
            neighborhoods[str(pid)] = [
                int(i) for i in topk if sims[i] > 0
            ]

    return neighborhoods

k = 2

neighborhoods = build_neighborhood(
    matrix,
    test_matrix,
    k
)


print(len(neighborhoods))
print(list(neighborhoods.items())[:3])


with open(OUTPUT_PATH, "w") as f:
    json.dump(neighborhoods, f)
