
import json
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds

TRAIN_MATRIX_PATH  = "playlist_track_matrix.npz"
TEST_MATRIX_PATH   = "test_matrix.npz"
TRACK_TO_COL_PATH  = "track_to_col.json"
TOP500_PATH        = "top_500_tracks.json"
ROW_TO_PID_PATH    = "row_to_pid_test.json"

N_RECOMMENDATIONS  = 500
K_VALUES = [10, 50, 100, 200] #características latentes a probar
BATCH_SIZE         = 50   # playlists por bloque (axustar según RAM)


def recommend_from_scores(score_row, present_uris, col_to_track,
                           popular_uris, n=N_RECOMMENDATIONS):
    """
    Dado un vector de scores pra unha playlist, devolve as N
    recomendacions excluindo o seed e rellenando con populares.

    Parámetros
    ----------
    score_row    : np.ndarray (n_tracks,) — scores estimados
    present_uris : set  — URIs xa no seed (a excluir)
    col_to_track : dict — col_index → URI
    popular_uris : list — pool de fallback ordenado por popularidade
    n            : int  — número de recomendacions a devolver

    Devuelve
    --------
    rec_uris : list de URIs ordenadas por score descendente
    """
    top_cols = np.argsort(-score_row)

    rec_uris = []
    for col in top_cols:
        if len(rec_uris) >= n:
            break
        uri = col_to_track.get(int(col))
        if uri is None:
            continue
        if uri in present_uris:
            continue
        rec_uris.append(uri)

    # Fallback con populares si non chega a N
    if len(rec_uris) < n:
        rec_set = set(rec_uris)
        for pop_uri in popular_uris:
            if len(rec_uris) >= n:
                break
            if pop_uri not in present_uris and pop_uri not in rec_set:
                rec_uris.append(pop_uri)
                rec_set.add(pop_uri)

    return rec_uris


if __name__ == "__main__":

    # Cargar todo unha vez
    print("Cargando matrices...")   
    train_matrix = load_npz(TRAIN_MATRIX_PATH).tocsr()
    test_matrix  = load_npz(TEST_MATRIX_PATH).tocsr()

    with open(TRACK_TO_COL_PATH) as f:
        track_to_col = json.load(f)
        col_to_track = {v: k for k, v in track_to_col.items()}

    with open(TOP500_PATH) as f:
        popular_uris = [e["track_uri"] for e in json.load(f)]

    with open(ROW_TO_PID_PATH) as f:
        row_to_pid = {int(k): v for k, v in json.load(f).items()}

    n_test = test_matrix.shape[0]
    # SVD proxectando pra cada valor das características latentes
    for K_LATENT in K_VALUES:
        print(f"\n{'='*50}")
        print(f"Probando K = {K_LATENT}")
        print(f"{'='*50}")

        # SVD solo sobre train — se recalcula para cada K
        print(f"Calculando SVD sobre train (k={K_LATENT})...")
        U_train, sigma, Vt = svds(train_matrix.astype(np.float32), k=K_LATENT)

        idx     = np.argsort(-sigma)
        sigma   = sigma[idx]
        Vt      = Vt[idx, :]
        V       = Vt.T

        sigma_inv   = 1.0 / sigma
        V_sigma_inv = V * sigma_inv   # (n_tracks, K_LATENT)

        recommendations         = []
        playlists_with_fallback = 0
        cold_start_count        = 0

        for start in range(0, n_test, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_test)
            if start % 1000 == 0:
                print(f"  Playlist {start}/{n_test}...")

            R_block      = test_matrix[start:end].toarray().astype(np.float32)
            U_block      = R_block @ V_sigma_inv          # proyección (ec. 4)
            scores_block = (U_block * sigma) @ Vt         # scoring   (ec. 5)

            for local_idx in range(scores_block.shape[0]):
                global_idx = start + local_idx
                pid = row_to_pid.get(global_idx, global_idx)

                row = test_matrix.getrow(global_idx)
                present_uris = {
                    col_to_track[int(col)]
                    for col in row.indices
                    if int(col) in col_to_track
                }

                if len(present_uris) == 0:
                    cold_start_count += 1

                rec_uris = recommend_from_scores(
                    scores_block[local_idx], present_uris,
                    col_to_track, popular_uris
                )

                if len(rec_uris) < N_RECOMMENDATIONS:
                    playlists_with_fallback += 1

                recommendations.append({"pid": pid, "recommendations": rec_uris})

            del R_block, U_block, scores_block

        output_path = f"recommendations_svd_proj_k{K_LATENT}.json"
        with open(output_path, "w") as f:
            json.dump(recommendations, f)

        lengths = [len(r["recommendations"]) for r in recommendations]
        print(f"K={K_LATENT} → cold-start: {cold_start_count} | "
            f"fallback: {playlists_with_fallback} | "
            f"min/max: {min(lengths)}/{max(lengths)}") #check das recomendacións
        print(f"Guardado: {output_path}")

