
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
K_LATENT           = 50   # número de características latentes
BATCH_SIZE         = 50   # playlists por bloque (axustar según RAM)


def recommend_from_scores(score_row, present_uris, col_to_track,
                           popular_uris, n=N_RECOMMENDATIONS):
    """
    Dado un vector de scores para una playlist, devuelve las N
    recomendaciones excluyendo el seed y rellenando con populares.

    Parámetros
    ----------
    score_row    : np.ndarray (n_tracks,) — scores estimados
    present_uris : set  — URIs ya en el seed (a excluir)
    col_to_track : dict — col_index → URI
    popular_uris : list — pool de fallback ordenado por popularidad
    n            : int  — número de recomendaciones a devolver

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

    # ------------------------------------------------------------------
    # 1. Cargar matrices e mappings
    # ------------------------------------------------------------------
    print("Cargando matrices...")
    train_matrix = load_npz(TRAIN_MATRIX_PATH).tocsr()
    test_matrix  = load_npz(TEST_MATRIX_PATH).tocsr()
    print(f"  Train shape: {train_matrix.shape}")
    print(f"  Test shape:  {test_matrix.shape}")

    with open(TRACK_TO_COL_PATH) as f:
        track_to_col = json.load(f)
    col_to_track = {v: k for k, v in track_to_col.items()}

    with open(TOP500_PATH) as f:
        popular_uris = [e["track_uri"] for e in json.load(f)]

    with open(ROW_TO_PID_PATH) as f:
        row_to_pid = {int(k): v for k, v in json.load(f).items()}

    n_test = test_matrix.shape[0]

    # ------------------------------------------------------------------
    # 2. SVD só sobre datos de training
    #
    #    svds devolve os K_LATENT valores singulares MÁIS GRANDES:
    #      U_train : (n_train, K_LATENT)  — playlists en espazo latente
    #      sigma   : (K_LATENT,)          — pesos de cada característica
    #      Vt      : (K_LATENT, n_tracks) — tracks en espazo latente
    #
    #    svds devolve en orde ASCENDENTE → invertimos
    # ------------------------------------------------------------------
    print(f"Calculando SVD só sobre train (k={K_LATENT})...")
    U_train, sigma, Vt = svds(train_matrix.astype(np.float32), k=K_LATENT)

    # Reordenar de maior a menor valor singular
    idx    = np.argsort(-sigma)
    sigma  = sigma[idx]
    U_train = U_train[:, idx]
    Vt     = Vt[idx, :]

    print(f"  U_train shape: {U_train.shape}")
    print(f"  sigma shape:   {sigma.shape}")
    print(f"  Vt shape:      {Vt.shape}")

    # V é a trasposta de Vt: (n_tracks, K_LATENT)
    V = Vt.T   # (n_tracks, K_LATENT)

    # Precalculamos V × Σ^-1 para proxección eficiente
    # (n_tracks, K_LATENT) × diag(1/sigma) = (n_tracks, K_LATENT)
    sigma_inv     = 1.0 / sigma                  # (K_LATENT,)
    V_sigma_inv   = V * sigma_inv                # broadcast: (n_tracks, K_LATENT)

    # Precalculamos Σ × Vt para scoring eficiente
    # diag(sigma) × Vt = (K_LATENT, n_tracks)  — multiplicamos cada fila de Vt por sigma
    sigma_Vt = sigma[:, np.newaxis] * Vt         # (K_LATENT, n_tracks)

    # ------------------------------------------------------------------
    # 3. Proxección + scoring en bloques
    #
    #    Para cada playlist de test con seed r_{m+1} (vector esparso):
    #
    #      Proxección ao espazo latente:
    #        u_{m+1} = r_{m+1} × V˜ × Σ^{-1}          (ec. 4)
    #                shape: (1, K_LATENT)
    #
    #      Scores para todos os tracks:
    #        r̂_{m+1} = u_{m+1} × Σ × Vt               (ec. 5)
    #                shape: (1, n_tracks)
    #
    #    Nota: u × Σ × Vt = (r × V × Σ^{-1}) × Σ × Vt = r × V × Vt
    #    Pero manteemos a fórmula explícita para claridade.
    # ------------------------------------------------------------------
    print(f"\nProxectando e recomendando en bloques de {BATCH_SIZE}...")

    recommendations         = []
    playlists_with_fallback = 0
    cold_start_count        = 0

    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)

        if start % 1000 == 0:
            print(f"  Playlist {start}/{n_test}...")

        # Extraer bloque de test como densa: (batch, n_tracks)
        R_block = test_matrix[start:end].toarray().astype(np.float32)

        # Proxección: (batch, n_tracks) × (n_tracks, K_LATENT) → (batch, K_LATENT)
        U_block = R_block @ V_sigma_inv         # ec. 4

        # Scoring: (batch, K_LATENT) × (K_LATENT, n_tracks) → (batch, n_tracks)
        # Equivalente a u × Σ × Vt fila a fila
        scores_block = (U_block * sigma) @ Vt  # ec. 5

        for local_idx in range(scores_block.shape[0]):
            global_idx = start + local_idx
            pid = row_to_pid.get(global_idx, global_idx)

            # Seed da playlist
            row = test_matrix.getrow(global_idx)
            present_uris = set()
            for col in row.indices:
                uri = col_to_track.get(int(col))
                if uri is not None:
                    present_uris.add(uri)

            if len(present_uris) == 0:
                cold_start_count += 1

            rec_uris = recommend_from_scores(
                score_row    = scores_block[local_idx],
                present_uris = present_uris,
                col_to_track = col_to_track,
                popular_uris = popular_uris
            )

            if len(rec_uris) < N_RECOMMENDATIONS:
                playlists_with_fallback += 1

            recommendations.append({
                "pid": pid,
                "recommendations": rec_uris
            })

        del R_block, U_block, scores_block

    # ------------------------------------------------------------------
    # 4. Gardar
    # ------------------------------------------------------------------
    output_path = f"recommendations_svd_proj_k{K_LATENT}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f)
    print(f"\nRecomendaciones gardadas en: {output_path}")

    # ------------------------------------------------------------------
    # 5. Diagnóstico
    # ------------------------------------------------------------------
    print("\n--- DIAGNÓSTICO ---")
    print(f"K latente:                          {K_LATENT}")
    print(f"Total playlists procesadas:         {n_test}")
    print(f"  Cold start (sen seed):            {cold_start_count} "
          f"({100*cold_start_count/n_test:.1f}%)")
    print(f"  Con seed parcial:                 {n_test - cold_start_count} "
          f"({100*(n_test-cold_start_count)/n_test:.1f}%)")
    print(f"Playlists con fallback:             {playlists_with_fallback} "
          f"({100*playlists_with_fallback/n_test:.1f}%)")
    lengths = [len(r["recommendations"]) for r in recommendations]
    print(f"Longitud mín/máx de listas:         {min(lengths)} / {max(lengths)}")
    assert min(lengths) == N_RECOMMENDATIONS, \
        f"Hai playlists con menos de {N_RECOMMENDATIONS} recomendacións!"
    print("OK: todas las playlists tienen exactamente 500 recomendaciones.")
