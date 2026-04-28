## Código pra evaluar as 500 recomendacións en .json pra cada playlist.

import json
import math
import numpy as np


# Implementación métricas

# Cantas recomendacións nosas son relevantes do 0 ao 1
def r_precision(recs, ground_truth):
    R = len(ground_truth)
    if R == 0:  # Seguridade
        return 0

    top_R = recs[:R]
    hits = sum(1 for r in top_R if r in ground_truth) 

    return hits / R


# Mide a calidade do ranking, non só se está recomendada a canción senón onde está na 
# lista de recomendación
def ndcg(recs, ground_truth):
    dcg = 0.0

    for i, r in enumerate(recs): # recomendacións en orde
        if r in ground_truth:
            dcg += 1 / math.log2(i + 2)  #penalizamos logarítmicamente por desposición

    R = len(ground_truth)
    idcg = sum(1 / math.log2(i + 2) for i in range(R)) # o ideal 

    return dcg / idcg if idcg > 0 else 0 # normalizamos

# Cantos clics tarda en atopar un usario unha recomendación correcta
def recommended_songs_clicks(ranking, relevant_tracks, k=10, max_clicks=50):
    for click in range(max_clicks):
        start = click * k
        end = start + k

        block = ranking[start:end] # 10 cancións do bloque

        if any(track in relevant_tracks for track in block):
            return click

    return max_clicks + 1  # 51


EVAL_PATH       = "test_eval_trimmed.json"

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_playlists = json.load(f)["playlists"]

pid_to_gt = {
    pl["pid"]: {t["track_uri"] for t in pl["tracks"]}
    for pl in eval_playlists
}
gt_sizes = [len(v) for v in pid_to_gt.values()]
print(f"Ground truth — min: {min(gt_sizes)}, max: {max(gt_sizes)}, media: {np.mean(gt_sizes):.1f}")

MODES_ITER2 = ["fism", "slim"] 

print(f"\n{'Modo':>6} | {'R-Precision':>12} | {'NDCG':>8} | {'Clicks':>8} | {'Evaluadas':>10}")
print("-" * 65)

for mode in MODES_ITER2:
        recs_path = f"recommendations_{mode}.json"
        try:
            with open(recs_path, "r", encoding="utf-8") as f:
                recs_data = json.load(f)
        except FileNotFoundError:
            print(f"  {recs_path} no encontrado, saltando...")
            continue

        # Construir mapeo inverso: pid_real → recomendaciones
        pid_to_recs = {
            pl["pid"]: pl["recommendations"]
            for pl in recs_data
        }

        r_precs, ndcgs, clicks_list = [], [], []
        missing = 0

        for pid, gt_tracks in pid_to_gt.items():
            if pid not in pid_to_recs:
                missing += 1
                continue
            recs = pid_to_recs[pid]
            r_precs.append(r_precision(recs, gt_tracks))
            ndcgs.append(ndcg(recs, gt_tracks))
            clicks_list.append(recommended_songs_clicks(recs, gt_tracks))

        print(f"{mode:>6}| {np.mean(r_precs):>12.4f} | "
              f"{np.mean(ndcgs):>8.4f} | {np.mean(clicks_list):>8.2f} | "
              f"{len(r_precs):>10}")
