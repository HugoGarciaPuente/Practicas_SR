## Código pra evaluar as 500 recomendacións en .json pra cada playlist.

import json
import math
import numpy as np

# Importa a API pública de rank_eval (revisión sobre as nosas métricas)
from rank_eval import Qrels, Run, evaluate

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


# Rutas
RECS_PATH = "recommendations_per_playlist.json"
EVAL_PATH = "../spotify_test_playlists/test_eval_playlists.json"

#  Cargamos os datos
with open(RECS_PATH, "r", encoding="utf-8") as f:
    recs_data = json.load(f)

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_playlists = json.load(f)["playlists"]

# Obter ambos diccionarios a comparar
# pid → recomendaciones
pid_to_recs = {
    pl["pid"]: pl["recommendations"]
    for pl in recs_data
}

# pid → ground truth (TODAS as tracks do JSON) (seguimos usando set para as nosas métricas
# só importa se esta na playlist ou non na orde, obtemos O(1))
pid_to_gt = {
    pl["pid"]: {t["track_uri"] for t in pl["tracks"]}
    for pl in eval_playlists
}

# EVALUACIÓN

r_precisions = []
ndcgs = []
clicks = []

missing_pids = 0 # playlist do ground_truth sen a nosa recomendación

for pid, gt_tracks in pid_to_gt.items():
    # se non recomendamos pra a playlist saltamola
    if pid not in pid_to_recs:
        missing_pids += 1
        continue

    recs = pid_to_recs[pid]

    r_precisions.append(r_precision(recs, gt_tracks))
    ndcgs.append(ndcg(recs, gt_tracks))
    clicks.append(recommended_songs_clicks(recs, gt_tracks))

# RESULTADOS

print("Evaluation results")
print("===================")
print(f"Playlists evaluadas: {len(r_precisions)}")
print(f"Playlists sin recs:  {missing_pids}")
print()
print(f"R-Precision: {np.mean(r_precisions):.4f}")
print(f"NDCG:        {np.mean(ndcgs):.4f}")
print(f"Clicks:      {np.mean(clicks):.2f}")

#  Preparar estructuras pra rank_eval 
# rank_eval é unha librería externa que implementa as mesmas métricas 
# independentemente. Verificamos que a nosa implementación é correcta,
# comparando os seus resultados agregados cos nosos.

# Necesita dous obxectos:
#  Qrels: o ground truth (qué cancións son relevantes para cada playlist)
#  Run:   as nosas recomendacións (qué lle recomendamos a cada playlist)
qrels = Qrels()
run = Run()

# Listas auxiliares
q_ids = []          # identificadores de query (pid en string)
qrels_docs = []     # por query: lista de doc_ids relevantes
qrels_scores = []   # por query: lista de puntuacións de relevancia (1 = relevante)
run_docs = []       # por query: lista de doc_ids recomendados (en orde)
run_scores = []     # por query: scores ficticios decrecentes para indicar a orde


# Usamos só os pids que teñen recomendacións (coma antes)
for pid, gt_set in pid_to_gt.items():
    if pid not in pid_to_recs:
        continue

    recs = pid_to_recs[pid]
    qid = str(pid)  # rank_eval trabaja con ids en string

    q_ids.append(qid)
    # Para qrels usamos a lista de relevantes (orden irrelevante)
    qrels_docs.append(list(gt_set))
    qrels_scores.append([1] * len(gt_set))  # relevancia binaria

    # Para run usamos ranking tal cual; scores ficticios  decrecientes pra reflectir orden
    # posicion 1 score = 500
    run_docs.append(recs)
    run_scores.append([len(recs) - i for i in range(len(recs))])

# Engadimos todo aos obxectos
qrels.add_multi(q_ids=q_ids, doc_ids=qrels_docs, scores=qrels_scores)
run.add_multi(q_ids=q_ids, doc_ids=run_docs, scores=run_scores)

# Escollemos un k representativo para ndcg@k (lonxitude das recomendaciones asumese constante)
# Se as lonxitudes varían, poderíase escoller max(...) ou un valor fixo como 500
representative_recs_len = 0
if len(run_docs) > 0:
    representative_recs_len = len(run_docs[0])
else:
    representative_recs_len = 500  # fallback

#  COMPARISON: evaluar con rank_eval 


# Pedimos rPrecision ( promedio sobre queries) e ndcg@k con k = representative_recs_len
metrics_to_compute = ["r-precision", f"ndcg@{representative_recs_len}"]

results = evaluate(qrels, run, metrics_to_compute) #devolve diccionario da métrica, valor medio

# Extraemos valores agregados
lib_rp = results.get("r-precision", 0.0)
lib_ndcg = results.get(f"ndcg@{representative_recs_len}", 0.0)

diff_rp = []
diff_ndcg = []

for pid, gt_set in pid_to_gt.items():
    if pid not in pid_to_recs:
        continue

    recs = pid_to_recs[pid]

    my_rp = r_precision(recs, gt_set)
    my_n = ndcg(recs, gt_set)

    # Comparamos coas nosas métricas por playlist co valor agregado de rank_eval
    diff_rp.append(abs(my_rp - lib_rp))
    diff_ndcg.append(abs(my_n - lib_ndcg))

# Comparamos a diferenza absoluta entre o noso valor por playlist e o valor
# agregado (media) que devolve rank_eval. Idealmente as diferenzas deberían
# ser case cero, o que confirmaría que a nosa implementación é correcta.

print("Metric comparison")
print()
print(f"R-Precision mean abs diff: {np.mean(diff_rp):.10f}")
print(f"NDCG mean abs diff:        {np.mean(diff_ndcg):.10f}")
print()
print("Max differences")
print(f"R-Precision max diff: {np.max(diff_rp):.10f}")
print(f"NDCG max diff:        {np.max(diff_ndcg):.10f}")

