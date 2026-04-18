# Spotify Playlist Continuation — MPD Challenge (SVD)

Sistema de recomendación automática de canciones para el *Million Playlist Dataset (MPD) Challenge* de Spotify. Implementa dos enfoques de **filtrado colaborativo basado en factorización matricial SVD**: descomposición conjunta train+test y proyección de test sobre el espacio latente del train.

---

## Estructura del proyecto

```
.
├── matrix_create.py              # Construye la matriz train (playlist × track)
├── test_matrix_create_00.py      # Construye la matriz test en el mismo espacio de columnas
├── svd_both.py                   # SVD sobre la matriz conjunta train+test
├── svd_proyect.py                # SVD sobre train; proyección de test al espacio latente
├── evaluation_2.py               # Evalúa R-Precision, NDCG y Clicks
└── generar_submission_2.py       # Exporta el CSV de submission al formato del challenge
```

---

## Decisiones de diseño

### Representación matricial
Ambos enfoques parten de la misma **matriz binaria playlist × track** construida desde el dataset de entrenamiento. Las matrices train y test comparten el mismo espacio de columnas (índices de tracks), lo que garantiza coherencia al calcular scores en el espacio latente.

### SVD sobre train+test (`svd_both.py`)
Se concatenan verticalmente las matrices train y test en una única matriz `R_full` y se aplica SVD truncada con `scipy.sparse.linalg.svds`. Las filas de `U` correspondientes a las playlists de test se extraen directamente para calcular los scores:

```
scores = (U_test · Σ) · Vᵀ
```

Este enfoque permite que las playlists de test influyan en los factores latentes aprendidos, pero recalcula el SVD completo para cada valor de K.

### SVD con proyección (`svd_proyect.py`)
El SVD se calcula únicamente sobre la matriz de entrenamiento. Las playlists de test se proyectan al espacio latente aprendido usando la pseudoinversa de la matriz de ítems:

```
U_test = R_test · V · Σ⁻¹        (proyección, ec. 4)
scores = (U_test · Σ) · Vᵀ       (scoring,    ec. 5)
```

Este enfoque es más fiel a un escenario de producción real: el modelo se entrena offline y las nuevas playlists se proyectan sin reentrenar. Además, al procesar el test en bloques densos, el consumo de memoria está más controlado.

### Número de características latentes K
Se evalúan K ∈ {10, 50, 100, 200} para estudiar el trade-off entre capacidad de representación y coste computacional. Los valores se procesan en bucle, recalculando el SVD para cada K.

### Procesamiento en batches
Ambos scripts procesan las playlists de test en bloques de 50 (`BATCH_SIZE = 50`) para controlar el uso de RAM al materializar la matriz de scores.

### Fallback por popularidad
Cuando el SVD no genera suficientes candidatos para llegar a 500 recomendaciones (p.ej. playlists de cold-start sin ningún track seed), se rellena con los tracks más populares del training (`top_500_tracks.json`), excluyendo siempre los ya presentes en la playlist.

---

## Ejecución

> **Prerequisitos:** Python 3.8+, `numpy`, `scipy`.

### 1. Construir las matrices

```bash
python matrix_create.py            # Genera playlist_track_matrix.npz y track_to_col.json
python test_matrix_create_00.py    # Genera test_matrix.npz y row_to_pid_test.json
```

### 2. Generar recomendaciones

```bash
# SVD sobre train+test
python svd_both.py     # → recommendations_svd1_k{10,50,100,200}.json

# SVD con proyección de test
python svd_proyect.py  # → recommendations_svd_proj_k{10,50,100,200}.json
```

### 3. Evaluar

```bash
python evaluation_2.py
```

### 4. Generar submission

Editar `INPUT_JSON` en `generar_submission_2.py` para seleccionar el modelo y k deseados, luego:

```bash
python generar_submission_2.py   # → submission.csv
```

---

## Resultados

### SVD train+test (`svd_both.py`)

| K   | R-Precision | NDCG   | Clicks |
|-----|-------------|--------|--------|
| 10  | 0.0802      | 0.2066 | 11.05  |
| 50  | 0.1189      | 0.2752 |  7.10  |
| 100 | 0.1268      | 0.2851 |  6.44  |
| 200 | 0.1289      | 0.2848 |  5.84  |

### SVD con proyección (`svd_proyect.py`)

| K   | R-Precision | NDCG   | Clicks |
|-----|-------------|--------|--------|
| 10  | 0.0797      | 0.2041 | 11.83  |
| 50  | 0.1185      | 0.2726 |  8.22  |
| 100 | 0.1262      | 0.2824 |  7.51  |
| 200 | 0.1283      | 0.2820 |  7.14  |

---

## Conclusiones

`svd_both` supera sistemáticamente a `svd_proj` en las tres métricas para todo K, lo que es esperable: al incluir las playlists de test en el SVD, sus patrones influyen directamente en los factores latentes aprendidos. El salto más grande de rendimiento se produce de K=10 a K=50 (~0.04 de R-Precision en ambos enfoques), mientras que de K=100 a K=200 la mejora es marginal (~0.002), lo que sugiere que el modelo satura en torno a K=100. La diferencia en Clicks entre enfoques es especialmente pronunciada (5.84 vs 7.14 en K=200), indicando que `svd_proj` ordena peor las primeras recomendaciones aunque recupere cantidades similares de tracks relevantes. Comparado con la iteración anterior, el mejor SVD (`svd_both` K=200, R-Precision=0.1289) queda por debajo del mejor item-based (K=150, R-Precision=0.1477) y del mejor user-based (K=150, R-Precision=0.1506), por lo que los enfoques de vecindario resultan más competitivos en este dataset con los hiperparámetros evaluados.

---

## Métricas

- **R-Precision**: fracción de tracks relevantes recuperados en los primeros R puestos (R = tamaño del ground truth).
- **NDCG**: mide la calidad del ranking, penalizando logarítmicamente las recomendaciones relevantes que aparecen en posiciones bajas.
- **Clicks**: número de bloques de 10 canciones que un usuario tendría que revisar antes de encontrar al menos una recomendación relevante.

---

## Autores

Manuel David Barreiro Rodríguez · Carlos Hugo García Puente · Alejandro Varela Vázquez  
Universidade da Coruña

Max differences
R-Precision max diff: 0.7934940158
NDCG max diff:        0.7185838036
