# Spotify Playlist Continuation — MPD Challenge

Sistema de recomendación automática de canciones para el *Million Playlist Dataset (MPD) Challenge* de Spotify. Implementa dos enfoques de **filtrado colaborativo basado en vecindarios**: user-based e item-based.

---

## Estructura del proyecto

```
.
├── matrix_create.py              # Construye la matriz train (playlist × track)
├── test_matrix_create_00.py      # Construye la matriz test en el mismo espacio de columnas
├── create_neighborhood_item.py   # Calcula vecindario item-based (similitud coseno)
├── create_neighborhood_user.py   # Calcula vecindario user-based (similitud coseno)
├── Recomendate_item.py           # Genera recomendaciones item-based
├── Recomendate_user.py           # Genera recomendaciones user-based
├── evaluation_2.py               # Evalúa R-Precision, NDCG y Clicks
└── generar_submission_2.py       # Exporta el CSV de submission al formato del challenge
```

---

## Decisiones de diseño

### Representación matricial
Ambos enfoques parten de una **matriz binaria playlist × track** construida desde el dataset de entrenamiento. Las matrices train y test comparten el mismo espacio de columnas (índices de tracks), lo que evita cualquier reindexado al calcular similitudes.

### Similitud coseno
Se usa similitud coseno en ambos enfoques porque normaliza por la longitud del vector, siendo robusta ante playlists de tamaños muy distintos. Para el cálculo eficiente:
- **Item-based**: se transpone la matriz train y se opera fila a fila sobre los tracks del test.
- **User-based**: se procesa en batches de 500 playlists para controlar el uso de RAM al densificar la matriz de scores.

### Vecindario precomputado (offline)
Los vecindarios se calculan una sola vez y se guardan en JSON (`neighborhood_item.json`, `neighborhood_user.json`) con hasta **K=150 vecinos**. La fase de recomendación los carga directamente, lo que permite probar distintos valores de k sin recalcular.

### Fallback por popularidad
Cuando el vecindario no genera suficientes candidatos para llegar a 500 recomendaciones, se rellena con los tracks más populares del training (`top_500_tracks.json`), excluyendo siempre los ya presentes en la playlist.

### Valor de K
Se evalúan K ∈ {15, 30, 60, 100, 150} para estudiar el trade-off entre precisión y cobertura del vecindario.

---

## Ejecución

> **Prerequisitos:** Python 3.8+, `numpy`, `scipy`.

### 1. Construir las matrices

```bash
python matrix_create.py            # Genera playlist_track_matrix.npz y track_to_col.json
python test_matrix_create_00.py    # Genera test_matrix.npz y row_to_pid_test.json
```

### 2. Calcular vecindarios

```bash
python create_neighborhood_item.py   # → neighborhood_item.json
python create_neighborhood_user.py   # → neighborhood_user.json
```

### 3. Generar recomendaciones

```bash
python Recomendate_item.py   # → recommendations_item_k{15,30,60,100,150}.json
python Recomendate_user.py   # → recommendations_user_k{15,30,60,100,150}.json
```

### 4. Evaluar

```bash
python evaluation_2.py
```

### 5. Generar submission

Editar `INPUT_JSON` en `generar_submission_2.py` para seleccionar el modelo y k deseados, luego:

```bash
python generar_submission_2.py   # → submission.csv
```

---

## Resultados

### Item-based
 
| K   | R-Precision | NDCG   | Clicks |
|-----|-------------|--------|--------|
| 15  | 0.1307      | 0.2437 | 6.12   |
| 30  | 0.1410      | 0.2709 | 5.90   |
| 60  | 0.1450      | 0.2928 | 5.77   |
| 100 | 0.1466      | 0.3061 | 5.74   |
| 150 | 0.1477      | 0.3141 | 5.72   |
 
### User-based
 
| K   | R-Precision | NDCG   | Clicks |
|-----|-------------|--------|--------|
| 15  | 0.1174      | 0.2391 | 7.50   |
| 30  | 0.1352      | 0.2695 | 6.89   |
| 60  | 0.1452      | 0.2932 | 6.36   |
| 100 | 0.1489      | 0.3088 | 6.06   |
| 150 | 0.1506      | 0.3191 | 5.93   |
 
---

## Métricas

- **R-Precision**: fracción de tracks relevantes recuperados en los primeros R puestos (R = tamaño del ground truth).
- **NDCG**: mide la calidad del ranking, penalizando logarítmicamente las recomendaciones relevantes que aparecen en posiciones bajas.
- **Clicks**: número de bloques de 10 canciones que un usuario tendría que revisar antes de encontrar al menos una recomendación relevante.

---

## Autores

Manuel David Barreiro Rodríguez · Carlos Hugo García Puente · Alejandro Varela Vázquez  
Universidade da Coruña
