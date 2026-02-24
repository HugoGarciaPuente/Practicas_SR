## Creación do top500 cancións máis populares, realmente creánse 500 cancións + 
## len(playlist máis longa do dataset) máis populares, para poder recomendar 500 
## cancións populares a cada playlist.

import json
import numpy as np
from scipy.sparse import load_npz
import os

# Supoñemos que temos no working directory a matriz e o metadata de columna-canción

# Rutas
MATRIX_PATH = "playlist_track_matrix.npz"
TRACK_MAP_PATH = "track_to_col.json"  
OUTPUT_PATH = "top_500_tracks.json"

# Carga da matriz
matrix = load_npz(MATRIX_PATH)

print("Matriz cargada")
print("Shape:", matrix.shape)
print("NNZ:", matrix.nnz)

# Número de cancións por playlist
playlist_sizes = np.asarray(matrix.sum(axis=1)).ravel()

# Tamaño da playlist máis grande
max_playlist_size = int(playlist_sizes.max())

print("Tamaño máximo da playlist:", max_playlist_size)

# Popularidade de cada canción
track_counts = np.asarray(matrix.sum(axis=0)).ravel() # aplana a un vector 1D
# track_counts[i], ten o número de playlist nas que esta a canción i.

# Top (500 + max_playlist_size) cancións
top_k = 500 + max_playlist_size
top_indices = np.argsort(track_counts)[-top_k:][::-1] #Ordena por popularidade as top500k

# Cargamos o metadata das cancións
with open(TRACK_MAP_PATH, "r") as f:
    track_to_col = json.load(f)

# Columna -> cancións
col_to_track = {int(v): k for k, v in track_to_col.items()}

# Obtención do top500+
top_tracks = [
    {
        "rank": i + 1,
        "track_uri": col_to_track[idx],
        "playlist_count": int(track_counts[idx])
    }
    for i, idx in enumerate(top_indices)
]

# Gardado en json pra evaluar as métricas
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(top_tracks, f, indent=2)

print(f"Top {top_k} tracks guardado en {OUTPUT_PATH}")
