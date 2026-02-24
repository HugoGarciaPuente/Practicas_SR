import zipfile
import json
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz #para crear matriz dispersa en memoria 
zip_path = "../spotify_train_dataset.zip"

#Preparamos las partes de la matriz a construir 
rows = []
cols = []
values = []

track_to_col = {} #asigna cad track_uri a una columan numérica 
next_col = 0

with zipfile.ZipFile(zip_path, "r") as zipf:
    #Recorremos cada archivo JSON dentro del zip 
    for file in zipf.namelist():
        if not file.endswith(".json"):
            continue
   
            #Abre el archivo y extrae la lista de playlists 
        with zipf.open(file) as f:
            data = json.loads(f.read())
            playlists = data["playlists"]

            for playlist in playlists:
                pid = playlist["pid"]  #pid marca el índice de fila 
                for track in playlist["tracks"]:
                    track_uri = track["track_uri"] 

                #Indice de columna para cada canción 
                    if track_uri not in track_to_col:
                        track_to_col[track_uri] = next_col
                        next_col += 1

                    col = track_to_col[track_uri]

                    rows.append(pid)
                    cols.append(col)
                    values.append(1)

n_playlists = max(rows) + 1
n_tracks = next_col

#Creación de la matriz dispersa 
matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(n_playlists, n_tracks)
)

save_npz("playlist_track_matrix.npz", matrix)

print("Matriz creada")
print("Playlists:", n_playlists)
print("Tracks únicos:", n_tracks)
print("NNZ:", matrix.nnz)

import json

with open("track_to_col.json", "w") as f:
    json.dump(track_to_col, f)

