import zipfile
import json
from scipy.sparse import csr_matrix, save_npz

zip_path = "../spotify_train_dataset.zip"

rows = []
cols = []
values = []

track_to_col = {}
next_col = 0

next_row = 0

with zipfile.ZipFile(zip_path, "r") as zipf:
    for file in sorted(zipf.namelist()):  # sorted para orden determinista
        if not file.endswith(".json"):
            continue
        with zipf.open(file) as f:
            data = json.load(f)           # stream en vez de f.read()
            playlists = data["playlists"]
            playlists.sort(key=lambda x: x["pid"])

            for playlist in playlists:
                row = next_row
                next_row += 1

                for track in playlist["tracks"]:
                    track_uri = track["track_uri"]

                    if track_uri not in track_to_col:
                        track_to_col[track_uri] = next_col
                        next_col += 1

                    col = track_to_col[track_uri]

                    rows.append(row)
                    cols.append(col)
                    values.append(1)

n_playlists = next_row
n_tracks = next_col

matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(n_playlists, n_tracks)
)

save_npz("playlist_track_matrix.npz", matrix)

print("Matriz creada")
print("Playlists:", n_playlists)
print("Tracks únicos:", n_tracks)
print("NNZ:", matrix.nnz)

with open("track_to_col.json", "w") as f:
    json.dump(track_to_col, f)
