import zipfile
import json
from scipy.sparse import csr_matrix, save_npz

zip_path = "spotify_test_playlists.zip"
TRACK_TO_COL_PATH = "track_to_col.json"   # reutilizamos el espacio de columnas del train

rows = []
cols = []
values = []

with open(TRACK_TO_COL_PATH) as f:
    track_to_col = json.load(f)

with zipfile.ZipFile(zip_path, "r") as zipf:
    for file in zipf.namelist():
        if not file.endswith(".json"):
            continue
        with zipf.open(file) as f:
            if file != "test_input_playlists.json":
                continue
            data = json.loads(f.read())
            playlists = data["playlists"]

            # Ordenar por pid: fila 0 = pid más bajo, fila 1 = siguiente, etc.
            playlists.sort(key=lambda x: x["pid"])

            for row, playlist in enumerate(playlists):
                for track in playlist["tracks"]:
                    track_uri = track["track_uri"]

                    col = track_to_col[track_uri]

                    rows.append(row)
                    cols.append(col)
                    values.append(1)

            row_to_pid = {i: pl["pid"] for i, pl in enumerate(playlists)}
            n_playlists = len(playlists)

n_tracks = len(track_to_col)


matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(n_playlists, n_tracks)
)

save_npz("test_matrix.npz", matrix)

with open("row_to_pid_test.json", "w") as f:
    json.dump(row_to_pid, f)


print("Matriz creada")
print("Playlists:", n_playlists)
print("Tracks únicos en train (espacio de columnas):", n_tracks)
print("NNZ:", matrix.nnz)
