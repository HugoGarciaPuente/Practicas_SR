## Código para transformar as recomendacións por playlist en .json ao formato 
## do challenge One Million Playlist

import json

# Datos do equipo

MEMBERS   = "Manuel David Barreiro Rodríguez, Carlos Hugo García Puente, Alejandro Varela Vázquez"   
EMAILS    = "david.barreiro2@udc.es, c.hugo.gpuente@udc.es, alejandro.varela.vazquez1@udc.es"

INPUT_JSON = "recommendations_svd1_k10.json"   # o recommendations_sen proxectar e 10 características latentes
#INPUT_JSON = "recommendations_svd_proj_k10.json"  # o recommendations proxectar e 10 características latentes
OUTPUT_CSV  = "submission.csv"

ROW_TO_PID_PATH = "row_to_pid_test.json"

with open(ROW_TO_PID_PATH, "r", encoding="utf-8") as f:
    row_to_pid = json.load(f)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    recommendations = json.load(f)

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:

    # Liña de info do equipo
    f.write(f"{MEMBERS}, {EMAILS}\n")
    f.write("\n")

    # Unha fila por playlist
    for entry in recommendations:
        pid = row_to_pid[str(entry["pid"])]
        uris = entry["recommendations"]
        # Extraemos só o ID final, eliminando o prefixo "spotify:track:"
        track_ids = [uri.split(":")[-1] for uri in uris]
        line = str(pid) + ", " + ", ".join(track_ids)
        f.write(line + "\n")

print(f"Submission generada: {OUTPUT_CSV}")
print(f"  · Playlists escritas: {len(recommendations)}")
