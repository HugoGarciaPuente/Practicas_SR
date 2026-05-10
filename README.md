# Spotify Playlist Continuation — MPD Challenge (SLIM & FISM)

Sistema de recomendación automática de cancións para o *Million Playlist Dataset (MPD) Challenge* de Spotify. Implementa dous enfoques de **filtrado colaborativo baseado en métodos lineais sparse**: SLIM, que aprende directamente unha matriz de similitude ítem-ítem, e FISM, que a factoriza en dúas matrices de menor rango.

---

## Estrutura do proxecto

```
.
├── slim.py                       # SLIM: aprende S directamente por gradient descent
├── fism.py                       # FISM: factoriza S = P·Qᵀ e optimiza P e Q
├── evaluation_03.py              # Avalía R-Precision, NDCG e Clicks
└── generar_submission_04.py      # Exporta o CSV de submission ao formato do challenge
```

---

## Decisións de deseño

### Representación matricial
Ambos enfoques parten da mesma **matriz binaria playlist × track** X construída desde o dataset de adestramento. X[i, j] = 1 se a playlist i contén o track j. As playlists de test non se inclúen en X — os seus tracks seed úsanse unicamente na fase de inferencia.

Para o trimmed dataset, X ten forma (4.347 × 1.704) cunha densidade de 0.000915, o que reflicte a alta sparsidade típica de datasets de recomendación reais.

### SLIM — Sparse Linear Methods (`slim.py`)
SLIM aprende unha matriz de coeficientes ítem-ítem S ∈ ℝ^(items × items) resolvendo o seguinte problema de optimización:

```
min_S  ‖X − X·S‖²_F  +  λ·‖S‖₁  +  β·‖S‖²_F
s.t.   S ≥ 0
       diag(S) = 0
```

O primeiro termo garante fidelidade de reconstrución: X·S debe aproximar X, é dicir, a playlist debe poder reconstruírse como combinación lineal dos seus ítems. O segundo termo (L1) induce esparsidade en S, facendo que cada ítem só teña relación significativa con uns poucos ítems. O terceiro (L2) evita sobreaxuste penalizando coeficientes grandes.

As restricións impleméntanse mediante **projected gradient descent**:

- `S ≥ 0`: aplícase `relu` tras cada actualización do gradiente.
- `diag(S) = 0`: multiplícase por unha máscara que anula a diagonal, evitando que un ítem se recomiende a si mesmo.

A optimización usa `tf.GradientTape` para diferenciación automática e SGD como optimizador.

**Hiperparámetros empregados:**

| Parámetro | Valor | Rol |
|-----------|-------|-----|
| λ (L1)    | 0.01  | Controla a esparsidade de S |
| β (L2)    | 0.01  | Evita sobreaxuste |
| lr        | 0.01  | Taxa de aprendizaxe (SGD) |
| Epochs    | 200   | Iteracións de adestramento |
| Top-K     | 500   | Recomendacións por playlist |

### FISM — Factored Item Similarity Model (`fism.py`)
FISM estende SLIM descompoñendo S en dúas matrices de menor rango:

```
S = P · Qᵀ     onde P, Q ∈ ℝ^(items × k)
```

Isto reduce drasticamente o número de parámetros respecto a SLIM:

```
SLIM:  items²            = 1.704²       = 2.903.616 parámetros
FISM:  2 × items × k     = 2 × 1.704 × 5 = 17.040 parámetros   (con k = 5)
```

A función de perda é análoga a SLIM pero opera sobre P e Q:

```
min_{P,Q}  ‖X − X·P·Qᵀ‖²_F  +  λ·‖P·Qᵀ‖₁  +  β·‖P·Qᵀ‖²_F
s.t.       diag(P·Qᵀ) = 0
```

A restrición S ≥ 0 rélaxase en FISM porque a factorización con valores arbitrarios de P e Q non garante non-negatividade de forma natural; manter só `diag(S) = 0` é suficiente para evitar a auto-recomendación.

A optimización usa **Adam** en lugar de SGD, xa que en problemas bilineais (P·Qᵀ) os gradientes de P dependen de Q e viceversa, o que pode causar diverxencia con SGD puro. Adam estabiliza as actualizacións normalizando cada gradiente pola súa magnitude histórica. Combínase con **gradient clipping** para maior robustez numérica.

Na fase de inferencia, o score calcúlase eficientemente como:

```
scores = (x_i · P) · Qᵀ
```

evitando materializar S completa, o que permite escalar a datasets máis grandes.

**Hiperparámetros empregados:**

| Parámetro  | Valor | Rol |
|------------|-------|-----|
| k          | 5     | Dimensión do espazo latente |
| λ (L1)     | 0.01  | Esparsidade de S |
| β (L2)     | 0.01  | Evita sobreaxuste |
| lr         | 0.005 | Taxa de aprendizaxe (Adam) |
| Epochs     | 500   | Iteracións de adestramento |
| Grad clip  | 5.0   | Limita a norma dos gradientes para evitar NaN |
| Top-K      | 500   | Recomendacións por playlist |

### Fallback por popularidade
Cando unha playlist de test non ten tracks seed no vocabulario de train (cold-start), recoméndanse os tracks máis frecuentes do adestramento, excluíndo sempre os xa presentes na playlist. Este mesmo mecanismo rechea ata 500 recomendacións cando os scores non xeran suficientes candidatos.

---

## Execución

> **Prerrequisitos:** Python 3.8+, `tensorflow`, `numpy`, `scipy`.

### 1. Xerar recomendacións

```bash
python slim.py    # → recommendations_slim.json
python fism.py    # → recommendations_fism.json
```

### 2. Avaliar

```bash
python evaluation_03.py
```

### 3. Xerar submission

Editar `INPUT_JSON` en `generar_submission_04.py` para seleccionar o modelo desexado, logo:

```bash
python generar_submission_04.py   # → submission.csv
```

---

## Resultados

Ground truth do trimmed dataset: entre 1 e 3 tracks por playlist (media: 1.3), avaliado sobre 29 playlists de test.

### SLIM

| λ (L1) | β (L2) | lr   | Epochs | R-Precision | NDCG   | Clicks |
|--------|--------|------|--------|-------------|--------|--------|
| 0.01   | 0.01   | 0.01 | 200    | 0.0690      | 0.3190 |   9.14 |

### FISM

| k | λ (L1) | β (L2) | lr    | Epochs | Grad clip | R-Precision | NDCG   | Clicks |
|---|--------|--------|-------|--------|-----------|-------------|--------|--------|
| 5 | 0.01   | 0.01   | 0.005 | 500    | 5.0       | 0.0690      | 0.3022 |   7.14 |

### Análise dos resultados

Ambos modelos obteñen a mesma R-Precision (0.0690), o que indica que recuperan un número similar de tracks relevantes entre os primeiros R recomendados. A diferenza principal está no NDCG e nos Clicks:

- **SLIM** acada un NDCG lixeiramente superior (0.3190 vs 0.3022), o que reflicte que ordena mellor as recomendacións relevantes dentro da lista. Isto é coherente co feito de que SLIM aprende S directamente sen restrición de rango, polo que ten maior capacidade de reconstrución.

- **FISM** obtén mellores Clicks (7.14 vs 9.14), o que significa que o usuario atopa antes a primeira recomendación correcta revisando bloques de 10 cancións. A factorización S = P·Qᵀ con k=5 actúa como regularización implícita, xeneralizando mellor en cold-start.

---

## Métricas

- **R-Precision**: fracción de tracks relevantes recuperados nos primeiros R postos (R = tamaño do ground truth). Mide cobertura.
- **NDCG**: mide a calidade do ranking, penalizando logaritmicamente as recomendacións relevantes que aparecen en posicións baixas. Mide orde.
- **Clicks**: número de bloques de 10 cancións que un usuario tería que revisar antes de atopar polo menos unha recomendación relevante. Mide experiencia de usuario.

---

## Autores

Manuel David Barreiro Rodríguez · Carlos Hugo García Puente · Alejandro Varela Vázquez  
Universidade da Coruña
