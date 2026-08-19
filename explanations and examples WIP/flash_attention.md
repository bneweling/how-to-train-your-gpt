# Flash Attention: Attention schnell machen

## Die kurze Antwort

Flash Attention ist keine neue Art von Attention. Es ist dieselbe
Mathematik, nur schneller. Standard-Attention berechnet die vollständige
Matrix aus Q mal K transponiert, wendet dann Softmax an und
multipliziert anschließend mit V. Das erfordert das Lesen und Schreiben
einer riesigen Zwischenmatrix im GPU-Speicher. Flash Attention vermeidet
es, diese Matrix überhaupt zu speichern. Es berechnet Attention in
kleinen Blöcken und hält alles im schnellen On-Chip-Speicher der GPU.
Das Ergebnis ist ein zwei- bis viermal schnelleres Training und eine
zwei- bis viermal schnellere Generierung bei langen Sequenzen.

Jedes bedeutende LLM-Training seit 2022 hat Flash Attention verwendet.
Es ist keine optionale Optimierung. Es ist der Unterschied zwischen
einem Training mit 2048 Token und einem Training mit 32768 Token auf
derselben GPU.

## Wo der Flaschenhals liegt

GPUs verfügen über zwei Arten von Speicher. Der große Speicher, HBM
oder VRAM genannt, speichert alles. Modellgewichte. KV-Caches.
Trainingsdaten. Er ist groß, aber langsam. Ein Lesezugriff darauf
dauert Hunderte von Zyklen. Der kleine Speicher, SRAM genannt, befindet
sich direkt auf dem GPU-Chip. Er ist winzig, aber extrem schnell. Nur
wenige Megabyte. Ein Lesezugriff darauf dauert nur wenige Zyklen.

Die Standardimplementierung von Attention schreibt die vollständige
Attention-Matrix in den HBM. Dann liest sie diese für den Softmax
wieder ein. Dann schreibt sie die Gewichte zurück in den HBM. Dann
liest sie diese für die Multiplikation mit V erneut ein. Jeder dieser
Lese- und Schreibvorgänge ist ein Flaschenhals.

```
Standard-Attention-Speicherfluss:

  Q, K, V im HBM
    → Q, K laden, um Scores zu berechnen
    → Scores-Matrix in den HBM schreiben  (seq_len × seq_len)
    → Scores aus dem HBM für den Softmax lesen
    → Gewichtsmatrix in den HBM schreiben (seq_len × seq_len)
    → Gewichte und V aus dem HBM für die Ausgabeberechnung lesen
    → Ausgabe in den HBM schreiben

  Problem: Die seq_len × seq_len Matrix ist riesig.
  Bei 8192 Token: 8192 × 8192 × 2 Byte = 134 MB nur für die Scores.
  Jeder Transformer-Layer liest und schreibt das.
  96 Layer × 134 MB = ~13 GB HBM-Traffic nur für Attention.
```

Flash Attention schreibt die vollständige Matrix nie. Es verarbeitet
Attention in kleinen Tiles, die vollständig in den SRAM passen. Die
Berechnung erfolgt on-chip. Nur die Ausgabe wird zurück in den HBM
geschrieben.

```
Flash-Attention-Speicherfluss:

  Q, K, V im HBM
    → Ein Tile von Q in den SRAM laden
    → Ein Tile von K in den SRAM laden
    → Scores für dieses Tile-Paar berechnen
    → Softmax online anwenden (inkrementell)
    → Zugehöriges V-Tile laden
    → Gewichtete Values im SRAM akkumulieren
    → Nur das finale Ausgabe-Tile in den HBM schreiben

  Ergebnis: Die seq_len × seq_len Matrix existiert nie im HBM.
  Der Speicherverkehr ist proportional zu seq_len, nicht zu seq_len
  im Quadrat.
```

## Die zwei zentralen Ideen

### Tiling

Anstatt die gesamten Q- und K-Matrizen auf einmal zu verarbeiten,
werden sie in kleine Tiles aufgeteilt. Ein Tile von Q hat 128 Token mal
64 Dimensionen. Ein Tile von K hat 128 Token mal 64 Dimensionen. Die
Attention-Scores zwischen diesen Tiles ergeben 128 mal 128. Das passt
in den SRAM. Jeweils ein Tile-Paar wird verarbeitet. Die Ergebnisse
werden akkumuliert.

Das ist, als würde man ein Buch Seite für Seite lesen, anstatt alle
Seiten auf dem Boden auszubreiten. Man kann immer nur eine Seite in
den Händen halten, aber man kann das gesamte Buch lesen, ohne einen
größeren Tisch zu brauchen.

### Online-Softmax

Softmax wird normalerweise in zwei Durchgängen berechnet. Zuerst wird
der Maximalwert für numerische Stabilität ermittelt. Dann werden die
Exponentialwerte berechnet und durch die Summe geteilt. Das setzt
voraus, dass alle Scores gleichzeitig verfügbar sind.

Online-Softmax berechnet Softmax inkrementell, sobald jedes Tile von
Scores eintrifft. Dabei werden ein laufendes Maximum und eine laufende
Summe geführt. Trifft ein neues Tile mit einem größeren Maximum ein,
werden die bisherigen Ergebnisse neu skaliert. Sobald das letzte Tile
verarbeitet ist, steht die finale Ausgabe fest. Ein zweiter Durchgang
ist nicht nötig.

```
Online-Softmax für Tile i:

  m_new = max(m_old, max(scores_i))
  sum_new = sum_old × exp(m_old - m_new) + sum(exp(scores_i - m_new))
  output = (output_old × sum_old × exp(m_old - m_new)
           + V_i × sum(exp(scores_i - m_new))) / sum_new
  m_old = m_new
  sum_old = sum_new
```

Die Neuskalierung mit `exp(m_old - m_new)` korrigiert die alte Ausgabe,
wenn das Maximum steigt. Das ist mathematisch identisch zur Berechnung
von Softmax über alle Scores auf einmal. Nur eben stückweise
durchgeführt.

## Warum das für die Causal Mask wichtig ist

Flash Attention ist mit einer Causal Mask sogar noch schneller. Tiles
im oberen Dreieck der Attention-Matrix sind null. Es besteht keinerlei
Notwendigkeit, sie überhaupt zu berechnen. Sie werden vollständig
übersprungen. Für eine Sequenz der Länge N muss nur etwa die Hälfte
der Tiles berechnet werden. Die andere Hälfte ist allein aufgrund der
Maske als null bekannt.

```
Causal-Attention-Tile-Muster (N=6 Tiles zu je 128 Token):

            K0  K1  K2  K3  K4  K5
         Q0 ██  ░░  ░░  ░░  ░░  ░░
         Q1 ██  ██  ░░  ░░  ░░  ░░
         Q2 ██  ██  ██  ░░  ░░  ░░
         Q3 ██  ██  ██  ██  ░░  ░░
         Q4 ██  ██  ██  ██  ██  ░░
         Q5 ██  ██  ██  ██  ██  ██

  ██ = dieses Tile berechnen
  ░░ = überspringen (durch die Causal Mask komplett null)

  Berechnete Tiles: 21 von 36 = 58 Prozent der vollständigen Matrix
```

Standard-Attention schreibt weiterhin Nullen für die Tiles im oberen
Dreieck. Flash Attention rührt sie nie an. Es berechnet nur die Tiles
im unteren Dreieck. Bei langen Sequenzen halbiert das den ohnehin
schon reduzierten Rechenaufwand noch einmal.

## Der Geschwindigkeitsunterschied

Für Training mit einer Sequenzlänge von 2048 auf einer A100-GPU.

```
Standard-Attention:     85 ms pro Forward Pass
Flash Attention V1:     24 ms pro Forward Pass  (3,5x schneller)
Flash Attention V2:     19 ms pro Forward Pass  (4,5x schneller)

Speicherverbrauch für Attention-Scores:
Standard-Attention:     64 MB pro Layer
Flash Attention:         0 MB pro Layer (nie materialisiert)

Maximale Sequenzlänge auf einer 40-GB-A100:
Standard-Attention:     ~4096 Token
Flash Attention:        ~16384 Token (4× länger)
```

Der Geschwindigkeitsunterschied wächst mit der Sequenzlänge. Bei 8192
Token kann Flash Attention 8x schneller sein als Standard-Attention.
Die Speicherersparnis bedeutet, dass man auf viermal so langen
Sequenzen trainieren oder die vierfache Batch Size auf derselben GPU
verwenden kann.

## Eine vereinfachte Code-Skizze

Das ist kein Produktionscode. Es zeigt die Struktur. Echtes Flash
Attention ist in CUDA C++ geschrieben und stark für bestimmte
GPU-Architekturen optimiert.

```python
def flash_attention_forward(Q, K, V, causal=True, block_size=128):
    """
    Vereinfachte Flash Attention. Verarbeitet Attention in Blöcken,
    um die vollständige N×N-Attention-Matrix nicht materialisieren zu müssen.
    """
    batch, heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        q_block = Q[:, :, i:i + block_size]
        q_max = torch.full((batch, heads, block_size, 1),
                           float('-inf'), device=Q.device)
        q_sum = torch.zeros(batch, heads, block_size, 1, device=Q.device)
        out_block = torch.zeros_like(q_block)

        j_end = (i + block_size) if causal else seq_len
        for j in range(0, j_end, block_size):
            k_block = K[:, :, j:j + block_size]
            v_block = V[:, :, j:j + block_size]

            scores = (q_block @ k_block.transpose(-2, -1)) * scale

            new_max = torch.maximum(q_max, scores.max(dim=-1, keepdim=True).values)
            correction = torch.exp(q_max - new_max)

            exp_scores = torch.exp(scores - new_max)
            new_sum = correction * q_sum + exp_scores.sum(dim=-1, keepdim=True)

            out_block = (correction * q_sum / new_sum) * out_block
            out_block = out_block + (exp_scores / new_sum) @ v_block

            q_max = new_max
            q_sum = new_sum

        output[:, :, i:i + block_size] = out_block

    return output
```

Das entscheidende Detail: `out_block` wird über die innere Schleife
hinweg direkt akkumuliert, ohne jemals die vollständige Scores-Matrix
zu speichern. Das laufende Maximum und die laufende Summe verfolgen
den Softmax-Zustand. Wenn das Maximum steigt, skaliert der
Korrekturterm die vorherige Ausgabe neu. Sobald alle K-Tiles verarbeitet
sind, besitzt der Block von Q-Tiles seine vollständige
Attention-Ausgabe.

## Kann ich Flash Attention in diesem Projekt verwenden

Ja. Das pip-Paket heißt `flash-attn`. Installiere es und ersetze den
Attention-Forward-Pass durch `flash_attn_func`. Es gibt jedoch zwei
Haken.

Erstens funktioniert es nur auf NVIDIA-GPUs mit Compute Capability 8.0
oder höher. Das bedeutet A100, H100, RTX 3090, RTX 4090. Es
funktioniert nicht auf der CPU, auf Apple MPS oder älteren GPUs.

Zweitens erwartet die API ein leicht anderes Tensor-Layout. Der Input
muss die Form batch seq_len num_heads head_dim haben, wobei die
Sequenzdimension vor der Head-Dimension liegt. Unser Code verwendet
batch num_heads seq_len head_dim. Man muss also transponieren.

```python
# Verwendung von Flash Attention mit unserem Modell
from flash_attn import flash_attn_func

# Unsere Attention (vereinfacht):
qkv = self.qkv_proj(x)
qkv = qkv.reshape(batch, seq, 3, num_heads, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
q, k, v = qkv[0], qkv[1], qkv[2]

# Flash Attention erwartet [batch, seq, heads, head_dim]:
q = q.permute(0, 2, 1, 3)  # [batch, seq, heads, head_dim]
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)

output = flash_attn_func(q, k, v, causal=True)

output = output.permute(0, 2, 1, 3)  # Zurück zu [batch, heads, seq, head_dim]

# Wie gewohnt fortsetzen
output = output.transpose(1, 2).contiguous()
output = output.reshape(batch, seq, d_model)
output = self.out_proj(output)
```

## Was man sich merken sollte

Flash Attention ist dieselbe Attention-Mathematik, nur schneller. Es
schreibt niemals die vollständige N-mal-N-Attention-Matrix in den
GPU-Speicher. Stattdessen verarbeitet es Attention in kleinen Tiles
unter Verwendung von schnellem On-Chip-Speicher. Online-Softmax
berechnet den Softmax inkrementell, ohne einen zweiten Durchgang.

Bei Sequenzen unter 512 Token ist der Unterschied gering. Bei
Sequenzen über 2048 Token ist Flash Attention 3x bis 8x schneller.
Außerdem reduziert es den Speicherverbrauch, da die Attention-Matrix
nicht gespeichert wird, was längere Sequenzen oder größere Batch
Sizes auf derselben GPU ermöglicht.

Jedes produktive LLM-Training seit 2022 verwendet Flash Attention. Für
ernsthafte Arbeit ist das keine optionale Optimierung. Es ist eine
Notwendigkeit.
