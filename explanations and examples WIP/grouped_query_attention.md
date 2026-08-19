# Grouped Query Attention und Multi-Query Attention

## Die kurze Antwort

Standard-Multi-Head-Attention gibt jedem Attention Head seinen eigenen
Query, Key und Value. Grouped Query Attention teilt Key- und
Value-Heads über Gruppen von Query-Heads hinweg. Ein Key-Value-Paar
bedient mehrere Queries. Das verkleinert die Größe des KV-Cache
während der Inference drastisch. Die Modellqualität sinkt kaum.
Multi-Query Attention treibt das auf die Spitze, indem ein einzelner
Key- und Value-Head für alle Query-Heads verwendet wird.

## Wo das eine Rolle spielt

Jedes Mal, wenn das Modell während der Inference ein neues Token
generiert, muss es die Key- und Value-Vektoren für dieses Token im
KV-Cache speichern. Bei Standard-Multi-Head-Attention speichert der
Cache ein K und ein V pro Head pro Layer pro Token.

```
Standard-MHA (12 Heads):
  Jedes neue Token fügt hinzu: 12 K-Vektoren + 12 V-Vektoren
  Cache-Größe für 1000 Token: 24 × 1000 × head_dim Floats

GQA (12 Q-Heads, 4 KV-Heads):
  Jedes neue Token fügt hinzu: 4 K-Vektoren + 4 V-Vektoren
  Cache-Größe für 1000 Token: 8 × 1000 × head_dim Floats
  Speicherersparnis: 3× kleinerer Cache

MQA (12 Q-Heads, 1 KV-Head):
  Jedes neue Token fügt hinzu: 1 K-Vektor + 1 V-Vektor
  Cache-Größe für 1000 Token: 2 × 1000 × head_dim Floats
  Speicherersparnis: 12× kleinerer Cache
```

Die Speicherersparnis ist real. Bei einem Modell mit 70 Milliarden
Parametern, das 4096 Token generiert, kann der KV-Cache mehrere
Gigabyte groß werden. Ihn um das 4-fache oder 8-fache zu verkleinern,
macht den Unterschied zwischen Hineinpassen in den GPU-Speicher und
einem Absturz aus.

## Warum das funktioniert

Brauchen wir wirklich zwölf separate Key- und Value-Heads? Jeder Head
in Standard-MHA hat seine eigene Perspektive auf die Eingabe. Die
Query-Heads brauchen diese Vielfalt. Unterschiedliche Queries suchen
nach unterschiedlichen Mustern. Grammatik-Heads achten auf
Subjekt-Verb-Kongruenz. Semantische Heads achten auf
Bedeutungsbeziehungen. Positions-Heads achten auf die Wortstellung.

Aber die Keys und Values brauchen nicht so viel Vielfalt. Sie
repräsentieren, was jedes Token zu bieten hat. Ein Token bietet
dieselbe grundlegende Information, unabhängig davon, welcher
Query-Head es betrachtet. Das Verb *sat* liefert denselben
semantischen Gehalt, egal ob der Grammatik-Head oder der semantische
Head danach fragt. Wenn Keys und Values über Heads hinweg geteilt
werden, geht nur wenig Information verloren.

GQA findet einen Mittelweg. Wenige KV-Heads liefern genug Vielfalt,
damit die Query-Heads finden, was sie brauchen. Das optimale
Verhältnis liegt bei etwa vier bis acht Query-Heads pro KV-Head.
Darüber hinaus sinkt die Qualität spürbar.

## Die Code-Änderung gegenüber Standard-MHA

Der Unterschied ist minimal. Standard-MHA projiziert auf 3 × d_model
für Q, K und V. GQA projiziert auf d_model × d_model + 2 × kv_heads ×
head_dim.

```python
# Standard-Multi-Head-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Projiziert auf: Q (d_model) + K (d_model) + V (d_model)
        # Alle Heads bekommen ihre eigenen Q, K, V

# Grouped Query Attention
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads

        # Q-Projektion: volle Größe (1 pro Head)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # K- und V-Projektionen: kleiner (1 pro KV-Head)
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
```

Der entscheidende Unterschied: `k_proj` und `v_proj` projizieren auf
`num_kv_heads × head_dim` statt auf `num_heads × head_dim`. Weniger
KV-Projektionen. Weniger Speicher. Schnellere Inference.

### Der Forward Pass

Der Forward Pass braucht einen zusätzlichen Schritt. Die KV-Heads
müssen wiederholt werden, damit sie der Anzahl der Query-Heads
entsprechen.

```python
def forward(self, x, mask=None):
    batch, seq, d_model = x.shape

    # Q vollständig projizieren, K und V mit weniger Heads
    q = self.q_proj(x)
    q = q.reshape(batch, seq, self.num_heads, self.head_dim)
    q = q.permute(0, 2, 1, 3)

    k = self.k_proj(x)
    k = k.reshape(batch, seq, self.num_kv_heads, self.head_dim)
    k = k.permute(0, 2, 1, 3)

    v = self.v_proj(x)
    v = v.reshape(batch, seq, self.num_kv_heads, self.head_dim)
    v = v.permute(0, 2, 1, 3)

    # KV-Heads wiederholen, damit sie den Query-Heads entsprechen
    # Beispiel: 12 Q-Heads, 4 KV-Heads → jeden KV-Head 3-mal wiederholen
    repeat_factor = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=1)
    v = v.repeat_interleave(repeat_factor, dim=1)

    # Ab hier Standard-Attention
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    output = weights @ v

    output = output.transpose(1, 2).contiguous()
    output = output.reshape(batch, seq, d_model)
    return self.out_proj(output)
```

Das `repeat_interleave` in Zeile 25 ist die einzige neue Operation. Es
nimmt die vier KV-Heads und wiederholt jeden davon dreimal, um auf
zwölf zu kommen. Die zwölf Query-Heads können dann Attention auf diese
wiederholten KV-Heads anwenden.

Während der Inference speichert der KV-Cache nur `num_kv_heads` Keys
und Values pro Layer. Die Wiederholung erfolgt zur Laufzeit. Die
Berechnung ist dieselbe wie bei Standard-Attention. Nur der
Speicherbedarf ändert sich.

## Multi-Query Attention: der Extremfall

MQA ist GQA mit `num_kv_heads` auf 1 gesetzt. Ein Key und ein Value
für alle Query-Heads. Der Wiederholungsfaktor entspricht `num_heads`.

```python
# MQA: num_kv_heads = 1
# k_proj projiziert auf nur 1 × head_dim Dimensionen
# v_proj projiziert auf nur 1 × head_dim Dimensionen
# repeat_interleave(num_heads) dupliziert, damit es den Query-Heads entspricht
```

MQA wurde 2019 von Google für Übersetzungsmodelle eingeführt. Es spart
das Maximum an Speicher, aber die Qualität sinkt stärker als bei GQA.
PaLM und Gemini setzen MQA erfolgreich ein, weil ihre Modelle groß
genug sind, dass der Qualitätsverlust durch das Teilen durch den
Qualitätsgewinn aus der Skalierung ausgeglichen wird.

## Welche Modelle was verwenden

| Modell | Architektur | Q-Heads | KV-Heads | Verhältnis |
|---|---|---|---|---|
| GPT-2 | MHA | 12 | 12 | 1:1 |
| GPT-3 | MHA | 96 | 96 | 1:1 |
| LLaMA 1 | MHA | variiert | variiert | 1:1 |
| LLaMA 2 7B/13B | MHA | 32 | 32 | 1:1 |
| LLaMA 2 70B | GQA | 64 | 8 | 8:1 |
| LLaMA 3 8B | GQA | 32 | 8 | 4:1 |
| LLaMA 3 70B | GQA | 64 | 8 | 8:1 |
| Mistral 7B | GQA | 32 | 8 | 4:1 |
| Gemma 7B | MHA | 16 | 16 | 1:1 |
| PaLM | MQA | variiert | 1 | variiert |
| Gemini | MQA | variiert | 1 | variiert |

Der Trend ist eindeutig. Kleinere Modelle behalten MHA bei, weil der
KV-Cache ohnehin klein genug ist. Größere Modelle wechseln zu GQA,
weil die Speicherersparnis bei dieser Größenordnung ins Gewicht fällt.
Einige sehr große Modelle gehen für maximale Ersparnis bis zu MQA.

## Wann Sie GQA im eigenen Modell einsetzen sollten

Wenn Ihr Modell unter etwa 13 Milliarden Parametern liegt, ist MHA
völlig ausreichend. Der KV-Cache ist klein. Die Speicherersparnis
durch GQA rechtfertigt nicht den zusätzlichen technischen Aufwand.

Liegt Ihr Modell zwischen 13 und 70 Milliarden Parametern, ist GQA der
Sweet Spot. Verwenden Sie ein Verhältnis von 4:1 oder 8:1. Die
Speicherersparnis ist erheblich. Der Qualitätsverlust ist kaum
messbar.

Hat Ihr Modell über 70 Milliarden Parameter und bedienen Sie viele
gleichzeitige Nutzer, ziehen Sie MQA in Betracht. Die extreme
Speicherersparnis erlaubt es, mehr Nutzer pro GPU zu bedienen. Der
Qualitätsverlust ist spürbar, aber die Wirtschaftlichkeit gewinnt oft.

## Ein vollständiges GQA-Codebeispiel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.repeat = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape

        q = self.q_proj(x)
        q = q.reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.k_proj(x)
        k = k.reshape(batch, seq, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.repeat_interleave(self.repeat, dim=1)

        v = self.v_proj(x)
        v = v.reshape(batch, seq, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.repeat_interleave(self.repeat, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch, seq, self.num_heads * self.head_dim)
        return self.out_proj(output)


# Speichervergleich
d_model = 4096
num_heads = 32
num_kv_heads = 8
seq_len = 2048
head_dim = d_model // num_heads

mha_kv_size = 2 * num_heads * seq_len * head_dim          # K + V
gqa_kv_size = 2 * num_kv_heads * seq_len * head_dim        # K + V

print(f"MHA KV cache per layer: {mha_kv_size * 2 / 1e6:.1f} MB (bfloat16)")
print(f"GQA KV cache per layer: {gqa_kv_size * 2 / 1e6:.1f} MB (bfloat16)")
print(f"Memory savings: {mha_kv_size / gqa_kv_size:.1f}x")

# Für 80 Layer:
print(f"\nFor an 80 layer model:")
print(f"MHA total KV cache: {mha_kv_size * 80 * 2 / 1e9:.2f} GB")
print(f"GQA total KV cache: {gqa_kv_size * 80 * 2 / 1e9:.2f} GB")
```

## Was Sie sich merken sollten

Grouped Query Attention teilt Key- und Value-Heads über Gruppen von
Query-Heads hinweg. Das reduziert die Größe des KV-Cache während der
Inference um das Verhältnis von Query-Heads zu KV-Heads. Ein
Verhältnis von 4:1 oder 8:1 ist Standard. Der Qualitätsverlust ist
minimal. Die Speicherersparnis ist bei großem Maßstab enorm.

Multi-Query Attention ist der Extremfall mit einem einzigen KV-Head
für alle Query-Heads. Maximale Speicherersparnis, aber spürbarer
Qualitätsverlust. Eingesetzt von PaLM und Gemini, wo das Modell groß
genug ist, um das auszugleichen.

Der Code-Unterschied zur Standard-Attention ist minimal. Getrennte
Q-, K- und V-Projektionen mit unterschiedlichen Ausgabegrößen. Eine
repeat-interleave-Operation. Der Rest der Attention-Berechnung ist
identisch.

Für Modelle unter 13 Milliarden Parametern verwenden Sie Standard-MHA.
Für größere Modelle wechseln Sie zu GQA. Die Speicherersparnis macht
den Unterschied zwischen einem Modell, das Nutzer bedienen kann, und
einem Modell, dem nach ein paar hundert Token der GPU-Speicher
ausgeht.
