# RoPE — Rotary Position Embeddings

## Was ist es

RoPE ist eine Möglichkeit, einem Sprachmodell mitzuteilen, *wo*
sich jedes Wort in einem Satz befindet. Ohne RoPE sieht das Modell
alle Wörter gleichzeitig und hat keine Ahnung, welches Wort zuerst
kam. RoPE versieht jedes Wort mit seiner Position, indem es ihm
eine winzige Rotation verpasst. Wörter am Anfang bekommen eine
kleine Drehung. Wörter weiter hinten im Satz bekommen eine größere
Drehung. Das Modell kann erkennen, wie stark zwei Wörter rotiert
sind, und daraus ihre Distanz ableiten.

## Wo wird es eingesetzt

RoPE befindet sich innerhalb des Attention-Layers. Genauer gesagt
wird es auf die Query- und Key-Vektoren angewendet, unmittelbar vor
dem Skalarprodukt, das bestimmt, wie viel Aufmerksamkeit zwei
Wörter einander schenken sollen.

```
Eingabe-Token
  → Embedding (Wortbedeutungen)
    → Attention (hier passiert RoPE)
      → Ausgabe des Transformer-Blocks
```

## Warum wird es verwendet

Vor RoPE nutzte man andere Tricks, um Wortpositionen zu markieren.
Manche addierten Positionsnummern zu den Wortvektoren. Andere
ließen das Modell die Position von Grund auf lernen. Beides
funktionierte, hatte aber Grenzen. Gelernte Positionen kamen mit
Sätzen, die länger waren als die Trainingsdaten, nicht zurecht.
Addierte Positionsnummern erfassten die relative Distanz zwischen
Wörtern nicht gut.

RoPE löst beide Probleme. Es erfasst die relative Distanz perfekt.
Wort fünf und Wort sieben liegen immer zwei Schritte auseinander,
egal ob sie am Anfang oder in der Mitte eines langen Absatzes
stehen. Und RoPE kommt mit jeder Satzlänge zurecht, selbst wenn das
Modell auf kürzeren Sätzen trainiert wurde. Deshalb setzen LLaMA,
Mistral und Qwen alle auf RoPE.

## Wann wurde es erfunden

RoPE wurde 2021 von einem Forscherteam in einem Paper namens
RoFormer veröffentlicht. Es dauerte ein paar Jahre, bis es sich
durchsetzte, aber bis 2023 hatte jedes große Open-Source-
Sprachmodell auf RoPE umgestellt.

## Wie es einfach erklärt funktioniert

Stell dir eine Uhr mit nur einem Zeiger vor. An Position null zeigt
der Zeiger senkrecht nach oben. An Position eins dreht sich der
Zeiger ein wenig. An Position zwei dreht er sich noch etwas weiter.
Jede Position bekommt einen einzigartigen Winkel. Das Modell
speichert diese Winkel als Kosinus- und Sinus-Werte, sodass es sie
während des Trainings nie berechnen muss.

Jedes Wort hat nun ein geheimes Zahlenpaar. RoPE nimmt dieses Paar
und rotiert es um den Winkel der jeweiligen Position. Nach der
Rotation haben zwei Wörter, die nah beieinanderstehen, ähnliche
Rotationen. Zwei weit auseinanderliegende Wörter haben sehr
unterschiedliche Rotationen. Wenn Attention das Skalarprodukt
zwischen einer Query und einem Key betrachtet, hängt das Ergebnis
davon ab, wie weit sie auseinanderliegen. Nicht von ihrer absoluten
Position.

## Ein kleines Codebeispiel

```python
import torch
import math

# RoPE für ein winziges Modell mit 4 Dimensionen einrichten
d_model = 4
max_seq_len = 16
theta = 10000.0

dim_indices = torch.arange(0, d_model, 2).float()
inv_freq = 1.0 / (theta ** (dim_indices / d_model))
positions = torch.arange(max_seq_len).float()
freqs = torch.outer(positions, inv_freq)
emb = freqs.repeat_interleave(2, dim=-1)

cos_cached = emb.cos()
sin_cached = emb.sin()

# Angenommen, wir haben einen Query-Vektor für ein Wort an Position 0
q = torch.tensor([0.8, 0.3, -0.5, 0.2])

seq_len = 4
cos = cos_cached[:seq_len]
sin = sin_cached[:seq_len]

# Rotation für Position 0 anwenden
rotated = q * cos[0] + torch.tensor([-0.3, 0.8, -0.2, -0.5]) * sin[0]
print(f"Position 0: {rotated.tolist()}")

# Rotation für Position 2 anwenden
rotated = q * cos[2] + torch.tensor([-0.3, 0.8, -0.2, -0.5]) * sin[2]
print(f"Position 2: {rotated.tolist()}")

print()
print("Same word at different positions gets different rotations.")
print("The model uses this difference to understand word order.")
```

## Was man sich merken sollte

RoPE rotiert Vektoren. Der Rotationswinkel hängt von der Position
ab. Das Skalarprodukt zwischen zwei rotierten Vektoren hängt nur
davon ab, wie weit sie auseinanderliegen. Genau das sollte
Attention interessieren. Nicht wo die Wörter stehen. Sondern wie
weit sie voneinander entfernt sind.

RoPE ist kostenlos. Keine gelernten Parameter. Kein zusätzlicher
Speicherbedarf. Kein Geschwindigkeitsverlust. Es funktioniert für
Sequenzen jeder Länge. Jedes moderne Sprachmodell nutzt es.
