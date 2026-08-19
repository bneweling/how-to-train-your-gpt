# Cheatsheet: Alles Wichtige auf einen Blick

## Modellgrößen

| Modell | Layer | d_model | Heads | Head Dim | Parameter |
|---|---|---|---|---|---|
| Tiny (unser Standard) | 4 | 256 | 4 | 64 | 17.1M |
| GPT-2 Small | 12 | 768 | 12 | 64 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 64 | 350M |
| GPT-2 Large | 36 | 1280 | 20 | 64 | 774M |
| GPT-3 1.3B | 24 | 2048 | 32 | 64 | 1.3B |
| GPT-3 6.7B | 32 | 4096 | 32 | 128 | 6.7B |
| GPT-3 175B | 96 | 12288 | 96 | 128 | 175B |
| LLaMA 7B | 32 | 4096 | 32 | 128 | 7B |
| LLaMA 13B | 40 | 5120 | 40 | 128 | 13B |
| LLaMA 70B | 80 | 8192 | 64 | 128 | 70B |

## Formel für die Parameteranzahl

Für unser Modell mit SwiGLU und Weight Tying:

```
Embedding:        vocab_size × d_model
Pro Block QKV:    3 × d_model × d_model
Pro Block Output: d_model × d_model
Pro Block SwiGLU: 3 × d_model × (4 × d_model)
Pro Block Norms:  2 × d_model
LM Head:          0 (Weight Tying mit dem Embedding)
```

Für das GPT-2-Small-Äquivalent (768 Dimensionen, 12 Layer, 50257 Vokabular):
```
152M gesamt = 38.6M (Embedding) + 113.3M (12 Blöcke) + 768 (Norm)
```

## Trainings-Hyperparameter

| Parameter | Unser Standard | Bereich | Hinweise |
|---|---|---|---|
| Learning Rate | 3e-4 | 1e-5 bis 5e-4 | Niedriger für Fine-Tuning (1e-5 bis 5e-5) |
| Weight Decay | 0.1 | 0.01 bis 0.3 | Nur bei 2D+-Parametern. Nicht bei Norms/Biases |
| Betas | (0.9, 0.95) | : | LLaMA-Standardwerte. Nicht ändern |
| Epsilon | 1e-8 | : | Muss nie angepasst werden |
| Warmup-Steps | 2000 | 500 bis 5000 | ~5 % der Gesamtschritte |
| Max Steps | 100K | Abhängig von den Daten | Mehr Daten = mehr Steps |
| Batch Size | 8 (× 4 Accum) | So groß wie möglich | Effektive Batch Size = 32 |
| Grad Clip | 1.0 | 0.5 bis 2.0 | 1.0 ist Standard |
| Dropout | 0.1 | 0.0 bis 0.3 | Höher = mehr Regularisierung |

## Sampling-Parameter

| Parameter | Standard | Anwendungsfall |
|---|---|---|
| temperature | 0.8 | Allgemein. Niedriger = fokussiert (0.3-0.5), höher = kreativ (1.2-1.5) |
| top_k | 50 | Eliminiert unsinnige Token. 0 = deaktiviert |
| top_p | 0.9 | Adaptiver Cutoff. 1.0 = deaktiviert |
| max_new_tokens | 100 | Abhängig von der Aufgabe. Länger für Geschichten, kürzer für Antworten |

## Wichtige Formeln

### Embedding
```
output = self.embed(token_ids)
Keine Skalierung bei RoPE (LLaMA-Konvention)
```

### RoPE-Rotationswinkel
```
θ_i = p / (10000^(2i / d_head))
cos_cached = cos(θ_i)  für alle p und i
sin_cached = sin(θ_i)  für alle p und i
x_rotated = x * cos + rotate_half(x) * sin
```

### Attention
```
Q = input @ W_q   [batch, heads, seq, head_dim]
K = input @ W_k   [batch, heads, seq, head_dim]
V = input @ W_v   [batch, heads, seq, head_dim]

Q_rot = RoPE(Q, seq_len)
K_rot = RoPE(K, seq_len)

scores = Q_rot @ K_rot^T / sqrt(head_dim)   [batch, heads, seq, seq]
scores = masks(scores)                        (kausal: obere Dreiecksmatrix = -inf)
weights = softmax(scores, dim=-1)              (Zeilensumme = 1.0)
output = weights @ V                          [batch, heads, seq, head_dim]

Heads konkatenieren → [batch, seq, d_model]
output = concat @ W_o
```

### RMSNorm
```
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

### SwiGLU-FFN
```
h = SiLU(input @ W1)     [batch, seq, 4×d_model]
g = input @ W2           [batch, seq, 4×d_model]
output = (h * g) @ W3     [batch, seq, d_model]
```

### Cross-Entropy-Loss
```
loss = -log(P_model(true_token))
Für ein zufälliges Modell: loss ≈ ln(vocab_size) = ln(50257) ≈ 10.82
```

### AdamW-Update
```
momentum = β1 × momentum + (1 - β1) × gradient
velocity = β2 × velocity + (1 - β2) × gradient^2

m_hat = momentum / (1 - β1^step)
v_hat = velocity / (1 - β2^step)

weight = weight * (1 - lr * weight_decay)     (entkoppelt)
weight = weight - lr * m_hat / (sqrt(v_hat) + ε)
```

### Cosine-Warmup-Schedule
```
if step < warmup:
    lr = max_lr * step / warmup
else:
    progress = (step - warmup) / (total - warmup)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

### Gradient Clipping
```
total_norm = sqrt(Σ grad_i^2)
if total_norm > max_norm:
    scale = max_norm / total_norm
    grad_i *= scale für alle i
```

## Speicherbedarf (ungefähr)

| Modellgröße | bfloat16-Weights | Optimizer (AdamW) | Gesamt (ohne Batch) |
|---|---|---|---|
| 17M (Tiny) | 34 MB | 136 MB | 170 MB |
| 152M (GPT-2 Small) | 304 MB | 1.2 GB | 1.5 GB |
| 7B (LLaMA) | 14 GB | 56 GB | 70 GB |
| 70B (LLaMA) | 140 GB | 560 GB | 700 GB |

Optimizer-States = 2 × params × 4 Bytes (float32). Gradients = params × 4 Bytes.
Gesamter Trainingsspeicher ≈ 2×weights + 2×optimizer + gradients + activations.
Bei bfloat16-Weights: ~8 Bytes pro Parameter für vollständiges Training.

## Shape-Konventionen

| Komponente | Input-Shape | Output-Shape |
|---|---|---|
| Tokenizer | Text | [batch, seq] |
| Embedding | [batch, seq] | [batch, seq, d_model] |
| RoPE | [batch, heads, seq, head_dim] | Gleich |
| Attention (intern Q/K/V) | [batch, heads, seq, head_dim] | Gleich |
| Attention (extern) | [batch, seq, d_model] | [batch, seq, d_model] |
| RMSNorm | [batch, seq, d_model] | Gleich |
| SwiGLU | [batch, seq, d_model] | Gleich |
| TransformerBlock | [batch, seq, d_model] | Gleich |
| LM Head | [batch, seq, d_model] | [batch, seq, vocab_size] |
| Loss | logits + targets | Skalar |

## Wie ein guter Loss aussieht

```
10.8: Zufällig. Modell weiß noch nichts
9.0:  Beginnt, Wortfrequenzen zu lernen
7.0:  Lernt grundlegende Grammatik
5.0:  Kohärente Phrasen entstehen
3.0:  Passable Sätze. Macht noch Fehler
2.0:  Gutes Modell. Plausibler Text
1.5:  Sehr gut. Nahe an Produktionsqualität
<1.0: Overfitting oder Auswendiglernen (bei kleinen Datensätzen)
```

## Datensatzgrößen

| Dataset | Dokumente | Token | Größe |
|---|---|---|---|
| WikiText-103 | 28,475 | 103M | 516 MB |
| BookCorpus | ~11,000 | 985M | 5 GB |
| C4 | 364M | 156B | 305 GB |
| The Pile | : | 825B | 825 GB |

## Häufige Fehlermeldungen

| Fehler | Bedeutung |
|---|---|
| `CUDA out of memory` | Batch zu groß oder Modell zu groß. Batch Size reduzieren oder Gradient Accumulation verwenden |
| `nan in loss` | Learning Rate zu hoch oder Gradients explodiert. LR senken oder Gradient Clipping hinzufügen |
| `loss = 10.82` | Modell ist zufällig. Normal bei Step 0. Bleibt es dort, Optimizer und Loss-Funktion prüfen |
| `size mismatch` | Shape-Fehler. Batch-/Seq-/Head-Dimensionen prüfen. Häufiger Bug bei reshape/permute |
| `weights_only load failed` | PyTorch 2.6+ benötigt `weights_only=False` für Checkpoints mit benutzerdefinierten Klassen |

## Wichtige Dateien im Repo

```
📦 how-to-train-your-gpt/
├── main.py                         ← Training in einer einzigen Datei. Diese ausführen.
├── requirements.txt                ← Abhängigkeiten
├── chapters/                       ← Lehrbuch mit 12 Kapiteln
├── notebooks/                      ← 8 Kapitel-Notebooks + Attention-Visualisierung + Colab
├── fine-tuning/                    ← 7 Dateien zu Fine-Tuning + LoRA-Notebook
├── explanations and examples WIP/  ← 18 Themen-Deep-Dives
│   ├── attention.md                ← Am beliebtesten. 363 Zeilen, durchgerechnetes Beispiel
│   ├── the_complete_story.md       ← 1166 Zeilen. Alles verbunden
│   ├── a_tokens_journey.md         ← 301 Zeilen. Ein Satz wird verfolgt
│   └── ... (15 weitere Themen)
└── README.md
```
