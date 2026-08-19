# LoRA: Low-Rank Adaptation

## Kurz gesagt

LoRA passt ein Modell per Fine-Tuning an, ohne seine ursprünglichen
Weights zu verändern. Es fügt kleine trainierbare Matrizen in bestimmte
Layer ein. Am Ende hat man das ursprüngliche Modell plus eine winzige
Adapter-Datei. Der Adapter ist meist nur wenige Megabyte groß. Man kann
Adapter für unterschiedliche Aufgaben trainieren und sie sofort
austauschen. Ein Modell. Viele Fähigkeiten.

## Wo es ansetzt

LoRA wird in die linearen Layer des Transformers eingefügt.
Typischerweise in die Query- und Value-Projektionen der Attention und
manchmal in die Feed-Forward-Layer. Die ursprüngliche Weight-Matrix
bleibt eingefroren. LoRA fügt zwei kleine Matrizen hinzu, die von Grund
auf trainiert werden.

```
Ohne LoRA:
  output = input @ W  (W ist groß, 768 × 768, 589,824 params)

Mit LoRA:
  output = input @ W + input @ A @ B
                      ^^^^^^^^^^^^^^^^
                      W ist eingefroren. A und B werden trainiert.
                      A ist 768 × 16 (12,288 params)
                      B ist 16 × 768 (12,288 params)
                      LoRA fügt nur 24,576 params pro Layer hinzu
```

Der Rank r ist klein. Typischerweise 8, 16 oder 32. Unser Beispiel
verwendet 16. Die ursprüngliche Weight-Matrix hat einen Rank von bis zu
768. LoRA beschränkt das Update auf einen niedrigen Rank. Die Hypothese
ist, dass Fine-Tuning-Updates ohnehin einen niedrigen Rank haben. Um
ein vortrainiertes Modell an eine neue Aufgabe anzupassen, braucht man
nicht den vollen Rank. Ein kleiner Unterraum genügt.

## Warum es funktioniert

Vortrainierte Modelle verfügen über reichhaltige Repräsentationen. Die
Weights kodieren bereits die Struktur der Sprache. Fine-Tuning muss dem
Modell nicht erneut Sprache beibringen. Es muss ihm nur ein neues
Verhalten auf Basis des vorhandenen Verständnisses vermitteln. Ein
neues Verhalten auf Basis einer reichhaltigen Repräsentation zu
vermitteln erfordert nur kleine Anpassungen. LoRA erfasst diese kleinen
Anpassungen in kompakter Form.

Der Rank r steuert diesen Tradeoff. Ein höherer Rank bedeutet mehr
Kapazität, um komplexe Muster zu lernen. Aber auch mehr Parameter und
langsameres Training. Für die meisten Instruction-Tuning-Aufgaben
reicht ein Rank von 8 oder 16 aus. Für Aufgaben, die das Erlernen
neuer Formate oder Muster erfordern, kann ein Rank von 32 oder 64
helfen.

## Die Mathematik

Ein vollständiges Fine-Tuning-Update einer Weight-Matrix W ist ΔW. Das
neue Weight ist W + ΔW. ΔW hat dieselbe Shape wie W. Bei einer
768-mal-768-Matrix hat ΔW 589824 unabhängige Parameter.

LoRA approximiert ΔW als das Produkt zweier kleinerer Matrizen A und B.
A hat die Shape 768 mal r. B hat die Shape r mal 768. Ihr Produkt A mal
B hat die Shape 768 mal 768, aber nur 2 mal 768 mal r unabhängige
Parameter. Für r gleich 16 sind das 24576 Parameter. Ein
Kompressionsverhältnis von 24.

```
ΔW ≈ A × B

ΔW[i][j] = sum over k from 0 to r-1: A[i][k] × B[k][j]
```

Während des Trainings ist W eingefroren. Gradients fließen nur durch A
und B. A wird mit zufälligen, normalverteilten Werten initialisiert. B
wird mit Null initialisiert, sodass der Adapter zu Beginn des
Trainings nichts bewirkt. Das Modell verhält sich exakt wie das
Basismodell, bis A und B zu lernen beginnen.

## Der Alpha-Parameter

LoRA hat einen Skalierungsfaktor namens Alpha. Der LoRA-Output wird mit
Alpha geteilt durch Rank skaliert, bevor er zum ursprünglichen Output
addiert wird.

```
output = input @ W + (alpha / r) × (input @ A @ B)
```

Alpha steuert, wie viel Einfluss der Adapter hat. Ein größeres Alpha
bedeutet, dass der Adapter mehr Auswirkung auf den Output hat. Der
Standardwert ist das Doppelte des Rank. Für Rank 16 ist Alpha also
üblicherweise 32.

Ändert man den Rank, bestimmt das Verhältnis von Alpha zu Rank die
effektive Learning Rate des Adapters. Hält man Alpha gleich dem
Doppelten des Rank, bleibt das Verhältnis unabhängig vom Rank bei 2.
Das macht das Hyperparameter-Tuning über verschiedene Rank-Werte hinweg
übertragbar.

## Der Dropout

LoRA-Adapter enthalten meist ein kleines Dropout auf dem Output der
A-Matrix. Der Dropout-Wert liegt bei 0,05 oder 0,1. Das verhindert,
dass der Adapter auf den begrenzten Fine-Tuning-Daten overfittet. Das
Basismodell sorgt für Regularisierung, weil seine Weights eingefroren
sind. Der Adapter benötigt eine eigene, kleine Regularisierung.

## Welche Layer angepasst werden

Die Standardwahl sind Query- und Value-Projektionen in jedem
Attention-Layer. Manche Konfigurationen passen auch die
Output-Projektion und die Feed-Forward-Layer an. Mehr angepasste Layer
bedeuten mehr Kapazität, aber auch mehr Parameter.

```
Minimal (für die meisten Aufgaben empfohlen):
  Query-Projektion (W_q)
  Value-Projektion (W_v)

Standard (gut für Instruction-Tuning):
  Query-Projektion (W_q)
  Value-Projektion (W_v)
  Key-Projektion (W_k)
  Output-Projektion (W_o)

Full (maximale Kapazität):
  Alle linearen Layer inklusive Feed-Forward
```

Für unser Modell mit 12 Layern fügt die minimale Konfiguration etwa 0,3
Millionen Parameter zu 152 Millionen eingefrorenen Parametern hinzu.
Der Adapter ist etwa 2 Megabyte groß. Ein vollständiger
Fine-Tuning-Checkpoint wäre bei demselben Modell 600 Megabyte groß.

## Adapter mergen

Zur Inferenzzeit gibt es zwei Möglichkeiten. Den Adapter getrennt
halten und das LoRA-Update zur Laufzeit berechnen. Oder den Adapter
für schnellere Inference in die Base-Weights mergen.

```
Separate (Training und Experimente):
  output = input @ W + input @ A @ B
  Flexibel. Adapter lassen sich leicht austauschen. Etwas langsamere Inference.

Merged (Deployment):
  W_merged = W + A @ B
  output = input @ W_merged
  Kein zusätzlicher Rechenaufwand. Keine Verlangsamung. Permanent.
```

Merging ist eine Einwegoperation. Nach dem Mergen lässt sich der
Adapter nicht mehr von den Base-Weights trennen. Aber die Inference
ist genauso schnell wie beim ursprünglichen Modell. Die meisten
Deployments mergen vor dem Serving.

## Ein kleines Code-Beispiel

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.rank = rank
        self.alpha = alpha

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_b(self.lora_a(x)) * (self.alpha / self.rank)
        return base + lora

# Ersetzt einen 768x768 Linear-Layer durch LoRA
layer = LoRALinear(768, 768, rank=16)

frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

print(f"Frozen parameters: {frozen_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Compression ratio: {frozen_params / trainable_params:.0f}x")
```

## Was man sich merken sollte

LoRA fügt kleine trainierbare Matrizen zu eingefrorenen, vortrainierten
Layern hinzu. Das Update ist auf einen niedrigen Rank beschränkt. Der
Rank steuert das Verhältnis von Kapazität zu Parameteranzahl. Der
Alpha-Parameter skaliert den Einfluss des Adapters. Adapter sind
winzige Dateien, die sich beliebig austauschen lassen. Merging
eliminiert den Overhead bei der Inference. LoRA macht Fine-Tuning auf
Consumer-GPUs zugänglich und erhält dabei den Großteil der Qualität
von vollständigem Fine-Tuning.
