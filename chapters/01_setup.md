# Kapitel 1 — Setup & Tooling

## Was du wissen musst, bevor wir starten

### "Was ist Python eigentlich?"

Python ist einfach eine **Sprache, um dem Computer zu sagen, was er tun soll**. Du schreibst Anweisungen in eine `.py`-Datei, und Python "liest" sie und führt sie eine nach der anderen aus. Wenn du schon einmal Python geschrieben hast — und sei es nur `print("hello")` — bist du bereit.

### "Was ist eine GPU, und warum brauche ich eine?"

**Analogie:** Stell dir vor, du musst 10.000 winzige Fliesen bemalen.

- Eine **CPU** ist wie ein Meistermaler, der die Fliesen EINE nach der anderen bemalt — präzise, aber langsam.
- Eine **GPU** ist wie 10.000 Kunststudenten, die jeweils gleichzeitig EINE Fliese bemalen — schneller, auch wenn jeder Student "weniger geschickt" ist als der Meister.

Das Training neuronaler Netze besteht aus Millionen **identischer, unabhängiger mathematischer Operationen** (Matrixmultiplikationen). GPUs verfügen über Tausende kleiner Kerne, die genau dafür ausgelegt sind. Eine GPU kann beim Training 50- bis 100-mal schneller sein als eine CPU.

**Brauchst du unbedingt eine GPU?** Nein — unser winziges Testmodell läuft auch auf der CPU, nur sehr langsam (Minuten statt Stunden). Für echtes Training ist eine GPU jedoch unverzichtbar.

| Deine Hardware | Was du trainieren kannst | Ungefähre Geschwindigkeit |
|---|---|---|
| Nur CPU | Winziges Modell (4 Layer, 256 Dimensionen) | Stunden |
| Apple M1/M2/M3 | Kleines Modell (12 Layer, 768 Dimensionen) | Stunden |
| RTX 3060/4060 (12GB) | GPT-2 small (124M Parameter) | Wenige Stunden |
| RTX 3090/4090 (24GB) | GPT-2 medium (350M) | Wenige Stunden |
| A100 (80GB) | GPT-2 large (774M) | Stunden |

### "Was ist eine virtuelle Umgebung?"

Eine virtuelle Umgebung (`venv`) ist wie eine **saubere, leere Küche** nur für dieses Projekt. Ohne sie würdest du die Zutaten deines Projekts (Python-Pakete) mit allem anderen auf deinem Computer vermischen — das führt zu Konflikten, wenn zwei Projekte unterschiedliche Versionen desselben Pakets benötigen.

```bash
# Eine saubere Küche erstellen
python -m venv gpt_env

# Hineingehen
source gpt_env/bin/activate          # Mac/Linux
# ODER:
gpt_env\Scripts\activate             # Windows

# Jetzt wirkt sich pip install nur auf diese Küche aus
# Zum Verlassen: `deactivate` eingeben
```

### "Was ist pip?"

`pip` ist Pythons **Paketinstallationsprogramm**. Es lädt Code, den andere Leute geschrieben haben (Bibliotheken), aus dem Internet herunter und installiert ihn in deiner Umgebung. Stell es dir wie einen "App Store" für Python-Code vor.

### "Was ist PyTorch?"

PyTorch ist das Framework, mit dem wir unser neuronales Netz bauen werden. Es bietet:

| PyTorch-Feature | Was es macht | Analogie |
|---|---|---|
| `torch.Tensor` | Mehrdimensionale Arrays | Wie NumPy-Arrays, können aber auf der GPU liegen |
| `torch.nn.Module` | Bausteine für Netze | LEGO-Steine, die du zusammensteckst |
| `torch.optim` | Algorithmen, die Weights aktualisieren | Der "lernende" Teil von Machine Learning |
| `autograd` | Automatische Gradientenberechnung | Erledigt die Analysis automatisch für dich |
| `DataLoader` | Liefert Daten effizient | Ein Förderband, das Trainingsdaten anliefert |

## Installation — Schritt für Schritt

```bash
# Schritt 1: Virtuelle Umgebung erstellen
python -m venv gpt_env

# Schritt 2: Aktivieren
source gpt_env/bin/activate          # Mac/Linux
# gpt_env\Scripts\activate           # Windows

# Schritt 3: PyTorch installieren (die passende Variante wählen)
# Nur CPU (Standard, funktioniert überall):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Für Apple Silicon (M1/M2/M3):
# pip install torch torchvision torchaudio

# Für NVIDIA-GPU (CUDA 11.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Für NVIDIA-GPU (CUDA 12.1 - neuere Karten wie die RTX-40er-Serie):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Schritt 4: Restliche Pakete installieren
pip install tiktoken datasets numpy matplotlib

# Schritt 5: Prüfen, ob alles funktioniert
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Was jede Bibliothek macht (im Detail)

| Bibliothek | Was sie macht | Warum wir sie brauchen |
|---|---|---|
| **torch** | PyTorch-Kern: Tensoren, GPU-Operationen, autograd | Das Fundament — alles andere baut darauf auf |
| **tiktoken** | Schneller BPE-Tokenizer von OpenAI | Derselbe Tokenizer, den GPT-3.5/4 verwenden. In Rust geschrieben, extrem schnell |
| **datasets** (HuggingFace) | Lädt Trainingsdaten herunter und cached sie | Erspart uns das manuelle Herunterladen und Parsen von Wikipedia |
| **numpy** | Schnelle numerische Arrays auf der CPU | Für schnelle Datenmanipulation (auch wenn PyTorch das meiste übernimmt) |
| **matplotlib** | Erstellt Diagramme und Grafiken | Um unseren Trainings-Loss zu visualisieren — lernt das Modell? |
| **math** (eingebaut) | sqrt, sin, cos, pi | Mathematische Konstanten für das Positional Encoding |
| **time** (eingebaut) | Verstrichene Zeit messen | Trainingsgeschwindigkeit in Tokens/Sekunde verfolgen |
| **os** (eingebaut) | Verzeichnisse erstellen, Dateien speichern | Modell-Checkpoints speichern, damit kein Fortschritt verloren geht |

## Unser vollständiger Import-Block

```python
# ===== WAS: Standard-Python-Bibliotheken =====
import math              # WARUM: sqrt(), sin(), cos() für die Mathematik des Positional Encoding
import time              # WARUM: Trainingsgeschwindigkeit messen (Tokens pro Sekunde)
import os                # WARUM: Verzeichnisse erstellen, Modell-Checkpoint-Dateien speichern/laden
from dataclasses import dataclass  # WARUM: saubere Config-Klasse — keine unübersichtlichen Dictionaries

# ===== WAS: NumPy — die Array-Bibliothek für die CPU =====
import numpy as np       # WARUM: schnelle numerische Operationen auf CPU-Arrays
                         #      (meist für schnelle Datenchecks, nicht für die Schwerstarbeit)

# ===== WAS: PyTorch — das Framework für neuronale Netze =====
import torch             # WARUM: Kernbibliothek — Tensoren, GPU-Unterstützung, autograd
import torch.nn as nn               # WARUM: Bausteine für neuronale Netze:
                                     #      Linear (dense layers), Embedding (Lookup-Tabellen),
                                     #      Dropout (Regularisierung), ModuleList (Layer stapeln)
import torch.nn.functional as F     # WARUM: zustandslose Funktionen, verwendet innerhalb von forward():
                                     #      softmax (Umwandlung in Wahrscheinlichkeiten),
                                     #      cross_entropy (misst den Vorhersagefehler),
                                     #      silu (SwiGLU-Aktivierungsfunktion)
from torch.utils.data import Dataset, DataLoader  # WARUM: effiziente Daten-Pipeline
#                                  Dataset = definiert, wie ein einzelnes Sample geladen wird
#                                  DataLoader = bündelt sie in Batches, mischt, lädt vor

# ===== WAS: tiktoken — OpenAIs schneller BPE-Tokenizer =====
import tiktoken          # WARUM: derselbe Byte-Pair-Encoding-Tokenizer wie GPT-3.5/GPT-4
                         #      In Rust geschrieben, ~100x schneller als reine Python-Tokenizer
                         #      Verarbeitet ein Vokabular von 50K+ effizient

# ===== WAS: HuggingFace datasets — Trainingstext herunterladen =====
from datasets import load_dataset    # WARUM: eine Zeile, um WikiText-103 herunterzuladen
                                     #      Übernimmt Caching (lädt nur einmal herunter),
                                     #      Streaming (für Datasets, die zu groß für die Festplatte sind)
                                     #      und automatische Formatkonvertierung

# ===== WAS: matplotlib — Loss-Kurven plotten =====
import matplotlib.pyplot as plt      # WARUM: Trainingsfortschritt visualisieren
                                     #      Sinkt der Loss? Stagniert er?
                                     #      Ein Bild sagt mehr als 1.000 Log-Zeilen

# ===== WAS: Schnelle Verifizierung =====
# WARUM: Teste deine Umgebung immer, bevor du 500 Zeilen Code schreibst.
#      Ein fehlender Import jetzt erspart später Stunden des Debuggens.
print("All imports ready!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Erwartete Ausgabe (mit GPU):**
```
All imports ready!
PyTorch version: 2.1.0
CUDA available:  True
GPU:             NVIDIA GeForce RTX 3090
GPU Memory:      24.0 GB
```

**Erwartete Ausgabe (nur CPU):**
```
All imports ready!
PyTorch version: 2.1.0
CUDA available:  False
```

Wenn du die GPU-Ausgabe siehst, bist du bereit fürs Training. Wenn du nur die CPU-Ausgabe siehst, funktioniert das Training trotzdem — nur langsamer. So oder so, machen wir weiter.

---

## Wie du den Rest dieses Guides angehen solltest

Jedes Kapitel folgt diesem Muster:

1. **Analogie** — Das Konzept in einfachen Worten erklären (als würde man es einem Fünfjährigen beibringen)
2. **Mathematik** — Die tatsächlichen Formeln zeigen und erklären, warum sie funktionieren
3. **Code** — Jede einzelne Zeile kommentiert mit WAS sie tut und WARUM
4. **Visualisierung** — Diagramm oder durchgerechnetes Beispiel, das zeigt, wie die Daten durchfließen

Wenn du dich jemals verloren fühlst, kehre zur Analogie zurück. Wenn dir der Code zu viel wird, konzentriere dich auf die WAS/WARUM-Kommentare — sie sind so gestaltet, dass man sie von oben nach unten wie eine Geschichte lesen kann.

---

**Zurück:** [Kapitel 0 — Überblick](00_overview.md)
**Weiter:** [Kapitel 2 — Tokenisierung](02_tokenization.md)
