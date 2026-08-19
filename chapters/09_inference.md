# Kapitel 9 — Inference: Es zum Sprechen bringen

## Wie sich Generierung vom Training unterscheidet

| Aspekt | Training | Inference (Generierung) |
|---|---|---|
| **Ziel** | Lernen, das nächste Token korrekt vorherzusagen | Tatsächlich neuen Text generieren |
| **Input** | Vollständige Sequenz mit Zielwert | Nur Prompt (kein Zielwert) |
| **Teacher Forcing** | Ja — korrekte Antwort wird gezeigt | Nein — Modell generiert seine eigene Zukunft |
| **Forward Pass** | Ein Pass für die gesamte Sequenz | Ein Pass PRO neuem Token |
| **Geschwindigkeit** | Schnell (parallel im Batch) | Langsam (sequenziell, Token für Token) |
| **Causal Mask** | Verhindert das Sehen zukünftiger Tokens | Zukünftige Tokens existieren noch nicht |
| **Dropout** | Aktiv (zur Regularisierung) | Deaktiviert (für deterministischen Output) |
| **Gradient** | Ja (Backward Pass) | Nein (kein Lernen findet statt) |

## Die naive Generierungsschleife

Unsere einfachste Implementierung (durchläuft bei jedem Schritt erneut alle Tokens):

```python
for _ in range(max_new_tokens):
    # Berechnet JEDES Mal die GESAMTE Sequenz neu!  ← verschwenderisch!
    logits, _ = model(input_ids)           # Verarbeitet ALLE Tokens
    next_token = sample(logits[:, -1, :])  # Nutzt nur die LETZTE Vorhersage
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

**Problem:** Token 500 wurde bereits 499 Mal verarbeitet, bevor wir Token 501 hinzufügen!

## KV-Cache — Der größte Geschwindigkeitsgewinn

### Die zentrale Erkenntnis

Während der autoregressiven Generierung ändern sich bereits berechnete Keys und Values nicht. Die K- und V-Werte von Token 0 sind dieselben, egal ob wir Token 1 oder Token 500 vorhersagen.

**Ohne KV-Cache:** K und V werden bei jedem Schritt für ALLE Tokens neu berechnet.  
**Mit KV-Cache:** K,V werden nur für das NEUE Token berechnet. An den Cache angehängt. Alte werden wiederverwendet.

```
Schritt 1: Verarbeite "The"           → Speichere K["The"], V["The"] im Cache
Schritt 2: Verarbeite "cat"           → Wiederverwende K,V für "The", berechne K,V für "cat"
Schritt 3: Verarbeite "sat"           → Wiederverwende K,V für "The","cat", berechne für "sat"
...
Schritt 500: Verarbeite "mat"         → Wiederverwende K,V für 499 Tokens, berechne 1 neues
```

### Geschwindigkeitsverbesserung

| Sequenzlänge | Ohne KV-Cache | Mit KV-Cache | Speedup |
|---|---|---|---|
| 100 | 5,050 Operationen | 100 Operationen | 50× |
| 500 | 125,250 Operationen | 500 Operationen | 250× |
| 1000 | 500,500 Operationen | 1000 Operationen | 500× |
| 4096 | 8.3M Operationen | 4096 Operationen | **2048×** |

Je länger die Generierung, desto wichtiger wird der KV-Cache!

### Speicherkosten

Der KV-Cache speichert `2 * num_layers * num_heads * seq_len * head_dim` Floats:

Für GPT-2 Small, das 1000 Tokens generiert:
```
2 × 12 × 12 × 1000 × 64 = 18,432,000 floats
= 18.4M × 4 bytes (float32) = 73.7 MB
= 18.4M × 2 bytes (bfloat16) = 36.8 MB
```

Gut handhabbar für kleine Modelle, aber für GPT-3 (96 Layer, 96 Heads) × 4096 Tokens:
```
2 × 96 × 96 × 4096 × 128 = 9.66 MILLIARDEN floats = 38.6 GB!
```

Deshalb benötigt Inference mit langem Kontext enormen GPU-Speicher oder speichereffiziente KV-Cache-Techniken.

### Konzept der KV-Cache-Implementierung

```python
# Vereinfachter KV-Cache (konzeptionell — keine vollständige Implementierung)
class GPTWithKVCache(GPT):
    def generate_with_cache(self, input_ids, max_new_tokens):
        # Prefill: Prompt verarbeiten, K,V speichern
        kv_cache = []  # Liste von (K, V)-Tupeln pro Layer
        
        # Erster Forward-Pass: gesamten Prompt verarbeiten
        logits, new_kv = self.forward_with_cache(input_ids, kv_cache=None)
        kv_cache = new_kv  # Zur Wiederverwendung speichern
        
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1, :])
            # Nur das NEUE Token vorwärts durchlaufen, gecachte K,V wiederverwenden
            logits, new_kv = self.forward_with_cache(
                next_token.unsqueeze(1),  # Nur 1 neues Token!
                kv_cache=kv_cache
            )
            kv_cache = new_kv  # Neue K,V an den Cache anhängen
            input_ids = torch.cat([input_ids, next_token], dim=1)
```

## Sampling-Strategien — Wie das nächste Token gewählt wird

### Greedy Sampling (Temperature = 0)

Wählt immer das eine wahrscheinlichste Token.

```
Prompt: "The cat sat on the"
Logits: [the: 9.2,  a: 8.1,  my: 3.2,  their: 1.1, ...]
                                ↑ immer dieses wählen
Ergebnis: "The cat sat on the mat. The cat sat on the mat. The cat..."  ← wiederholt sich!
```

**Problem:** Deterministisch → derselbe Prompt liefert immer dieselbe Ausgabe. Neigt zu Wiederholungen.

### Temperature Sampling

Skaliert die Logits vor dem Softmax. Niedrigere Temperature = schärfere Verteilung (selbstbewusstere Auswahl). Höhere = flacher (zufälliger).

```python
# Effekt der Temperature auf eine Beispielverteilung:
logits = [2.0, 1.0, 0.5, 0.1]  # 4 mögliche Tokens

# T = 0.5 (kalt — selbstbewusst):
scaled = [2.0/0.5, 1.0/0.5, 0.5/0.5, 0.1/0.5]  # → [4.0, 2.0, 1.0, 0.2]
probs  = softmax([4.0, 2.0, 1.0, 0.2])          # → [0.86, 0.12, 0.02, 0.00]
# Token 0 hat 86% Wahrscheinlichkeit — sehr selbstbewusst!

# T = 1.0 (Standard):
probs = softmax([2.0, 1.0, 0.5, 0.1])           # → [0.56, 0.21, 0.13, 0.10]
# Token 0 = 56% — ausgewogene Verteilung

# T = 2.0 (heiß — kreativ):
scaled = [2.0/2.0, 1.0/2.0, 0.5/2.0, 0.1/2.0]  # → [1.0, 0.5, 0.25, 0.05]
probs  = softmax([1.0, 0.5, 0.25, 0.05])         # → [0.36, 0.22, 0.22, 0.20]
# Flacher — Token 0 nur 36%, Tokens 2 und 3 sind konkurrenzfähig
```

**Gleicher Prompt, unterschiedliche Temperatures:**

```
T=0.2 (fokussiert): "The capital of France is Paris, which is located in the Île-de-France region."
T=0.8 (ausgewogen): "The capital of France is Paris, a city known for its art, cuisine, and the Eiffel Tower."
T=1.5 (kreativ):    "The capital of France is Paris, where baguettes dream of becoming croissants under moonlight."
```

### Top-K Sampling

Berücksichtigt nur die K wahrscheinlichsten Tokens. Alles andere → Wahrscheinlichkeit 0.

```
K=50: Nur die Top 50 Tokens. Guter Standardwert — filtert offensichtlichen Unsinn heraus.
K=10: Aggressiver Filter. Neigt zu Wiederholungen, aber nie unsinnig.
K=1:  Entspricht Greedy (wählt immer #1).
```

### Top-P (Nucleus) Sampling

Berücksichtigt nur die KLEINSTE Menge an Tokens, deren kumulative Wahrscheinlichkeit P übersteigt.

```
Nach Wahrscheinlichkeit sortierte Tokens: [0.45, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

Top-P = 0.9:
  Kumulativ: 0.45+0.22+0.13+0.08+0.05 = 0.93 > 0.9
  Behalte die ersten 5 Tokens. Verwirf den Rest.

Top-P = 0.5:
  Kumulativ: 0.45+0.22 = 0.67 > 0.5
  Behalte die ersten 2 Tokens.
```

**Warum Top-P statt Top-K?** Top-P passt sich an die Konfidenz des Modells an:
- Wenn das Modell sehr sicher ist: wenige Tokens werden behalten (scharfe Verteilung)
- Wenn das Modell unsicher ist: viele Tokens werden behalten (flache Verteilung)

Top-K behält immer genau K Tokens, unabhängig von der Konfidenz.

### Beam Search

Anstatt jeweils ein Token einzeln auszuwählen, werden mehrere "Beams" (Kandidatensequenzen) parallel verfolgt:

```
Beam-Breite = 3:

Schritt 1: "The" → 3 beste nächste Tokens: ["cat"(0.3), "dog"(0.2), "man"(0.1)]
Schritt 2: "The cat" → 3 beste Fortsetzungen: ["sat"(0.4), "is"(0.2), "was"(0.15)]
        "The dog" → 3 beste: ["ran"(0.35), "is"(0.2), "barked"(0.1)]
        "The man" → 3 beste: ["walked"(0.3), "said"(0.25), "is"(0.1)]
        Wähle die 3 insgesamt besten Sequenzen:
        "The cat sat" (0.3×0.4=0.12), "The dog ran" (0.2×0.35=0.07), ...
```

**Beam Search**: Höherwertiger Output, aber deterministisch (immer derselbe Output) und langsamer. Wird häufig für Übersetzung eingesetzt, nicht für kreatives Schreiben.

### Repetition Penalty

Während der Generierung werden Tokens bestraft, die bereits aufgetreten sind:

```
Für jedes Kandidaten-Token:
  penalty = 1.0, falls Token NICHT in der jüngsten Historie vorkommt
  penalty = 0.5, falls Token einmal kürzlich aufgetreten ist
  penalty = 0.2, falls Token mehrfach aufgetreten ist

logits = logits * penalty
```

Das verhindert, dass sich das Modell in einer Schleife verfängt: `"I like cats. I like cats. I like cats..."`

### Vergleichstabelle

| Strategie | Zufälligkeit | Qualität | Geschwindigkeit | Anwendungsfall |
|---|---|---|---|---|
| **Greedy** (T=0) | Keine | Gut für Fakten | Schnell | Übersetzung, Code |
| **Temperature** | Steuerbar | Variiert | Schnell | Kreatives Schreiben |
| **Top-K=50** | Niedrig-moderat | Guter Standard | Schnell | Allgemeine Generierung |
| **Top-P=0.9** | Adaptiv | Guter Standard | Schnell | Chat, Konversation |
| **Beam Search** | Keine | Beste | 3-5× langsamer | Übersetzung, Zusammenfassung |
| **T=0.7 + Top-P=0.9** | Moderat | Hervorragend | Schnell | 🏆 Empfohlener Standard |

## Vollständiger Inference-Code

### Einen Checkpoint laden

```python
import torch


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    WAS: Lädt ein trainiertes GPT-Modell aus einer gespeicherten Checkpoint-Datei.
    WARUM: Nach dem Training speichern wir den Modellzustand. Um Text zu generieren,
           müssen wir diesen Zustand wieder laden — Weights, Config, alles.
    """
    # WAS: Lädt das Checkpoint-Dictionary von der Festplatte
    # WARUM: map_location stellt sicher, dass auf das richtige Device geladen wird (CPU/GPU)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # WAS: Erstellt das Modell anhand der gespeicherten Config neu
    model = GPT(checkpoint["config"])

    # WAS: Lädt die trainierten Weights in das Modell
    # WARUM: state_dict enthält jeden während des Trainings gelernten Parameterwert
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)  # Auf die GPU verschieben
    model.eval()              # Dropout für Inference deaktivieren

    print(f"Loaded model from step {checkpoint['step']}, "
          f"loss: {checkpoint['loss']:.4f}")
    return model
```

### Wrapper für die Textgenerierung

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None,
):
    """
    WAS: Generiert Text aus einem Prompt mithilfe eines trainierten GPT-Modells.
    WARUM: High-Level-Schnittstelle — tokenisieren → generieren → dekodieren.

    Parameter-Leitfaden:
      temperature: 0.2 = sachlich, 0.8 = ausgewogen, 1.5 = wild
      top_k:       50 = Standard, 10 = konservativ, 0 = deaktiviert
      top_p:       0.9 = empfohlen, 0.5 = eng, 1.0 = deaktiviert
    """
    device = device or next(model.parameters()).device

    # WAS: Wandelt den Prompt-String in Token-IDs um
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    # WAS: Führt die autoregressive Generierung aus
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # WAS: Wandelt die generierten Token-IDs zurück in einen String um
    return tokenizer.decode(output_ids[0].tolist())
```

### Beispiel für interaktive Generierung

```python
# Lädt das trainierte Modell
model = load_checkpoint("checkpoints/best_model.pt", device)

# Testet verschiedene Generierungsstrategien
prompts = [
    "Once upon a time, in a land far away,",
    "The secret to happiness is",
    "If I could travel anywhere in the world, I would go to",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Konservativ — gut für Fakten
    text = generate_text(
        model, tokenizer, prompt, temperature=0.3, top_k=20,
    )
    print(f"\nConservative (T=0.3, K=20):")
    print(f"  {text[:300]}")

    # Ausgewogen — guter Standard
    text = generate_text(
        model, tokenizer, prompt, temperature=0.8, top_k=50, top_p=0.9,
    )
    print(f"\nBalanced (T=0.8, K=50, P=0.9):")
    print(f"  {text[:300]}")

    # Kreativ — gut zum Schreiben
    text = generate_text(
        model, tokenizer, prompt, temperature=1.3, top_k=100, top_p=0.95,
    )
    print(f"\nCreative (T=1.3, K=100, P=0.95):")
    print(f"  {text[:300]}")
```

---

**Zurück:** [Kapitel 8 — Training](08_training.md)
**Weiter:** [Kapitel 10 — Vollständiges Skript](10_full_script.md)
