# Kapitel 4 — Positional Encoding: Reihenfolge lehren

## Die Analogie für Fünfjährige

Betrachte zwei Sätze:
- „The **dog** bit the **man**.“ — gruselig
- „The **man** bit the **dog**.“ — seltsam

Gleiche Wörter, andere Reihenfolge -> **völlig andere Bedeutung**.

Aber der Transformer liest alle Wörter **gleichzeitig** (nicht nacheinander wie Menschen es tun). Er hat **keine Ahnung**, welches Wort zuerst kommt! Deshalb müssen wir **jedes Wort mit seiner Position stempeln**, bevor wir es dem Modell zuführen.

## Die drei Generationen der Positionskodierung

| Methode | Funktionsweise | Vorteile | Nachteile | Verwendet in |
|---|---|---|---|---|
| **Gelernt** | Jede Position erhält ihren eigenen gelernten Vektor | Einfach, flexibel | Kann keine Sequenzen verarbeiten, die länger sind als die Trainingslänge | GPT-2, BERT |
| **Sinusoidal** | Feste Sinus-/Kosinus-Wellen je nach Position | Funktioniert für jede Länge | Schwächer bei relativen Positionen | Ursprünglicher Transformer |
| **RoPE** | Rotiert Q,K-Vektoren um einen positionsabhängigen Winkel | Perfekte relative Positionen, jede Länge | Etwas komplexer | LLaMA, Mistral, Qwen, Gemma |
| **ALiBi** | Fügt den Attention-Scores einen Bias basierend auf der Distanz hinzu | Keine gelernten Parameter, sehr schnell | Weniger ausdrucksstark | BLOOM, MPT |

## Moderner Ansatz: Rotary Position Embeddings (RoPE)

Statt Positionsnummern zu den Embeddings zu **addieren**, **rotiert** RoPE die Query- und Key-Vektoren um einen Winkel, der von der Position abhängt.

### Die mathematische Intuition

In 2D ergibt das Rotieren eines Vektors `(x, y)` um den Winkel `θ`:
```
x' = x*cos(θ) - y*sin(θ)
y' = x*sin(θ) + y*cos(θ)
```

RoPE macht das für JEDES Paar von Dimensionen in den Query- und Key-Vektoren. Der Rotationswinkel für Position `p` und das Dimensionspaar `2i, 2i+1` ist:

```
θ(p, i) = p / (10000^(2i/d_model))
```

**Zentrale Erkenntnis:** Der Winkel hängt von `p` (Position) und `i` (Index des Dimensionspaars) ab. Niedrigere Dimensionspaare rotieren SCHNELL (sie erfassen lokale Wortbeziehungen). Höhere Paare rotieren LANGSAM (sie erfassen weitreichende Beziehungen).

### Durchgerechnetes Zahlenbeispiel

Verfolgen wir RoPE anhand eines winzigen Modells: `d_model=4`, bei der Verarbeitung von Position `p=1`:

**Schritt 1: Frequenzen für jedes Dimensionspaar berechnen**

```
Paar 0 (Dims 0,1): freq = 1 / 10000^(0/4)   = 1 / 1       = 1.000
Paar 1 (Dims 2,3): freq = 1 / 10000^(2/4)   = 1 / 10000^0.5 = 1 / 100 = 0.010
```

**Schritt 2: Rotationswinkel für Position p=1 berechnen**

```
Paar 0 Winkel: θ₀ = p * freq₀ = 1 * 1.000 = 1.000 Radiant (≈ 57.3°)
Paar 1 Winkel: θ₁ = p * freq₁ = 1 * 0.010 = 0.010 Radiant (≈ 0.57°)
```

**Schritt 3: Rotation auf einen Query-Vektor an Position 1 anwenden**

```
Vor RoPE: q₁ = [0.8, 0.3, -0.5, 0.2]

Rotiere Paar 0 (Dims 0,1) um 57.3°:
  dim0' = 0.8*cos(1.0) - 0.3*sin(1.0) = 0.8*0.540 - 0.3*0.842 = 0.432 - 0.253 = 0.179
  dim1' = 0.8*sin(1.0) + 0.3*cos(1.0) = 0.8*0.842 + 0.3*0.540 = 0.674 + 0.162 = 0.836

Rotiere Paar 1 (Dims 2,3) um 0.57°:
  dim2' = -0.5*cos(0.01) - 0.2*sin(0.01) = -0.5*1.000 - 0.2*0.010 = -0.500 - 0.002 = -0.502
  dim3' = -0.5*sin(0.01) + 0.2*cos(0.01) = -0.5*0.010 + 0.2*1.000 = -0.005 + 0.200 = 0.195

Nach RoPE: q₁' = [0.179, 0.836, -0.502, 0.195]
```

Berechnen wir nun, was an den Positionen 1 und 3 passiert:

```
Position 1: θ₀ = 1.0 rad,  θ₁ = 0.01 rad
Position 3: θ₀ = 3.0 rad,  θ₁ = 0.03 rad

Das Skalarprodukt q₁ · k₃ hängt von der DIFFERENZ ab:
  Δθ₀ = 3.0 - 1.0 = 2.0 rad
  Δθ₁ = 0.03 - 0.01 = 0.02 rad
  
Diese Differenz hängt NUR von (3-1)=2 ab, dem relativen Abstand!
Absolute Positionen spielen keine Rolle — nur wie weit sie auseinanderliegen.
```

Deshalb ist RoPE so genial: Der Attention-Score zwischen Position `i` und `j` hängt **nur** von ihrer relativen Distanz `(j-i)` ab, nicht von ihren absoluten Positionen.

### Warum theta=10000?

Die Basisfrequenz `theta = 10000` steuert die „Streuung“ der Frequenzen:

```
Niedriges theta (z. B. 100):
  - Alle Dimensionspaare rotieren ähnlich
  - Modell ist eher „positionsagnostisch“ — besser für lange Kontexte
  - Verliert aber feingranulare Positionsauflösung

Hohes theta (z. B. 100000):
  - Sehr unterschiedliche Rotationsgeschwindigkeiten über die Dimensionen hinweg
  - Besser darin, nahe beieinanderliegende Positionen zu unterscheiden
  - Hat aber Schwierigkeiten bei sehr langen Kontexten

10000 wurde empirisch als guter Kompromiss zwischen diesen Zielkonflikten ermittelt.
```

### Den Kontext über die Trainingslänge hinaus erweitern

Was, wenn wir mit 2048 Tokens trainiert haben, aber bei der Inference 4096 verwenden wollen?

**Das Problem:** RoPE wurde für die Positionen 0-2047 vorab berechnet. Position 3000 wurde nie gesehen.

**Lösungen:**
| Methode | Funktionsweise | Qualität |
|---|---|---|
| **Lineare Interpolation** | Position / Skalierung (z. B. p/2 für die doppelte Länge) | OK, verliert Auflösung |
| **NTK-aware Scaling** | Skaliert theta je Frequenz unterschiedlich | Gut |
| **YaRN** | NTK + Temperature Scaling | Am besten (im Produktiveinsatz verwendet) |
| **Retrain** | Einfach auf längeren Sequenzen trainieren | Perfekt, aber teuer |

Für unseren kleinen Trainingslauf spielt das keine Rolle — aber wisse, dass dies für Produktionsmodelle ein heißes Forschungsgebiet ist.

## RoPE-Code — kommentiert

```python
import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    WAS: Rotary Position Embeddings (RoPE).
    WARUM: Statt Positionsinformationen zu den Embeddings zu ADDIEREN,
           ROTIEREN wir die Q- und K-Vektoren um positionsabhängige Winkel.
           Das Skalarprodukt q_i · k_j hängt dann NUR von (j-i) ab,
           genau das, worauf es bei Attention ankommen sollte.

           Paper: "RoFormer" (Su et al., 2021)
           Verwendet in: LLaMA 1/2/3, Mistral, Mixtral, Qwen 1/2, Gemma

           Funktionsweise im Überblick:
           1. Für jedes Paar von Dimensionen (0,1), (2,3), (4,5), ...
           2. Rotation um Winkel = Position * Frequenz
           3. Niedrigere Dims rotieren schnell (lokale Position)
              Höhere Dims rotieren langsam (globale Position)
           4. Das Skalarprodukt hängt auf natürliche Weise von der relativen Distanz ab
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        WAS: Rotationsfrequenzen für schnellen Zugriff vorab berechnen.

        Args:
            d_model:     Head-Dimension (z. B. 64 bei GPT-2). Muss gerade sein.
            max_seq_len: Berechnet Winkel für Positionen 0..max_seq_len-1 vorab.
            theta:       Basisfrequenz. 10000 ist Standard. Steuert die
                         Streuung zwischen schnellen und langsamen Rotationsfrequenzen.
        """
        super().__init__()

        # WAS: Prüfen, dass d_model gerade ist (es braucht Paare zum Rotieren)
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) must be even for RoPE. "
            f"Each pair of dimensions needs a partner to rotate with."
        )

        # WAS: Dimensionsindizes erzeugen: [0, 2, 4, ..., d_model-2]
        # WARUM: Jedes Paar (2i, 2i+1) bekommt dieselbe Rotationsfrequenz.
        #        Wir brauchen nur die Hälfte der Indizes, weil sich Paare
        #        die Frequenz teilen.
        dim_indices = torch.arange(0, d_model, 2).float()

        # WAS: Rotationsfrequenzen berechnen
        # WARUM: theta_i = 1 / (theta ^ (2i / d_model))
        #
        #        i=0:  1 / 10000^(0/64)      = 1.0      → schnelle Rotation (lokal)
        #        i=30: 1 / 10000^(60/64)     ≈ 0.0001   → langsame Rotation (global)
        #
        #        Dieser Multi-Scale-Ansatz sorgt dafür, dass manche
        #        Dimensionen die lokale Wortreihenfolge erfassen, während
        #        andere weitreichende Positionsbeziehungen erfassen.
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # WAS: Winkel für alle Positionen vorab berechnen
        # WARUM: cos/sin während des Trainings zu berechnen ist teuer.
        #        Sie einmal vorab zu berechnen und zu cachen ist 100x schneller.
        positions = torch.arange(max_seq_len).float()     # [0, 1, 2, ..., 2047]

        # WAS: Äußeres Produkt: jede Position x jede Frequenz
        #      freqs[p, i] = p * inv_freq[i] = Winkel für Position p, Dimensionspaar i
        #      Shape: [max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # WAS: Auf die volle Dimension duplizieren
        # WARUM: Jedes Dimensionspaar (2i, 2i+1) bekommt denselben Winkel,
        #        also kopieren wir jeden Winkel: [θ0, θ1, θ2, ...] -> [θ0, θ0, θ1, θ1, ...]
        emb = freqs.repeat_interleave(2, dim=-1)         # [max_seq_len, d_model]

        # WAS: cos und sin für alle Positionen cachen
        # WARUM: register_buffer bedeutet, dass diese sich mit model.to(device)
        #        mitbewegen und mit model.state_dict() gespeichert werden, aber
        #        KEINE trainierbaren Parameter sind (keine Gradienten nötig).
        self.register_buffer("cos_cached", emb.cos())   # cos jedes Winkels
        self.register_buffer("sin_cached", emb.sin())   # sin jedes Winkels

    @staticmethod
    @staticmethod
    def rotate_half(x):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        WAS: RoPE auf Queries oder Keys anwenden.

        Input:  [batch, num_heads, seq_len, head_dim]
                x kann entweder Q oder K sein (NICHT V — Values brauchen keine Position)
        Output: Gleiche Shape, rotiert um positionsabhängige Winkel

        WARUM nur auf Q und K angewendet:
        Der Attention-Score = Q_i · K_j bestimmt, WELCHEN Values Aufmerksamkeit
        geschenkt wird. Wir wollen, dass dieser Score von der relativen Position
        abhängt. Die VALUE-Vektoren tragen den Inhalt — Position ist für den
        Inhalt selbst irrelevant. Position ist nur relevant für die Entscheidung,
        welchen Tokens Aufmerksamkeit geschenkt wird.
        """
        # WAS: cos und sin für die aktuelle Sequenzlänge extrahieren
        # WARUM: Wenn seq_len=512, aber max_seq_len=2048 ist, brauchen wir
        #        nur die ersten 512 Zeilen der gecachten cos/sin-Tabellen.
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # WAS: Batch- und Head-Dimensionen für Broadcasting hinzufügen
        # WARUM: cos/sin haben die Form [seq_len, head_dim]. Wir müssen sie
        #        mit x [batch, heads, seq_len, head_dim] multiplizieren.
        #        unsqueeze(0).unsqueeze(0) fügt Dimensionen an Position 0 und 1 hinzu:
        #        [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #        Jetzt werden sie korrekt über Batch und Heads gebroadcastet.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # WAS: Rotation ausführen: x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # WARUM: Das ist mathematisch äquivalent zur Anwendung einer 2D-
        #        Rotationsmatrix auf jedes Dimensionspaar, aber implementiert
        #        in reinen elementweisen Operationen — viel schneller und
        #        parallelisierbar.
        return (x * cos) + (self.rotate_half(x) * sin)
```

---

**Zurück:** [Kapitel 3 — Embeddings](03_embeddings.md)
**Weiter:** [Kapitel 5 — Attention](05_attention.md)
