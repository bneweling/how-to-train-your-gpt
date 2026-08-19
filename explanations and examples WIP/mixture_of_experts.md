# Mixture of Experts: Wie man skaliert, ohne alles mitzuskalieren

## Die kurze Antwort

Ein Mixture-of-Experts-Modell hat viele Feed-Forward-Netzwerke statt
eines einzigen. Jedes Token wird nur an wenige davon geroutet. Das
Modell hat weit mehr Parameter insgesamt als ein Standardmodell,
nutzt aber für jedes einzelne Token nur einen Bruchteil davon. So
erreichen Modelle wie Mixtral 8x7B und GPT-4 die Leistung deutlich
größerer Modelle bei den Rechenkosten deutlich kleinerer.

Man kann es sich wie ein Krankenhaus vorstellen. Ein Standardmodell
hat einen einzigen Arzt, der jeden Patienten behandelt. Ein
MoE-Modell hat viele Fachärzte. Jeder Patient wird zu den zwei oder
drei Ärzten geschickt, die für seinen Fall am besten qualifiziert
sind. Das Krankenhaus verfügt insgesamt über mehr Fachwissen, aber
jeder Patient verbringt nur Zeit mit den jeweils relevanten
Spezialisten.

## Wo es einzuordnen ist

MoE ersetzt das Feed-Forward-Netzwerk in jedem Transformer-Block.
Attention bleibt gleich. Normalisierung bleibt gleich. Nur die FFN
ändert sich von einem einzelnen dichten Netzwerk zu einer Sammlung
von Experts mit einem Router, der entscheidet, welche Experts
verwendet werden.

```
Standard-Transformer-Block:
  x → RMSNorm → Attention → +x
    → RMSNorm → FFN → +x

MoE-Transformer-Block:
  x → RMSNorm → Attention → +x
    → RMSNorm → Router → Expert 1 (in 30 % der Fälle genutzt)
                      → Expert 2 (25 % genutzt)
                      → Expert 3 (20 % genutzt)
                      → ...
                      → Expert 8 (5 % genutzt)
                                → Ausgaben kombinieren → +x
```

Jeder Expert ist ein vollständiges Feed-Forward-Netzwerk. Dieselbe
Architektur wie unsere Standard-FFN mit SwiGLU. Dieselben Eingabe-
und Ausgabedimensionen. Der Unterschied ist, dass es acht davon
gibt statt einem.

## Warum MoE funktioniert

Dense-Modelle nutzen jeden Parameter für jedes Token. Das ist
verschwenderisch. Manche Parameter kümmern sich um Grammatik. Manche
um Fakten. Manche um logisches Schließen. Für ein gegebenes Token
ist tatsächlich nur eine Teilmenge der Parameter nützlich. Der Rest
trägt Rauschen bei oder ist praktisch inaktiv.

MoE erlaubt es unterschiedlichen Tokens, unterschiedliche Pfade
durch das Modell zu nehmen. Ein Token, das ein Verb repräsentiert,
könnte Experts nutzen, die sich auf Verbkonjugation und
Argumentstruktur spezialisiert haben. Ein Token, das ein Nomen
repräsentiert, könnte Experts nutzen, die sich auf Entity Recognition
und Koreferenz spezialisiert haben. Der Router lernt, jedes Token an
die richtigen Experts zu senden.

Die Gesamtzahl der Parameter ist größer, weil Experts dupliziert
werden. Aber die Rechenkosten pro Token bleiben ungefähr gleich, weil
nur wenige Experts aktiviert werden. Das Modell erhält mehr Kapazität
ohne mehr Rechenaufwand. Das ist Kapazität bei niedrigen aktiven
Parametern.

## Der Router

Der Router ist ein kleiner linearer Layer. Er nimmt den Hidden State
eines Tokens entgegen und gibt für jeden Expert einen Score aus. Das
Token wird an die Experts mit den höchsten Scores gesendet.

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.gate(x)  # [batch, seq, num_experts]

        # Top-k Experts pro Token auswählen
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_weights
```

Für jedes Token wählt der Router die beiden besten Experts aus. Er
gibt außerdem Gewichte für jeden ausgewählten Expert aus. Die
Gewichte bestimmen, wie stark jeder Expert zur finalen Ausgabe
beiträgt. Die Gewichte werden berechnet, indem Softmax auf die
Top-k-Logits angewendet wird, sodass sie sich zu eins summieren.

## Top-k-Auswahl

Das Modell nutzt nicht alle Experts. Es verwendet pro Token nur die
besten zwei. Das ist der Schlüssel zur Effizienz von MoE. Bei acht
Experts und Top-2-Auswahl sind nur 25 Prozent der FFN-Parameter für
jedes Token aktiv. Das Modell hat die 8-fache Menge an
FFN-Parametern eines Dense-Modells, aber nur die 2-fache Rechenlast.

```
Dense-Modell-FFN:    768 → 3072 → 768   (7,1 Mio. Parameter pro Block)
MoE-Modell-FFN:      8 × (768 → 3072 → 768) = 56,7 Mio. Parameter pro Block
Aktiv pro Token:     2 × (768 → 3072 → 768) = 14,2 Mio. genutzte Parameter

FFN-Parameter gesamt: 8× größer
Aktive Rechenlast:     2× größer (Top-2 von 8)
```

Man erhält die 8-fache Kapazität für die 2-fache Rechenlast. Dieses
Verhältnis verbessert sich, je mehr Experts es gibt. Mit 64 Experts
und Top-2-Auswahl erhält man die 64-fache Kapazität für die 2-fache
Rechenlast.

## Load Balancing

Der Router kann lernen, Tokens immer an denselben Expert zu senden.
Expert 1 erhält 80 Prozent der Tokens. Die Experts 2 bis 8 bleiben
untätig. Das Modell wird effektiv zu einem Dense-Modell mit sieben
verschwendeten Experts.

Um das zu verhindern, ist ein Load-Balancing-Loss erforderlich. Der
Loss ermutigt den Router, Tokens gleichmäßig auf die Experts zu
verteilen. Jeder Expert sollte im Verlauf des Trainings ungefähr die
gleiche Anzahl an Tokens erhalten.

```python
def load_balancing_loss(router_logits, expert_mask, num_experts):
    """
    router_logits: [batch * seq, num_experts] : rohe Router-Scores
    expert_mask:   [batch * seq, num_experts] : 1, wenn der Expert ausgewählt wurde

    Gibt einen skalaren Loss zurück, der ungleichmäßiges Routing bestraft.
    Der Loss ist null, wenn jeder Expert gleich viele Tokens erhält.
    """
    # Anteil der Tokens, die an jeden Expert geroutet werden
    fraction_per_expert = expert_mask.float().mean(dim=0)

    # Durchschnittliche Router-Wahrscheinlichkeit für jeden Expert
    router_probs = F.softmax(router_logits, dim=-1)
    avg_prob_per_expert = router_probs.mean(dim=0)

    # Loss = num_experts * sum(Anteil × Wahrscheinlichkeit)
    # Minimal, wenn Anteile und Wahrscheinlichkeiten gleichverteilt sind
    return num_experts * (fraction_per_expert * avg_prob_per_expert).sum()
```

Der Load-Balancing-Loss wird mit einem kleinen Koeffizienten,
typischerweise 0,01, zum eigentlichen Language-Modeling-Loss
addiert. Er drängt den Router in Richtung gleichmäßiger Nutzung,
ohne das Trainingsziel zu dominieren.

## Expert-Kapazität

Selbst mit Load Balancing können manche Experts in einem gegebenen
Batch mehr Tokens erhalten als andere. Um Speicherspitzen zu
verhindern, hat jeder Expert ein Kapazitätslimit. Werden mehr Tokens
an einen Expert geroutet, als seine Kapazität zulässt, werden die
überschüssigen Tokens verworfen. Sie durchlaufen die Residual
Connection unverändert.

```
Expert-Kapazität = (Tokens_pro_Batch / Anzahl_Experts) × Kapazitätsfaktor

Kapazitätsfaktor = 1,25 bedeutet, dass jeder Expert 25 % mehr Tokens
verarbeiten kann als sein fairer Anteil. Überschüssige Tokens über
diesem Limit werden verworfen.
```

Der Kapazitätsfaktor liegt typischerweise zwischen 1,0 und 1,5.
Höhere Werte verschwenden Rechenleistung, weil Experts für Tokens
vorgehalten werden, die sie nie erhalten. Niedrigere Werte erhöhen
die Wahrscheinlichkeit, dass Tokens verworfen werden, was die
Modellqualität verschlechtert.

Verworfene Tokens sind kein vollständiger Verlust. Die Residual
Connection trägt die Information des Tokens weiterhin voran. Das
Token überspringt die FFN in diesem Layer und wird von den Experts
des nächsten Layers verarbeitet.

## Der vollständige MoE-Transformer-Block

```python
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_experts=8, top_k=2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)

        # MoE ersetzt die einzelne FFN
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x, mask=None):
        batch, seq, d_model = x.shape

        # Attention
        x = x + self.attention(self.norm1(x), mask)

        # MoE-FFN
        residual = x
        x = self.norm2(x)

        # Tokens an Experts routen
        logits = self.router(x)  # [batch, seq, num_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [batch, seq, top_k]

        # Tokens durch die ausgewählten Experts verarbeiten
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Tokens finden, die an diesen Expert geroutet wurden
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            # Die Tokens für diesen Expert holen
            tokens = x[expert_mask]  # [num_routed, d_model]

            # Durch den Expert laufen lassen
            expert_output = self.experts[expert_idx](tokens)

            # Mit dem Router-Gewicht für diesen Expert gewichten
            weight_idx = (top_k_indices == expert_idx).float().argmax(dim=-1)
            weights = top_k_weights[expert_mask]
            weights = weights.gather(-1, weight_idx.unsqueeze(-1))

            # Gewichtete Ausgabe akkumulieren
            output[expert_mask] += weights * expert_output

        x = residual + output
        return x
```

Der Forward Pass läuft in einer Schleife über die Experts. In
reinem Python ist das langsam, aber in optimierten Implementierungen
schnell, die alle Expert-Berechnungen zu einer einzigen
Matrixmultiplikation bündeln. Bibliotheken wie Megablocks und
DeepSpeed MoE übernehmen das effizient.

## Welche Modelle MoE verwenden

| Modell | Parameter gesamt | Aktive Parameter | Experts pro Layer | Top-k |
|---|---|---|---|---|
| Mixtral 8x7B | 46,7 Mrd. | 12,9 Mrd. | 8 | 2 |
| Mixtral 8x22B | 141 Mrd. | 39 Mrd. | 8 | 2 |
| GPT-4 (Gerücht) | ~1,7 Bio. | ~280 Mrd. | 8 oder 16 | 2 |
| Gemini 1.5 (Gerücht) | Unbekannt | Unbekannt | Unbekannt | Unbekannt |
| Switch Transformer | 1,6 Bio. | 1,6 Bio. | 2048 | 1 |
| GLaM | 1,2 Bio. | 96 Mrd. | 64 | 2 |

Die Spalte der aktiven Parameter ist entscheidend für die
Inferenzgeschwindigkeit. Mixtral 8x7B hat 46,7 Milliarden Parameter
auf der Festplatte, aber nur 12,9 Milliarden sind pro Token aktiv.
Es läuft ungefähr so schnell wie ein Dense-Modell mit 13 Milliarden
Parametern und erreicht dabei die Qualität eines deutlich größeren.

Switch Transformer nutzte Top-1-Routing, bei dem jedes Token an
genau einen Expert gesendet wird. Das maximiert die Effizienz,
schadet aber der Qualität, weil Tokens nicht von mehreren
Spezialisten profitieren können. Die meisten modernen MoE-Modelle
nutzen Top-2 als optimalen Kompromiss.

## Warum man MoE nicht überall einsetzt

MoE hat Nachteile. Das Modell ist physisch größer und benötigt mehr
Speicher zur Ablage. Der Inferenzspeicher wird bei langen Sequenzen
vom KV-Cache dominiert, nicht von den Modellgewichten, daher spielt
das eine geringere Rolle, als es scheint. Aber das Laden eines
Modells mit 47 Milliarden Parametern erfordert immer noch mehr VRAM
als das Laden eines mit 13 Milliarden Parametern, selbst wenn sie
mit ähnlicher Geschwindigkeit laufen.

Das Training ist schwieriger. Der Load-Balancing-Loss und die
Kapazitätsbeschränkungen der Experts erhöhen die Komplexität. Der
Router kann kollabieren und immer denselben Expert nutzen, was
Überwachung und Eingriffe erfordert. MoE-Modelle neigen stärker zu
Trainingsinstabilität als Dense-Modelle.

Fine-Tuning von MoE-Modellen ist trickreicher. Mit LoRA muss man
entscheiden, ob man die Experts, den Router oder beides anpasst. Die
Wechselwirkungen zwischen Experts und Router sind noch nicht so gut
verstanden wie bei Standard-Transformer-Komponenten.

## Sollten wir MoE zu unserem Modell hinzufügen

Unser Modell ist zu klein, um von MoE zu profitieren. Die Anzahl
der Experts sollte die Anzahl der sinnvollen Spezialisierungen
übersteigen. Mit nur 4 Layern und 152 Millionen Parametern gibt es
nicht genug unterschiedliche sprachliche Muster, auf die sich
Experts spezialisieren könnten. Eine einzelne FFN pro Layer reicht
aus.

MoE beginnt sich etwa ab der Marke von 1 Milliarde Parametern zu
lohnen. Darunter überwiegt der Overhead von Routing und Load
Balancing den Nutzen der zusätzlichen Kapazität. Der Sweet Spot für
MoE liegt bei Modellen mit 10 Milliarden oder mehr Parametern, bei
denen die Rechenersparnis durch sparse Aktivierung erheblich ist.

## Was man sich merken sollte

Ein Mixture-of-Experts-Modell ersetzt jedes Feed-Forward-Netzwerk
durch mehrere Expert-Netzwerke. Ein Router sendet jedes Token an den
einen oder die zwei besten Experts. Das Modell hat mehr Parameter
insgesamt, nutzt aber pro Token nur einen Bruchteil davon. Das
verschafft die Kapazität eines großen Modells zu den Rechenkosten
eines kleineren.

Load Balancing stellt sicher, dass alle Experts genutzt werden,
statt auf einen einzigen bevorzugten Expert zu kollabieren.
Kapazitätslimits der Experts verhindern Speicherspitzen durch
ungleichmäßiges Routing. Router und Experts werden gemeinsam mit dem
Rest des Modells trainiert.

MoE ist vermutlich die Architektur hinter GPT-4 und Gemini. Es ist
der praktikabelste Weg, Modelle über den Punkt hinaus zu skalieren,
an dem Dense-Training unerschwinglich teuer wird. MoE zu verstehen
bedeutet zu verstehen, wie die größten KI-Systeme gebaut werden.
