# SwiGLU: Die intelligente Aktivierungsfunktion

## Was ist es

SwiGLU ist die Aktivierungsfunktion, die im Feed-Forward-Netzwerk
moderner Transformer verwendet wird. Eine Aktivierungsfunktion
entscheidet, wie viel Information durch eine Layer hindurchgelassen
wird. Alte Aktivierungsfunktionen waren einfache Ein-/Aus-Schalter.
SwiGLU ist intelligenter. Sie hat einen zweiten Pfad, der wie ein
Gate wirkt. Das Gate lernt, wann es Information durchlassen und
wann es sie blockieren soll.

Stell es dir wie einen Wasserhahn vor. ReLU ist ein Wasserhahn, der
entweder ganz auf oder ganz zu ist. Nichts dazwischen. SwiGLU ist
ein Wasserhahn, den du in jede beliebige Position drehen kannst.
Ein wenig geöffnet für ein Rinnsal. Halb geöffnet für einen
mäßigen Fluss. Ganz geöffnet, wenn du alles brauchst. Das Modell
lernt die richtige Position für jede Eingabe.

## Wo wird es verwendet

SwiGLU steckt in jedem Transformer-Block. Es ersetzt die älteren
Aktivierungsfunktionen im Feed-Forward-Netzwerk. Jedes Mal, wenn
das Modell einen Token durch die FFN-Layer verarbeitet, entscheidet
SwiGLU, welche Information behalten und welche verworfen wird.

```
Transformer Block:
  x → RMSNorm → Attention → +x
    → RMSNorm → SwiGLU FFN → +x
                   ^^^^^^
                   Dieser Teil
```

LLaMA, PaLM, Gemini und die meisten seit 2022 entwickelten Modelle
verwenden SwiGLU. GPT-2 und GPT-3 verwendeten GELU, das zuvor die
beste Wahl war. SwiGLU übertrifft GELU auf jeder Skalierungsstufe.

## Warum wir es anstelle von ReLU oder GELU verwenden

ReLU ist die einfachste Aktivierung. Sie gibt für negative Zahlen
null aus und tut bei positiven Zahlen nichts.

```
ReLU(x): max(0, x)

ReLU(-3.2) = 0    (blockiert)
ReLU(0.5)  = 0.5  (durchgelassen)
ReLU(4.1)  = 4.1  (durchgelassen)
```

Das Problem bei ReLU ist der harte Schnitt bei null. Jeder negative
Wert wird vollständig ausgelöscht. Die Information ist für immer
verloren. Man nennt das das Dying-ReLU-Problem. Neuronen, die nur
negative Eingaben erhalten, aktivieren sich nie wieder. Sie werden
zu totem Gewicht.

GELU behebt das, indem es den Schnitt weich macht. Anstelle einer
harten Null gibt GELU für negative Eingaben sehr kleine Werte aus.

```
GELU(-3.2) ≈ -0.002  (größtenteils blockiert, aber nicht tot)
GELU(0.5)  ≈ 0.346   (teilweise durchgelassen)
GELU(4.1)  ≈ 4.100   (größtenteils durchgelassen)
```

GELU ist besser als ReLU, hat aber immer noch nur einen
Entscheidungspunkt. Jede Eingabe wird gleich behandelt. Das Modell
hat keine Möglichkeit zu entscheiden, dass *diese* Eingabe mehr
durchgelassen werden soll als *jene* Eingabe.

SwiGLU fügt ein Gate hinzu. Die Eingabe wird in zwei Pfade
aufgeteilt. Ein Pfad berechnet Werte wie eine normale Aktivierung.
Der andere Pfad berechnet, wie viel von diesen Werten behalten
werden soll. Das Gate und die Werte werden aus derselben Eingabe
mit unterschiedlichen gelernten Gewichten berechnet.

```
SwiGLU(x) = (SiLU(x × W₁)) × (x × W₂)

Pfad 1 (Werte): SiLU(x × W₁) → die Information
Pfad 2 (Gate):  x × W₂       → wie viel Information durchgelassen wird
```

Das Gate kann eine beliebige Zahl ausgeben. Gibt das Gate 0.1 aus,
wird der Wertepfad auf zehn Prozent reduziert. Gibt das Gate 5.0
aus, wird der Wertepfad um das Fünffache verstärkt. Das Modell
lernt, was verstärkt und was unterdrückt werden soll. Deshalb
übertrifft SwiGLU sowohl ReLU als auch GELU bei großem Maßstab.

## Wann wurde es erfunden

Das Paper, das SwiGLU einführte, wurde 2020 von Noam Shazeer
veröffentlicht, einem bekannten Forscher, der den Transformer mit
erfunden hat. Das Paper verglich viele Aktivierungsvarianten und
stellte fest, dass Gated Linear Units durchgängig gewannen. PaLM
übernahm es 2022. LLaMA übernahm es 2023. Heute ist es der
Standard.

## Wie es Schritt für Schritt funktioniert

Verfolgen wir, wie eine einzelne Zahl durch SwiGLU fließt.

### Der Ausgangspunkt

```
Eingabe x = 1.5

Gewichte (gelernt während des Trainings):
W₁ = 0.8   (für den Wertepfad)
W₂ = 2.0   (für den Gate-Pfad)
```

### Pfad 1: den Wert berechnen

Zuerst wird die Eingabe mit W₁ multipliziert.

```
x × W₁ = 1.5 × 0.8 = 1.2
```

Dann wird SiLU angewendet. SiLU wird auch Swish-Funktion genannt.
Es ist x multipliziert mit dem Sigmoid von x.

```
SiLU(1.2) = 1.2 × sigmoid(1.2)

sigmoid(1.2) = 1 / (1 + e^(-1.2))
             = 1 / (1 + 0.301)
             = 1 / 1.301
             = 0.769

SiLU(1.2) = 1.2 × 0.769 = 0.922
```

SiLU ergibt 0.922. Das ist der verarbeitete Wert.

### Pfad 2: das Gate berechnen

Einfach die Eingabe mit W₂ multiplizieren.

```
x × W₂ = 1.5 × 2.0 = 3.0
```

Der Gate-Wert ist 3.0. Das bedeutet, dreimal so viel Information
durchzulassen. Das Gate ist weit geöffnet.

### Die beiden Pfade kombinieren

Den Wert mit dem Gate multiplizieren.

```
output = 0.922 × 3.0 = 2.766
```

Wäre das Gate kleiner gewesen, etwa 0.1, wäre die Ausgabe 0.092
gewesen. Wäre das Gate null gewesen, wäre die Ausgabe null
gewesen. Das Gate kontrolliert alles.

### Was ist mit negativen Eingaben

Versuchen wir es mit einer Eingabe von -2.0.

```
x = -2.0

Pfad 1 (Wert):
  x × W₁ = -2.0 × 0.8 = -1.6
  SiLU(-1.6) = -1.6 × sigmoid(-1.6)
  sigmoid(-1.6) = 1 / (1 + e^1.6) = 1 / 5.953 = 0.168
  SiLU(-1.6) = -1.6 × 0.168 = -0.269

Pfad 2 (Gate):
  x × W₂ = -2.0 × 2.0 = -4.0

Kombinieren:
  output = -0.269 × (-4.0) = 1.076
```

Obwohl die Eingabe negativ war, ist die Ausgabe positiv. Das
liegt daran, dass sowohl der Wertepfad als auch der Gate-Pfad
negativ wurden, und negativ mal negativ ergibt positiv. Der
Gating-Mechanismus gibt dem Modell zusätzliche Flexibilität,
negative Signale bei Bedarf in positive umzuwandeln. ReLU hätte
einfach null ausgegeben und alle Information verloren.

## Ein kleines Codebeispiel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = expansion_factor * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # Pfad 1: Werte, verarbeitet durch SiLU
        values = F.silu(self.w1(x))
        # Pfad 2: Gates, die steuern, wie viel durchgelassen wird
        gates = self.w2(x)
        # Kombinieren und zurück auf die ursprüngliche Größe projizieren
        return self.w3(values * gates)

# Test mit zufälliger Eingabe
d_model = 4
ffn = SwiGLU(d_model)
x = torch.tensor([[1.5, -2.0, 0.3, 4.1]])

output = ffn(x)
print(f"Input:  {x}")
print(f"Output: {output}")
print(f"Shape preserved: {x.shape == output.shape}")
```

Wenn du diesen Code ausführst, siehst du etwa Folgendes:

```
Input:  tensor([[ 1.5000, -2.0000,  0.3000,  4.1000]])
Output: tensor([[-1.234,  0.567, -0.891,  2.345]])
Shape preserved: True
```

## Warum der Expansionsfaktor wichtig ist

Beachte, dass die versteckte Dimension im Code viermal so groß ist
wie die Eingabedimension. Das ist der Expansionsfaktor. Das
Netzwerk geht von d_model auf das Vierfache von d_model und wieder
zurück.

```
768 → 3072 → 768
```

Dieses Muster aus Erweitern und dann Verengen gibt dem Netzwerk
Raum, um Information zu transformieren. In der mittleren Layer
gibt es viel mehr Neuronen als am Eingang oder Ausgang. Das ist,
als würde man ein Rohr verbreitern, damit mehr Wasser hindurch-
fließen kann, bevor man es wieder verengt. Die zusätzliche Breite
erlaubt es dem Modell, komplexere Transformationen zu lernen.

SwiGLU verwendet drei Gewichtsmatrizen statt der zwei, die ReLU-
oder GELU-Netzwerke verwenden. Die zusätzliche Matrix ist für das
Gate. Dadurch wird SwiGLU bei gleichem Expansionsfaktor etwa
fünfzig Prozent größer als ein Standard-FFN. Für unser Modell im
GPT-2-Maßstab bedeutet das etwa achtundzwanzig Millionen
zusätzliche Parameter. Jeder einzelne dieser Parameter trägt zu
besserer Leistung bei.

## Was du dir merken solltest

SwiGLU ist eine Gate-gesteuerte Aktivierungsfunktion. Sie teilt
das Feed-Forward-Netzwerk in einen Wertepfad und einen Gate-Pfad
auf. Das Gate steuert, wie viel von jedem Wert durchgelassen wird.
Das ist flexibler als ReLU oder GELU, die jede Eingabe gleich
behandeln.

Die SiLU-Funktion auf dem Wertepfad sorgt für eine glatte
Nichtlinearität. Das Gate auf dem Kontrollpfad sorgt für adaptive
Filterung. Zusammen übertreffen sie jede ältere Aktivierungs-
funktion bei großem Maßstab.

Jedes moderne Sprachmodell verwendet SwiGLU. Es ist eine
zusätzliche Matrixmultiplikation pro Forward Pass für eine
messbare Verbesserung in jeder Metrik, die zählt.
