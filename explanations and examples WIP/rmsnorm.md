# RMSNorm: Die einfachste Normalisierung, die funktioniert

## Was es ist

RMSNorm steht für Root Mean Square Normalization. Es ist eine
winzige Rechenoperation, die verhindert, dass Zahlen in einem
neuronalen Netz zu groß werden oder zu klein schrumpfen.

Stell dir vor, du stapelst Holzklötze. Jeder Klotz liegt auf dem
darunter. Wird ein Klotz breiter, neigt sich der Turm. Wird ein
Klotz dünner, fällt der Turm um. Nach dem Stapeln von hundert
Klötzen summieren sich die Größenunterschiede und der Turm stürzt
ein. Ein neuronales Netz hat dasselbe Problem. Werte fließen durch
Dutzende von Layern. Kleine Unterschiede am Anfang werden am Ende
zu riesigen Unterschieden. Das Modell wird instabil und hört auf
zu lernen.

RMSNorm behebt das, indem es die Ausgabe jedes Layers so skaliert,
dass alle Werte in einem gesunden Bereich bleiben. Bevor die
Zahlen in den nächsten Layer gehen, sorgt RMSNorm dafür, dass ihre
durchschnittliche Größe gleich eins ist. Jeder Layer erhält
saubere, ausgewogene Eingaben. Der Turm bleibt gerade.

## Wo es eingesetzt wird

RMSNorm steht vor jedem Attention-Layer und jedem Feed-Forward-
Layer innerhalb eines Transformer-Blocks. Zwei Normalisierungen
pro Block. Bei einem Modell mit zwölf Blöcken sind das
vierundzwanzig RMSNorm-Operationen für jeden Satz, den das Modell
liest.

```
Transformer-Block:
  x → RMSNorm → Attention → +x
    → RMSNorm → SwiGLU   → +x
```

Es ist das Allererste, was passiert, wenn Zahlen in den
Attention-Layer eintreten, und das Allererste, was passiert, wenn
Zahlen in den Feed-Forward-Layer eintreten. Es ist der Türsteher.
Keine ungebändigten Zahlen kommen durch.

## Warum wir es statt LayerNorm verwenden

Das ursprüngliche Transformer-Paper verwendete LayerNorm.
LayerNorm macht drei Dinge mit jedem Vektor. Es subtrahiert den
Mittelwert. Es dividiert durch die Standardabweichung. Dann
wendet es eine gelernte Skalierung und Verschiebung an. Zwei
dieser Schritte sind unnötig.

Das Subtrahieren des Mittelwerts sollte eigentlich helfen, aber
Experimente zeigten, dass es keinen Unterschied macht. Die
Residual Connections übernehmen die Zentrierung bereits implizit.
Der lernbare Verschiebungsparameter war aus demselben Grund
ebenfalls unnötig.

RMSNorm verzichtet auf beides. Es berechnet nur das quadratische
Mittel und dividiert dadurch. Keine Mittelwertsubtraktion. Kein
Verschiebungsparameter. Nur ein einfacher Skalierungsfaktor, den
das Modell lernen kann.

```
LayerNorm(x): ((x - mean) / std) × weight + bias   (4 Operationen)
RMSNorm(x):   (x / rms(x)) × weight                (2 Operationen)
```

Das Ergebnis ist mathematisch einfacher und etwa fünfzehn Prozent
schneller. Das Modell trainiert genauso gut. Jedes moderne
Sprachmodell, einschließlich LLaMA, Mistral und Gemma, verwendet
RMSNorm.

## Wann wurde es erfunden

RMSNorm wurde 2019 von Forschern bei Microsoft veröffentlicht. Sie
zeigten, dass man die Mittelwertzentrierung und den Bias aus
LayerNorm entfernen kann, ohne an Qualität einzubüßen. Die Idee
wurde 2023 vom LLaMA-Team bei Meta aufgegriffen. Sobald das
beliebteste Open-Source-Modell es verwendete, stiegen alle um.
Heute ist LayerNorm in neuen Modellen selten.

## Wie es Schritt für Schritt funktioniert

Gehen wir ein konkretes Beispiel mit einem Vektor aus vier Zahlen
durch, der durch einen Netzwerk-Layer fließt.

### Schritt 1: Die Zahlen kommen an

```
x = [3.2, -1.5, 0.8, -4.1]
```

Diese Zahlen kamen aus einem Attention-Layer. Manche sind groß.
Manche sind negativ. Wenn wir sie direkt an den nächsten Layer
weitergeben, könnte die Mathematik extreme Ergebnisse liefern.

### Schritt 2: Jede Zahl quadrieren

```
x² = [10.24, 2.25, 0.64, 16.81]
```

Durch das Quadrieren werden alle Zahlen positiv und Ausreißer
werden verstärkt. Das negative Vorzeichen von -4.1 verschwindet
beim Quadrieren.

### Schritt 3: Den Mittelwert der Quadrate bilden

```
mean(x²) = (10.24 + 2.25 + 0.64 + 16.81) / 4
         = 29.94 / 4
         = 7.485
```

Das sagt uns die durchschnittliche Energie des Vektors. Ein Wert
von 7.5 bedeutet, dass der Vektor recht weit gestreut ist.

### Schritt 4: Die Quadratwurzel ziehen, um den RMS zu erhalten

```
rms = sqrt(7.485) = 2.736
```

Das quadratische Mittel liegt bei etwa 2.7. Das ist die typische
Größenordnung einer Zahl in diesem Vektor. Die meisten Werte
liegen im Durchschnitt etwa 2.7 von null entfernt.

### Schritt 5: Jede Zahl durch den RMS dividieren

```
x / rms = [3.2/2.736, -1.5/2.736, 0.8/2.736, -4.1/2.736]
        = [1.169, -0.548, 0.292, -1.498]
```

Jetzt ist das quadratische Mittel dieses neuen Vektors genau 1.0.
Jede Zahl wurde proportional herunterskaliert. Die Form des
Vektors bleibt erhalten. Nur ihre Größe hat sich geändert.

### Schritt 6: Ein gelerntes Gewicht pro Dimension anwenden

Das Modell hat einen Gewichtsparameter für jede Dimension. Er
startet bei 1.0 und wird während des Trainings gelernt.

```
weight = [0.95, 1.12, 0.88, 1.05]  (während des Trainings gelernt)

output = [1.169×0.95, -0.548×1.12, 0.292×0.88, -1.498×1.05]
       = [1.111, -0.614, 0.257, -1.573]
```

Das Gewicht lässt das Modell entscheiden, welche Dimensionen
lauter und welche leiser sein sollen. Ein Gewicht über 1.0
verstärkt diese Dimension. Ein Gewicht unter 1.0 dämpft sie. Das
ist der einzige lernbare Teil von RMSNorm. Alles andere ist reine
Mathematik ohne Parameter.

## Warum das Gewicht wichtig ist

Ohne das gelernte Gewicht wäre das Modell gezwungen, jede
Dimension nach der Normalisierung auf exakt derselben Skala zu
halten. Das ist zu einschränkend. Manche Dimensionen tragen
wichtigere Informationen als andere. Das Gewicht erlaubt es dem
Modell, diese Unterschiede nach der Normalisierung zu bewahren.

Stell es dir vor wie das Einstellen der Lautstärke verschiedener
Instrumente in einem Song. RMSNorm sorgt dafür, dass die
Gesamtlautstärke immer gleich bleibt. Die Gewichte erlauben es dem
Schlagzeug, etwas lauter zu sein, und der Violine, etwas leiser zu
sein, während diese konstante Gesamtlautstärke erhalten bleibt.

## Ein kleines Codebeispiel

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        # Ein gelerntes Gewicht pro Dimension
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps  # Winzige Zahl, um Division durch Null zu verhindern

    def forward(self, x):
        # Jede Zahl quadrieren und über die letzte Dimension mitteln
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalisieren und gelerntes Gewicht anwenden
        return x * rms * self.weight

# Testen
norm = RMSNorm(d_model=4)
x = torch.tensor([[3.2, -1.5, 0.8, -4.1]])

output = norm(x)
print(f"Input:  {x}")
print(f"Output: {output}")
print()

# Prüfen, dass der RMS 1 ist
rms_check = torch.sqrt(output.pow(2).mean())
print(f"RMS of output: {rms_check.item():.4f}")
print(f"Close to 1.0: {abs(rms_check.item() - 1.0) < 0.01}")
```

Wenn du diesen Code ausführst, siehst du etwa Folgendes:

```
Input:  tensor([[ 3.2000, -1.5000,  0.8000, -4.1000]])
Output: tensor([[ 1.1694, -0.5482,  0.2924, -1.4984]])

RMS of output: 1.0000
Close to 1.0: True
```

Die Ausgabe hat dieselbe Form wie die Eingabe. Die relativen
Größenverhältnisse der vier Zahlen bleiben erhalten. Nur die
Gesamtskala wurde geändert, damit der RMS exakt eins ergibt.

## RMSNorm vs. LayerNorm vs. keine Normalisierung

Was passiert, wenn du die Normalisierung bei einem tiefen
Transformer mit sechsundneunzig Layern komplett entfernst.

```
Ohne Normalisierung:  Die Werte driften. Bis Layer 50 sind manche
Zahlen 100-mal so groß wie ursprünglich. Andere nur noch 0.01-mal
so groß. Das Modell kann nicht lernen. Das Training divergiert.

Mit LayerNorm:  Die Werte bleiben kontrolliert. Das Modell
trainiert, aber die Mittelwertzentrierung und der Bias kosten
Rechenleistung, ohne zu helfen. Etwas langsamer als nötig.

Mit RMSNorm:  Die Werte bleiben kontrolliert. Das Modell
trainiert. Keine verschwendete Rechenleistung für
Mittelwertzentrierung. Die schnellste Option, die funktioniert.
```

## Was du dir merken musst

RMSNorm verhindert, dass Zahlen explodieren oder verschwinden,
während sie durch Dutzende von Transformer-Layern fließen. Es
dividiert jeden Vektor durch sein quadratisches Mittel, um die
durchschnittliche Größe auf exakt 1.0 zu erzwingen. Anschließend
lässt es das Modell Gewichte pro Dimension lernen, um einzelne
Lautstärken anzupassen.

Es ist einfacher und schneller als LayerNorm, weil es zwei
unnötige Schritte überspringt. Jedes moderne Sprachmodell
verwendet es. Es ist eines dieser kleinen Details, die den
Unterschied ausmachen zwischen einem Modell, das trainiert, und
einem Modell, das nach zwanzig Layern in Unsinn abdriftet.
