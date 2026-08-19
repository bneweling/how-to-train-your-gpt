# Residual Connections: Der Gradienten-Highway

## Was ist das

Eine Residual Connection ist eine Abkürzung, die es Informationen
erlaubt, an einem Layer vorbeizulaufen. Statt den Input zu ersetzen,
addiert der Layer etwas zu ihm hinzu.

```
Ohne Residual:  output = layer(input)
Mit Residual:   output = input + layer(norm(input))
```

Stell es dir vor wie das Bearbeiten eines Dokuments. Ohne Residual
Connection wirfst du das Original weg und schreibst einen komplett
neuen Entwurf. Mit einer Residual Connection behältst du das
Original und nimmst nur kleine Korrekturen daran vor. Das Original
ist immer darunter vorhanden. Die Änderungen sind inkrementell.

Das wirkt wie ein kleiner Unterschied. Es ist die wichtigste
einzelne Design-Entscheidung, die tiefe neuronale Netze überhaupt
möglich macht. Ohne Residual Connections lässt sich ein Netz nicht
tiefer als etwa zwanzig Layer trainieren. Mit ihnen lassen sich
Netze mit Hunderten oder sogar Tausenden von Layern trainieren. Der
Unterschied ist keine Frage der Bequemlichkeit. Es ist der
Unterschied zwischen einem Modell, das lernt, und einem Modell, das
gar nichts tut.

## Wo wird es eingesetzt

Residual Connections umschließen jeden Sublayer im
Transformer-Block. Jeder Attention-Layer hat eine. Jeder
Feed-Forward-Layer hat eine. Bei einem Modell mit zwölf Blöcken gibt
es vierundzwanzig Residual Connections.

```
Transformer-Block:
  x → RMSNorm → Attention → +x  ← Residual hier
    → RMSNorm → SwiGLU   → +x  ← Residual hier
```

Ohne diese Pluszeichen könnte das Modell nicht trainiert werden. Die
ersten Layer würden kein Gradientensignal erhalten und sich nie
aktualisieren. Das Modell wäre für immer mit zufälligen Weights
festgefahren.

## Warum wir es brauchen: das Vanishing-Gradient-Problem

Um zu verstehen, warum Residual Connections wichtig sind, müssen
wir verstehen, wie neuronale Netze lernen.

Wenn das Modell eine Vorhersage macht und sie falsch ist, berechnet
es einen Loss. Dann wird ermittelt, wie stark jedes Weight zu
diesem Loss beigetragen hat. Diese Frage wandert rückwärts durch
das Netz, vom letzten Layer bis zum ersten. Bei jedem Layer wird
das Signal mit einer Zahl multipliziert, die man Weight-Gradient
nennt.

Ist der Weight-Gradient kleiner als eins, schrumpft das Signal bei
jedem Layer. Nach zehn Layern rückwärts ist das Signal winzig. Nach
zwanzig Layern ist es mikroskopisch klein. Nach hundert Layern ist
es praktisch null. Die ersten Layer erhalten überhaupt kein
Lernsignal. Sie bleiben für immer zufällig.

```
Gradient bei Layer 1 = Gradient bei Layer 100 × w₁ × w₂ × ... × w₉₉

Wenn jedes Weight 0.5 ist:
Gradient bei Layer 1 = Gradient bei Layer 100 × 0.5⁹⁹
                    = Gradient bei Layer 100 × 0.00000000000000000000000000000016
                    ≈ 0
```

Das ist das Vanishing-Gradient-Problem. Deshalb waren tiefe Netze
jahrzehntelang nicht trainierbar. Forscher versuchten es mit
größeren Computern und besseren Optimizern, aber nichts half. Die
Mathematik des Multiplizierens kleiner Zahlen gewinnt immer.

Residual Connections lösen dieses Problem, indem sie einen zweiten
Pfad hinzufügen. Der Gradient kann wie zuvor rückwärts durch den
Layer wandern. Oder er kann den Layer komplett überspringen und
direkt zum Input gelangen.

```
Ohne Residual:
  output = layer(input)
  Gradientenpfad: input ← layer ← loss (muss durch den Layer gehen)

Mit Residual:
  output = input + layer(input)
  Gradientenpfad: input ← loss (direkter Pfad, Gradient immer 1.0)
                input ← layer ← loss (indirekter Pfad, kann klein sein)
```

Der direkte Pfad liefert immer einen Gradienten von exakt 1.0. Egal
wie klein der Gradient des Layers ist, der direkte Pfad stellt
sicher, dass jeder Layer wenigstens ein gewisses Lernsignal erhält.
Das Signal verschwindet nie vollständig.

## Wann wurde es erfunden

Residual Connections wurden 2015 von Forschern bei Microsoft in
einem Paper über Bilderkennung eingeführt. Sie zeigten, dass ein
Netz mit 152 Layern und Residual Connections ein Netz mit 19 Layern
ohne diese übertraf. Die Idee wurde 2017 von den Autoren des
Transformer-Papers übernommen. Heute werden Residual Connections in
praktisch jedem Deep-Learning-Modell eingesetzt, unabhängig von der
Architektur.

## Wie es funktioniert: ein konkretes Beispiel

Verfolgen wir, wie eine einzelne Zahl durch eine Residual Connection
fließt.

### Ohne Residual

```
Input x = 2.0

Der Attention-Layer verarbeitet ihn:
attention_output = 0.1

Finaler Output = 0.1
```

Der ursprüngliche Wert von 2.0 ist vollständig verschwunden. Der
Layer hat ihn ersetzt. Wenn der Layer Müll ausgibt, wird dieser
Müll zum neuen Input für den nächsten Layer. Garbage in, Garbage
out.

### Mit Residual

```
Input x = 2.0

RMSNorm normalisiert ihn: norm(x) = 1.5
Der Attention-Layer verarbeitet ihn: attention(norm(x)) = 0.1

Finaler Output = x + attention(norm(x))
             = 2.0 + 0.1
             = 2.1
```

Der ursprüngliche Wert von 2.0 bleibt erhalten. Der Layer hat eine
kleine Korrektur von 0.1 hinzugefügt. Der Output liegt sehr nah am
Input. Wenn der Layer Müll ausgibt, lässt die Residual Connection
trotzdem den guten Input durch. Das Modell kann einen schlechten
Layer überstehen.

### Was das für das Lernen bedeutet

Das Modell muss den korrekten Output nicht bei jedem Layer von
Grund auf neu lernen. Es muss nur lernen, welche *Änderung* am
Input vorzunehmen ist. Das ist ein viel einfacheres Problem.

```
Lernziel ohne Residual: "Erzeuge die Zahl 2.1"
Lernziel mit Residual:  "Addiere 0.1 zum Input"
```

Das zweite Ziel ist einfacher, weil der Layer anfangs null ausgibt.
Bei der Initialisierung mit kleinen Weights geben die meisten Layer
eines neuronalen Netzes Werte sehr nahe null aus. Der
Residual-Block verhält sich also anfangs wie eine
Identitätsfunktion. Es ändert sich nichts. Im Laufe des Trainings
lernt das Modell dann, sinnvolle Deltas hinzuzufügen. Die
Architektur begünstigt, dass das Modell seinen Input bewahrt und
kleine Verbesserungen vornimmt. Genau das wollen wir.

## Ein winziges Code-Beispiel

```python
import torch
import torch.nn as nn

# Ein einfacher Layer mit und ohne Residual
class NoResidual(nn.Module):
    def forward(self, x):
        return torch.tanh(x)  # Nur der Layer-Output

class WithResidual(nn.Module):
    def forward(self, x):
        return x + torch.tanh(x)  # Input plus Layer-Output

x = torch.tensor([2.0, -1.0, 0.5, -3.0])

no_res = NoResidual()
with_res = WithResidual()

print(f"Input:           {x}")
print(f"Without residual: {no_res(x)}")
print(f"With residual:    {with_res(x)}")
print()
print("Without residual the output is bounded between -1 and 1.")
print("The original information is lost forever.")
print()
print("With residual the output is the input plus a small correction.")
print("The original information is always preserved in the sum.")
```

Wenn du diesen Code ausführst, siehst du etwa Folgendes:

```
Input:           tensor([ 2.0000, -1.0000,  0.5000, -3.0000])
Without residual: tensor([ 0.9640, -0.7616,  0.4621, -0.9950])
With residual:    tensor([ 2.9640, -1.7616,  0.9621, -3.9950])
```

Der Output ohne Residual wird in den Bereich von minus eins bis
eins gequetscht. Alle Information über die Größenordnung des Inputs
ist verloren. Der Output mit Residual bewahrt die ursprünglichen
Werte und addiert kleine Anpassungen obendrauf.

## Der Gradiententest

Wir können den Gradientenfluss tatsächlich messen. Stapeln wir
viele Layer übereinander und schauen wir, welche Variante den
Gradienten überleben lässt.

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0], requires_grad=True)
layer = nn.Linear(1, 1)

# 50 Layer OHNE Residuals stapeln
current = x
for _ in range(50):
    current = torch.tanh(layer(current))

current.backward()
print(f"Gradient after 50 layers WITHOUT residuals: {x.grad.item():.10f}")

# 50 Layer MIT Residuals stapeln
x.grad = None
current = x
for _ in range(50):
    current = current + torch.tanh(layer(current))

current.backward()
print(f"Gradient after 50 layers WITH residuals:    {x.grad.item():.4f}")
```

Wenn du diesen Code ausführst, siehst du etwa Folgendes:

```
Gradient after 50 layers WITHOUT residuals: 0.0000000000
Gradient after 50 layers WITH residuals:    0.2314
```

Ohne Residuals verschwindet der Gradient nach fünfzig Layern
vollständig. Der erste Layer kann nichts lernen. Mit Residuals
bleibt der Gradient gesund. Jeder Layer kann lernen.

## Das mentale Modell

Stell dir ein tiefes neuronales Netz vor, das versucht, eine
komplizierte Funktion zu lernen. Die Funktion könnte etwa lauten:
*verstehe diesen Textabschnitt*. Ohne Residuals muss das Netz diese
Funktion bei jedem Layer von Grund auf neu lernen. Jeder Layer muss
das Ganze aus dem rohen Input herausfinden. Das ist schwer.

Mit Residuals muss jeder Layer nur die *Differenz* zwischen dem
perfekten Output und dem aktuellen Output lernen. Der erste Layer
lernt ein bisschen. Der zweite Layer verfeinert. Der dritte Layer
verfeinert weiter. Jeder Layer nimmt eine kleine Verbesserung an
dem vor, was zuvor kam. Das ist wie Bildhauerei. Du fängst mit
einem Steinblock an. Du schlägst ein bisschen ab. Du schlägst noch
ein bisschen ab. Am Ende hast du eine Statue. Du hast den
ursprünglichen Block nie weggeworfen. Du hast ihn nur verfeinert.

## Was du dir merken musst

Residual Connections lassen den Input an jedem Layer vorbeilaufen
und zum Output addieren. Das schafft einen direkten Pfad, über den
Gradienten rückwärts durch das gesamte Netz fließen können, ohne
bei jedem Schritt mit kleinen Zahlen multipliziert zu werden.

Ohne Residual Connections leiden tiefe Netze unter Vanishing
Gradients und lassen sich nicht trainieren. Mit Residual
Connections überleben Gradienten sogar durch Hunderte von Layern
hindurch. Deshalb kann GPT-3 sechsundneunzig Layer haben und
trotzdem effektiv lernen. Der Gradienten-Highway bleibt vom letzten
bis zum ersten Layer durchgehend offen.

Die Lösung ist ein einziges Pluszeichen. Output gleich Input plus
Layer-Output. Diese eine Addition macht Deep Learning möglich.
