# Weight Tying: Zwei Aufgaben, eine Matrix

## Was ist das

Weight Tying bedeutet, dieselbe Weight-Matrix für den Embedding-
Layer und den Output-Layer zu verwenden. Der Embedding-Layer
wandelt Token-IDs in Vektoren um. Der Output-Layer wandelt
Vektoren zurück in Token-Wahrscheinlichkeiten. Es sind inverse
Operationen. Sie teilen sich eine Matrix.

Man kann sich das wie ein zweisprachiges Wörterbuch vorstellen.
Man kann ein englisches Wort nachschlagen, um seine französische
Übersetzung zu finden. Oder man kann ein französisches Wort
nachschlagen, um seine englische Übersetzung zu finden. Es ist
dasselbe Wörterbuch, das in zwei Richtungen genutzt wird. Weight
Tying ist dieselbe Idee, angewendet auf neuronale Netze. Eine
Matrix dient sowohl der Eingabeseite als auch der Ausgabeseite.

## Wo wird es eingesetzt

Weight Tying verbindet den allerersten Layer und den allerletzten
Layer des Modells. Das Token-Embedding und der Language-Model-
Head teilen sich die Weights.

```
Input-Tokens
  → Token-Embedding (Matrix A der Größe 50257 × 768)
    → Transformer-Blöcke
      → Finale Normalisierung
        → LM-Head (teilt sich Matrix A, verwendet als 768 × 50257)
          → Output-Logits
```

Im Code ist es eine einzige Zeile.

```python
self.token_embedding.weight = self.lm_head.weight
```

Dadurch zeigen beide Attribute auf denselben Tensor im Speicher.
Ändert man das eine, ändert sich das andere, weil es buchstäblich
dieselben Bytes sind.

## Warum wir es verwenden

Der offensichtlichste Grund ist die Einsparung von Parametern.
Die Embedding-Matrix hat die Größe Vokabular mal Embedding-
Dimension. Bei GPT-2 Small sind das 50257 mal 768, was etwa 38,6
Millionen Zahlen ergibt. Die Output-Matrix hat dieselbe Größe.
Ohne Tying wären das zwei separate Matrizen, die zusammen etwa 77
Millionen Parameter allein für Ein- und Ausgabe verbrauchen. Mit
Tying werden sie zu einer Matrix. Wir sparen 38,6 Millionen
Parameter.

Das sind etwa dreißig Prozent der gesamten Modellgröße von GPT-2
Small. Bei größeren Modellen fallen die Einsparungen noch größer
aus. GPT-3 Large mit einem Vokabular von 50257 und einer
Embedding-Dimension von 12288 würde über 600 Millionen Parameter
für eine zweite Kopie der Embedding-Matrix verschwenden. Diese
Parameter lassen sich besser für weitere Transformer-Blöcke
einsetzen.

Der weniger offensichtliche Grund ist besseres Lernen. Die
Embedding-Matrix ist das Tor in das Modell hinein. Jedes Token
durchläuft sie auf dem Weg hinein. Die Output-Matrix ist das Tor
aus dem Modell heraus. Jede Vorhersage durchläuft sie auf dem Weg
hinaus. Wenn das Modell eine Vorhersage falsch trifft, fließt der
Gradient rückwärts durch die Output-Matrix und bis zur Embedding-
Matrix. Da es sich um dieselbe Matrix handelt, erhalten die
Embedding-Vektoren Gradientensignale aus zwei Richtungen. Der
Forward Pass durch den Embedding-Layer und der Backward Pass
durch den Output-Layer aktualisieren beide dieselben Zahlen.
Dieses doppelte Signal hilft jedem Embedding-Vektor, zu einer
besseren Repräsentation zu konvergieren.

Der dritte Grund ist mathematische Eleganz. Der Embedding-Layer
bildet Token-IDs auf Vektoren ab. Der Output-Layer bildet Vektoren
auf Token-Wahrscheinlichkeiten ab. Wenn das Modell gute Embeddings
gelernt hat, sollten dieselben Vektoren, die ein Token auf der
Eingabeseite repräsentieren, auch dafür nützlich sein, dieses
Token auf der Ausgabeseite vorherzusagen. Das Tying der Weights
erzwingt diese Konsistenz. Der Embedding-Vektor eines Tokens ist
zugleich der Vektor, mit dem das Modell dieses Token als mögliches
nächstes Wort bewertet. Wenn das Token *cat* den Embedding-Vektor
v hat, dann ist der Score des Modells für die Vorhersage von *cat*
das Skalarprodukt des aktuellen Hidden State mit v. Das Embedding
übernimmt eine Doppelrolle: als Repräsentation und als
Klassifikationsgewicht.

## Wann wurde es erfunden

Weight Tying wurde bereits im ursprünglichen Transformer-Paper aus
dem Jahr 2017 verwendet. Es war zu diesem Zeitpunkt keine neue
Idee. Frühere Sprachmodelle wie word2vec, das 2013 veröffentlicht
wurde, verwendeten bereits verknüpfte Input- und Output-Embeddings.
Seitdem ist es gängige Praxis für Sprachmodelle. GPT-2 und GPT-3
verwenden beide Weight Tying. LLaMA verwendet Weight Tying. Jedes
Modell in diesem Tutorial verwendet Weight Tying.

Es gibt Fälle, in denen Weight Tying nicht eingesetzt wird. Manche
sehr großen Modelle trennen die Embedding- und die Output-Matrix,
um dem Output-Layer eine andere Struktur als dem Input-Layer zu
ermöglichen. Aber für die meisten Modelle, einschließlich unserem,
ist Weight Tying die richtige Wahl. Die Parametereinsparungen sind
zu groß, um sie zu ignorieren, und das doppelte Gradientensignal
ist beim Training wirklich hilfreich.

## Wie es in der Praxis funktioniert

Verfolgen wir Schritt für Schritt, was passiert, wenn wir mit
verknüpften Weights trainieren.

### Forward Pass mit verknüpften Weights

```
Schritt 1: Token 3797 ("cat") betritt das Modell
Schritt 2: Der Embedding-Layer schlägt Zeile 3797 der Matrix A nach
Schritt 3: Zeile 3797 ist der Embedding-Vektor für "cat" [768 Zahlen]
Schritt 4: Der Vektor durchläuft die Transformer-Blöcke
Schritt 5: Der Hidden State erreicht den Output-Layer
Schritt 6: Der Output-Layer multipliziert den Hidden State mit Matrix A^T
Schritt 7: Zeile 3797 von A^T ist Spalte 3797 von A
Schritt 8: Das ist derselbe Vektor, der "cat" bei der Eingabe repräsentiert hat
Schritt 9: Das Skalarprodukt ergibt den Score für die Vorhersage von "cat"
```

Man beachte, dass derselbe Vektor zweimal auftaucht. Einmal als
Repräsentation von *cat* bei der Eingabe. Einmal als
Vorhersageziel für *cat* bei der Ausgabe. Das Modell wird
gezwungen, diese beiden Verwendungen konsistent zu halten.

### Backward Pass mit verknüpften Weights

```
Schritt 1: Das Modell sagt falsch vorher (das richtige Wort war "mat", nicht "dog")
Schritt 2: Der Loss wird berechnet
Schritt 3: Der Gradient fließt zum Output-Layer
Schritt 4: Der Gradient aktualisiert Zeile 3797 der Matrix A
        (weil cat eine der falschen Vorhersagen war)
Schritt 5: Derselbe Gradient fließt auch zurück durch das Modell
Schritt 6: Erreicht schließlich den Embedding-Layer
Schritt 7: Zeile 3797 der Matrix A erhält ein zweites Gradientensignal
        (weil cat in der Eingabe vorkam)
Schritt 8: Beide Gradienten werden summiert und auf dieselben Zahlen angewendet
```

Das Embedding für *cat* wird pro Trainingsschritt zweimal
aktualisiert. Einmal für seine Rolle als Input-Token. Einmal für
seine Rolle als potenzielles Output-Token. Dieses doppelte Signal
führt dazu, dass die Embedding-Vektoren schneller lernen und
bessere Repräsentationen erreichen.

## Weight Tying im Code überprüfen

Man kann in PyTorch überprüfen, ob sich zwei Tensoren denselben
Speicher teilen.

```python
import torch

# Erstelle ein Embedding und einen Output-Layer
embedding = torch.nn.Embedding(1000, 768)
output = torch.nn.Linear(768, 1000, bias=False)

# Verknüpfe die Weights
embedding.weight = output.weight

# Überprüfe, ob sie sich den Speicher teilen
print(f"Same object: {embedding.weight is output.weight}")
print(f"Same memory: {embedding.weight.data_ptr() == output.weight.data_ptr()}")

# Ändere eines und beobachte, wie sich das andere ändert
old_value = embedding.weight[42, 0].item()
output.weight[42, 0] = 99.9
new_value = embedding.weight[42, 0].item()

print(f"\nAfter changing output.weight[42,0] to 99.9:")
print(f"Embedding weight[42,0] changed from {old_value} to {new_value}")
print(f"They are the same tensor. Changing one changes both.")
```

Die Ausführung dieses Codes ergibt:

```
Same object: True
Same memory: True

After changing output.weight[42,0] to 99.9:
Embedding weight[42,0] changed from 0.023 to 99.9
They are the same tensor. Changing one changes both.
```

## Die Parametereinsparung nach Modellgröße

```
Modell             Vokabular  Dim      Weight-Tying-Einsparung
GPT-2 Small        50,257   × 768   =  38.6 Millionen Parameter
GPT-2 Medium       50,257   × 1,024 =  51.5 Millionen Parameter
GPT-2 Large        50,257   × 1,280 =  64.3 Millionen Parameter
LLaMA 7B           32,000   × 4,096 = 131.1 Millionen Parameter
LLaMA 70B          32,000   × 8,192 = 262.1 Millionen Parameter
GPT-3 (full)       50,257   × 12,288 = 617.6 Millionen Parameter
```

Diese Einsparungen sind der Grund, warum Weight Tying nahezu
universell eingesetzt wird. Man bekommt bessere Embeddings und
besseres Training zum Preis von null zusätzlichen Parametern.
Tatsächlich bekommt man gleichzeitig weniger Parameter und
besseres Training. Es ist einer der seltenen Fälle im Machine
Learning, in denen es keinen Tradeoff gibt.

## Was man sich merken sollte

Weight Tying sorgt dafür, dass sich der Embedding-Layer und der
Output-Layer dieselbe Weight-Matrix teilen. Eine einzige Codezeile
spart Dutzende oder Hunderte Millionen Parameter. Die gemeinsame
Matrix erhält Gradientensignale sowohl aus der Input- als auch aus
der Output-Richtung, was zu besseren Embeddings führt. Jedes
moderne Sprachmodell nutzt diese Technik. Es ist bessere
Performance zum Nulltarif.
