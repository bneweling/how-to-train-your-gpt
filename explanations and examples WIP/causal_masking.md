# Causal Masking: Kein Blick in die Zukunft

## Was ist das

Causal Masking ist eine Regel, die verhindert, dass jedes Wort in
einem Satz die Wörter sehen kann, die danach kommen. Nehmen wir
einen Satz wie *The cat sat on the mat*: Das Wort *cat* kann *The*
sehen, aber nicht *sat*. Das Wort *sat* kann *The* und *cat* sehen,
aber nicht *on*. Jedes Wort tappt im Dunkeln darüber, was als
Nächstes kommt.

Stell es dir vor wie beim Lesen eines Buches, Seite für Seite. Du
hast die Seiten eins bis fünf gelesen. Seite sechs liegt umgedreht
auf dem Tisch. Du darfst nicht hinüberschauen. Du musst erraten,
was auf Seite sechs passiert, nur mit dem, was du bereits gelesen
hast.

Diese Regel wirkt einschränkend. Warum lässt man das Modell nicht
einfach alles sehen? Weil die gesamte Aufgabe des Modells darin
besteht, das nächste Wort vorherzusagen. Könnte das Modell das
nächste Wort sehen, müsste es es nicht vorhersagen. Es würde es
einfach kopieren. Das ist kein Lernen. Das ist Schummeln.

## Wo wird es verwendet

Die Causal Mask befindet sich innerhalb des Attention-Layers.
Direkt nachdem die Attention-Scores berechnet wurden und direkt
bevor der Softmax sie in Prozentwerte umwandelt.

```
Attention-Scores → Causal Mask anwenden → Softmax → Attention-Weights
```

Jeder Attention-Layer in jedem Transformer-Block wendet die Causal
Mask an. Bei einem Modell mit zwölf Blöcken wird die Maske für
jeden Satz zwölfmal angewendet. Immer dieselbe Maske. Immer
dieselbe Regel. Kein Blick in die Zukunft.

## Warum wir sie brauchen

Während des Trainings sieht das Modell vollständige Sätze. Der
Input ist *The cat sat on the mat* auf einmal. Ohne die Causal
Mask könnte das Wort *cat* auf *mat* schauen und sagen: *Ah, der
Satz endet mit mat, also muss auf cat sat folgen*. Das Modell
würde eine triviale Abbildung von vollständigen Sätzen auf sich
selbst lernen. Es würde nie lernen, vorherzusagen. Es würde nur
lernen zu kopieren.

Mit der Causal Mask ist das Modell gezwungen, sich seine
Vorhersagen zu erarbeiten. An Position zwei sieht es *The* und
*cat* und muss *sat* erraten. An Position drei sieht es *The cat*
und *sat* und muss *on* erraten. Das Modell kann nicht
vorausschauen, um Hinweise zu finden. Jede Vorhersage wird
ausschließlich mit den Informationen getroffen, die an dieser
Stelle im Satz verfügbar sind. Genau so funktioniert
Textgenerierung in der realen Welt. Man weiß nur, was vorher kam.
Man weiß nie, was als Nächstes kommt.

## Wann wurde es erfunden

Die Causal Mask wurde zusammen mit dem Transformer selbst im
Paper Attention Is All You Need von 2017 eingeführt. Sie war kein
nachträglicher Einfall. Sie war eine Design-Anforderung. Die
Autoren wussten, dass Sprachmodelle autoregressiv trainiert werden
müssen, das heißt Wort für Wort von links nach rechts. Die Causal
Mask erzwingt diese Einschränkung während des Trainings, sodass
sich das Modell bei der Generierung korrekt verhält.

## Wie es Schritt für Schritt funktioniert

### Die Attention-Matrix ohne Maske

Für den dreiwortigen Satz *I love dogs* berechnet das Modell einen
Attention-Score zwischen jedem Wortpaar. Jedes Wort kann jedes
andere Wort sehen.

```
Attention-Scores (ohne Maske):

           I       love    dogs
I         0.42    0.15    0.08     ← I kann love und dogs sehen
love      0.33    0.51    0.22     ← love kann I und dogs sehen
dogs      0.19    0.28    0.44     ← dogs kann I und love sehen
```

Das ist eine vollständig verbundene Matrix. Jedes Wortpaar hat
einen Score. Das Modell kann Informationen aus jedem Teil des
Satzes nutzen.

### Die Attention-Matrix mit Causal Mask

Nach dem Anwenden der Maske werden zukünftige Positionen auf minus
unendlich gesetzt. Nach dem Softmax wird minus unendlich zu genau
null. Diese Verbindungen werden gekappt.

```
Attention-Scores (mit Maske):

           I       love    dogs
I         0.42    -inf    -inf      ← I kann nur sich selbst sehen
love      0.33    0.51    -inf      ← love kann I und sich selbst sehen
dogs      0.19    0.28    0.44      ← dogs kann alle drei sehen
```

Nach dem Softmax werden aus diesen Scores Gewichte:

```
Attention-Weights (nach Softmax):

           I       love    dogs
I         1.00    0.00    0.00      ← 100 % auf sich selbst
love      0.46    0.54    0.00      ← aufgeteilt zwischen I und sich selbst
dogs      0.27    0.30    0.43      ← aufgeteilt auf alle drei
```

Das obere rechte Dreieck besteht nur aus Nullen. Das ist das
charakteristische Muster einer Causal Mask. Es handelt sich um
eine untere Dreiecksmatrix.

### Die Regel in einer Zeile

```
Ein Token an Position p kann nur Tokens an den Positionen
0 bis p sehen. Ein Token an Position p kann kein Token an
Position p+1 oder später sehen.
```

## Wie wir es implementieren

Eine Causal Mask zu erstellen ist überraschend einfach. PyTorch
hat eine Funktion namens tril, die das untere Dreieck einer Matrix
zurückgibt.

```python
import torch

seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len))

print(mask)

# Ausgabe:
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

Das ist die gesamte Maske. Vier Zeilen aus Einsen und Nullen.
Einsen bedeuten sichtbar. Nullen bedeuten verborgen.

Im eigentlichen Attention-Code verwenden wir diese Maske so:

```python
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)

# Innerhalb des Attention-Forward:
if mask is not None:
    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
```

Die Operation `masked_fill` setzt minus unendlich überall dort
ein, wo die Maske eine Null hat. Nach dem Softmax ist `e^(-inf)`
gleich null, und diese Positionen tragen nichts zum finalen Output
bei.

## Was während der Generierung passiert

Während des Trainings verhindert die Causal Mask Schummeln.
Während der Textgenerierung ist die Causal Mask noch immer
vorhanden, verrichtet aber weniger Arbeit.

Bei der Textgenerierung beginnen wir mit einem Prompt wie *Once
upon a*. Das Modell verarbeitet diese drei Tokens mithilfe der
Causal Mask. Token zwei kann Token null und Token eins sehen.
Token eins kann nur Token null sehen. Normal.

Dann sagt das Modell das nächste Token vorher. Nehmen wir an, es
sagt *time* vorher. Wir hängen *time* an die Sequenz an, sodass
daraus *Once upon a time* wird. Jetzt führen wir das Modell erneut
auf diesen vier Tokens aus. Die Causal Mask gilt weiterhin. Das
neue Token *time* kann *Once*, *upon* und *a* sehen, aber es kann
das nächste Token nicht sehen, weil das nächste Token noch nicht
existiert.

Die Causal Mask ist fest in die Architektur eingebaut. Sie ist
kein Feature, das nur beim Training vorkommt. Sie ist eine
grundlegende Einschränkung, die autoregressive Sprachmodelle
überhaupt erst möglich macht. Ohne sie könnten wir niemals Text
Token für Token generieren, weil das Modell ständig versuchen
würde, einen Blick auf Wörter zu erhaschen, die noch nicht
geschrieben wurden.

## Was ohne die Causal Mask passiert

Würden wir die Causal Mask während des Trainings entfernen, würde
das Modell lernen zu schummeln. Sein Trainings-Loss wäre extrem
niedrig, weil es die Antwort jederzeit sehen kann. Aber während
der Generierung, wenn zukünftige Tokens noch nicht existieren,
wäre das Modell völlig verloren. Es wüsste nicht, wie es das
Unbekannte vorhersagen soll. Sein Output wäre Kauderwelsch.

Das ist ein häufiger Fehlerfall bei Einsteigern, die ihren ersten
Transformer bauen. Der Trainings-Loss sieht fantastisch aus. Der
generierte Text ist Unsinn. Die Causal Mask wurde weggelassen.

## Was du dir merken solltest

Die Causal Mask macht Attention zu einer Einbahnstraße. Wörter
können zurückblicken, aber niemals vorausschauen. Das zwingt das
Modell dazu, jedes Token nur mithilfe der Tokens vorherzusagen,
die davor kamen. Genau so funktioniert Textgenerierung. Man
schreibt ein Wort nach dem anderen. Man weiß nie, was als Nächstes
kommt, bevor man es geschrieben hat.

Die Maske ist eine untere Dreiecksmatrix aus Einsen und Nullen.
Nullen werden in den Attention-Scores zu minus unendlich. Minus
unendlich wird nach dem Softmax zu null. Die Verbindungen zu
zukünftigen Wörtern werden gekappt. Nur die Vergangenheit bleibt
übrig. Diese einfache Regel ist es, was ein Sprachmodell von einem
Textkopierer unterscheidet.
