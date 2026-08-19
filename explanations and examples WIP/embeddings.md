# Embeddings: Zahlen eine Bedeutung geben

## Was es ist

Ein Embedding ist eine Liste von Zahlen, die die Bedeutung eines
Wortes erfasst.

Nach der Tokenisierung ist jedes Wort nur noch eine Zahl. *Cat* ist
9246. *Dog* ist 4821. Diese Zahlen sind nur Labels. Die Zahl 9246
bedeutet für sich genommen nichts. Das Modell kann aus einer Zahl
wie 9246 nichts lernen, denn 9246 ist der Zahl 4821 nicht näher als
der Zahl 279. Es sind einfach willkürliche IDs.

Ein Embedding verwandelt diese Zahl in einen Vektor. Ein Vektor ist
einfach eine Liste von Dezimalzahlen. Bei GPT-2 wird jedes Wort zu
einer Liste von 768 Zahlen. Diese Zahlen sind nicht willkürlich.
Wörter mit ähnlicher Bedeutung erhalten ähnliche Listen. Wörter mit
unterschiedlicher Bedeutung erhalten unterschiedliche Listen.

```
Token ID 9246 ("cat") → [0.023, -0.451, 0.789, ..., -0.102]  (768 Zahlen)
Token ID 4821 ("dog")  → [0.019, -0.443, 0.795, ..., -0.098]  (sehr ähnlich!)
Token ID 279  ("the")  → [0.891, 0.112, -0.334, ..., 0.567]   (sehr unterschiedlich!)
```

Die Kernidee ist, dass *cat* und *dog* in diesem Zahlenraum nah
beieinander liegen, weil beide Tiere sind. *The* liegt weit
entfernt, weil es ein Funktionswort mit einer völlig anderen Rolle
ist.

## Wo es eingesetzt wird

Die Embedding-Layer sitzt direkt nach dem Tokenizer und direkt vor
den Attention-Layern. Sie ist die Brücke zwischen der
Integer-Ausgabe des Tokenizers und der Fließkommazahl-Eingabe des
Transformers.

```
Rohtext: "The cat"
    ↓
Tokenizer: [1169, 3797]
    ↓
Embedding-Layer: zwei Vektoren mit je 768 Zahlen
    ↓
Transformer-Blöcke
```

Jedes moderne Sprachmodell besitzt eine Embedding-Layer. Sie ist die
allererste gelernte Komponente in der gesamten Pipeline.

## Warum wir es brauchen

Ohne Embeddings würde das Modell versuchen, mit Token-IDs zu
rechnen. Stellen wir uns vor, wir addieren zwei Wörter. Der Token
für *king* ist 9246. Der Token für *queen* ist 9247. Würden wir sie
addieren, erhielten wir 18493. Diese Zahl bedeutet nichts. Sie
entspricht keinem sinnvollen Wort. Das Modell kann aus Token-IDs
nicht lernen.

Mit Embeddings arbeitet das Modell mit kontinuierlichen Vektoren.
Der Vektor für *king* ist etwa [0.3, -0.5, 0.8, ...]. Der Vektor für
*queen* ist [0.2, -0.6, 0.7, ...]. Diese sind nah beieinander, aber
nicht identisch. Das Modell kann *king* minus *man* plus *woman*
berechnen und erhält etwas, das dem Vektor für *queen* sehr nahe
kommt. Das nennt man die Arithmetik-Eigenschaft von Embeddings.

```
embedding(king)  ≈ [0.30, -0.50, 0.80]
embedding(man)   ≈ [0.25, -0.45, -0.30]
embedding(woman) ≈ [0.22, -0.55, 0.70]
embedding(queen) ≈ [0.27, -0.60, 0.75]

king - man + woman = [0.27, -0.60, 0.75] ≈ queen!
```

Das wurde nicht von einem Menschen programmiert. Das Modell hat
selbst herausgefunden, dass das Ändern des Geschlechts eines Wortes
einer geraden Linie durch den Embedding-Raum entspricht. Es hat
das ausschließlich durch das Lesen von Millionen Sätzen gelernt, in
denen *king* und *queen* in ähnlichen Kontexten, aber mit
unterschiedlichen Pronomen auftauchten.

## Wann es erfunden wurde

Die Idee von Word-Embeddings ist alt. Eine Technik namens Word2Vec
wurde 2013 von Google veröffentlicht. Sie war die erste, die zeigte,
dass Wortvektoren Bedeutungsbeziehungen erfassen können. Die
Embedding-Layer in Transformern ist ein direkter Nachfahre von
Word2Vec. Der Unterschied besteht darin, dass Word2Vec-Embeddings
vorab berechnet und dann eingefroren wurden. Transformer-Embeddings
werden während des Trainings von Grund auf gelernt. Sie passen sich
an die konkrete Aufgabe an, die das Modell lernt.

## Wie es funktioniert: eine riesige Nachschlagetabelle

Man kann sich die Embedding-Layer als Tabelle mit 50257 Zeilen
vorstellen. Jede Zeile hat 768 Spalten. Zeile null ist der Vektor
für Token null. Zeile eins ist der Vektor für Token eins. Zeile 3797
ist der Vektor für das Wort *cat*. Der Forward Pass der
Embedding-Layer besteht einfach darin, Zeilen in dieser Tabelle
nachzuschlagen.

```python
# Gegeben Token-IDs: [1169, 3797]
# Zeile 1169 nachschlagen → Vektor für "The"  (768 Zahlen)
# Zeile 3797 nachschlagen → Vektor für "cat"  (768 Zahlen)
# Beide Vektoren zurückgeben
```

Das ist der gesamte Forward Pass der Embedding-Layer. Keine
Multiplikation. Keine Aktivierungsfunktion. Nur ein
Tabellen-Nachschlag.

### Wie die Tabelle aufgebaut wird

Die Tabelle beginnt vollständig zufällig. Jede Zeile wird mit Zahlen
gefüllt, die aus einer Normalverteilung mit Mittelwert 0 und
Standardabweichung 0.02 gezogen werden. Zu diesem Zeitpunkt sind
*cat* und *dog* einander genauso nah wie *cat* und *democracy*.
Alles ist zufälliges Rauschen.

Dann beginnt das Training. Das Modell liest einen Satz wie *The cat
sat on the mat*. Es sagt voraus, dass als Nächstes *mat* kommen
sollte. Liegt es falsch, ist der Loss hoch. Backpropagation schickt
ein winziges Signal zurück durch das gesamte Modell, einschließlich
der Embedding-Tabelle. Dieses Signal besagt:

„Der Vektor für *cat* sollte ein Stück in die Richtung verschoben
werden, die beim nächsten Mal hilft, *mat* vorherzusagen.“

Nach Millionen von Trainingsschritten verwandelt sich die Tabelle.
Wörter, die in ähnlichen Kontexten auftauchen, werden in ähnliche
Positionen gedrängt. Der Vektor für *cat* rückt nah an *dog* und
*pet* und *feline*. Der Vektor für *car* rückt nah an *vehicle* und
*drive* und *road*. Der Raum organisiert sich selbst in Nachbarschaften
von Bedeutung.

### Wie die Nachbarschaften aussehen

Nach dem Training besitzt der 768-dimensionale Raum eine natürliche
Struktur. Manche Richtungen in diesem Raum entsprechen realen
Konzepten.

```
Richtung 1 (Dimensionen 0 bis 63):     Lebendig vs. nicht lebendig
Richtung 2 (Dimensionen 64 bis 127):   Groß vs. klein
Richtung 3 (Dimensionen 128 bis 191):  Positiv vs. negativ
Richtung 4 (Dimensionen 192 bis 255):  Formell vs. leger
... und so weiter durch alle 768 Dimensionen
```

Diese Richtungen wurden nie programmiert. Sie sind auf natürliche
Weise entstanden, weil das Modell es nützlich fand, Wörter auf diese
Art zu organisieren. Wenn das Modell wissen muss, ob etwas lebendig
ist oder nicht, schaut es sich einen bestimmten Satz von Dimensionen
im Embedding-Vektor an.

## Ein winziges Codebeispiel

```python
import torch
import torch.nn as nn

# Eine kleine Embedding-Tabelle erstellen
vocab_size = 1000   # 1000 eindeutige Token
d_model = 4         # 4-dimensionale Vektoren (klein gehalten für das Beispiel)

embedding = nn.Embedding(vocab_size, d_model)

# Ein paar Token-IDs nachschlagen
token_ids = torch.tensor([[12, 45, 678]])
vectors = embedding(token_ids)

print("Token IDs:", token_ids)
print("Shape of output:", vectors.shape)
print()
print("Vector for token 12:", vectors[0, 0].tolist())
print("Vector for token 45:", vectors[0, 1].tolist())
print("Vector for token 678:", vectors[0, 2].tolist())
print()
print("Each token ID became a", d_model, "dimensional vector.")
print("Right now the values are random. After training they will")
print("capture meaning. Words with similar meanings will have")
print("similar vectors.")
```

Führt man diesen Code aus, sieht man etwa Folgendes:

```
Token IDs: tensor([[ 12,  45, 678]])
Shape of output: torch.Size([1, 3, 4])

Vector for token 12:  [0.031, -0.124, -0.847, 0.562]
Vector for token 45:  [-1.231, 0.789, 0.023, -0.441]
Vector for token 678: [0.892, -0.334, 0.671, -0.128]
```

Diese Vektoren sind im Moment zufällig. Sie haben keine Bedeutung.
Nach dem Training auf Milliarden von Sätzen werden Token 12 und
Token 45 nah beieinander liegen, wenn sie in ähnlichen Kontexten
auftauchen, oder weit auseinander, wenn nicht.

## Die Größe der Embedding-Tabelle

Die Embedding-Tabelle ist oft die größte Komponente des Modells,
gemessen an der Zahl der Parameter.

```
GPT-2 Small:  50257 Wörter × 768 Dimensionen  = 38.6 Millionen Zahlen
GPT-3:        50257 Wörter × 12288 Dimensionen = 617 Millionen Zahlen
```

Deshalb ist Weight Tying wichtig. Die Ausgabe-Layer benötigt
ebenfalls eine Matrix derselben Größe, um von den Hidden States
zurück auf Vorhersagen über das Vokabular zu projizieren. Statt zwei
riesige Matrizen zu speichern, teilen wir uns eine. Die
Embedding-Tabelle wird sowohl für die Eingabe als auch für die
Ausgabe verwendet.

## Embeddings für Satzzeichen und Sonderzeichen

Jeder Token erhält ein Embedding. Auch Satzzeichen und
Sonderzeichen. Der Punkt bekommt ein Embedding. Das Komma bekommt
ein Embedding. Der End-of-Text-Marker bekommt ein Embedding.

Diese Embeddings sind genauso wichtig wie Wort-Embeddings. Das
Modell lernt, dass auf das Embedding für einen Punkt das Embedding
für ein großgeschriebenes Wort folgt. Es lernt, dass auf das
Embedding für ein Fragezeichen das Embedding für eine Antwort folgt.
Die Struktur der Sprache steckt in diesen kleinen Token-Embeddings
genauso wie in den Wort-Embeddings.

## Was du dir merken solltest

Ein Embedding ist eine Liste von Zahlen, die die Bedeutung eines
Wortes repräsentiert. Wörter mit ähnlicher Bedeutung haben ähnliche
Listen. Wörter mit unterschiedlicher Bedeutung haben unterschiedliche
Listen.

Die Embedding-Tabelle beginnt zufällig. Das Training verschiebt
Wörter je nach den Kontexten, in denen sie auftauchen. Nach
genügend Training organisiert sich der Raum selbst. King minus man
plus woman ergibt queen. Das wurde nicht programmiert. Das Modell
hat es selbst herausgefunden.

Die Embedding-Layer ist nur eine Nachschlagetabelle. Keine Mathematik
im Inneren. Man gibt ihr eine Token-ID, und sie gibt einen Vektor
zurück. Dieser Vektor sind die Koordinaten des Wortes im
Bedeutungsraum. Alles, was das Modell über ein Wort weiß, steckt in
diesen 768 Zahlen.
