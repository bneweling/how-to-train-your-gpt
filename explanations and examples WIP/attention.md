# Attention : Wie Wörter miteinander sprechen

## Was ist das

Attention ist die Art und Weise, wie ein Sprachmodell entscheidet,
welche Wörter wichtig sind, wenn es einen Satz liest.

Stell dir vor, du bist auf einer lauten Party. Zehn Leute reden
gleichzeitig. Du versuchst, eine Person zu verstehen. Du hörst
nicht allen gleich gut zu. Du schenkst der Person, mit der du
sprichst, mehr Aufmerksamkeit. Der Person auf der anderen Seite des
Raums schenkst du weniger Aufmerksamkeit. Dein Gehirn konzentriert
sich ganz natürlich auf das, was wichtig ist.

Attention macht dasselbe mit Wörtern. Wenn das Modell einen Satz
liest, betrachtet es alle Wörter auf einmal. Dann entscheidet es,
wie sehr es sich um jedes Wort kümmern soll. Das Wort, um das es
sich am meisten kümmert, bekommt das meiste Gewicht. Das Wort, um
das es sich am wenigsten kümmert, bekommt fast kein Gewicht. Dann
kombiniert es alle Wörter zusammen mit diesen Gewichten. Das
Ergebnis ist ein neues Verständnis des aktuellen Wortes, das
Informationen von jedem anderen wichtigen Wort enthält.

## Wo wird es eingesetzt

Attention ist das Herzstück jedes Transformer-Modells. Es sitzt in
jedem Transformer-Block. Das Modell führt Attention einmal pro
Block aus. Wenn das Modell zwölf Blöcke hat, führt es Attention
zwölfmal aus. Jedes Mal werden die Wörter besser darin, einander zu
verstehen.

```
Satz → Embeddings → [Attention → FFN] × 12 → Ausgabe
```

## Warum wir es brauchen

Bevor es Attention gab, lasen Modelle Wörter eines nach dem anderen
von links nach rechts. Bis sie das Ende eines Satzes erreichten,
hatten sie vergessen, was am Anfang stand. Wie ein Mensch, der den
Anfang einer Geschichte vergisst, bis man mit dem Erzählen fertig
ist. Attention behebt das, indem es dem Modell erlaubt, jedes Wort
gleichzeitig zu betrachten. Kein Vergessen. Kein verblassendes
Gedächtnis.

Hier ist ein konkretes Beispiel:

```
"The cat sat on the mat because it was warm"
```

Was bedeutet *it* in diesem Fall. Ein Mensch weiß, dass es die
*mat* meint. Nicht die *cat*. Woher wissen wir das. Weil Wärme eine
Eigenschaft von Objekten wie Matten ist. Nicht von Tieren wie
Katzen. Wir verbinden das Wort *warm* mit dem Wort *mat* und
ignorieren das Wort *cat*. Wir tun das, ohne nachzudenken. Es
passiert automatisch.

Ein Modell ohne Attention liest von links nach rechts. Wenn es das
Wort *it* erreicht, liegt das Wort *mat* schon sechs Wörter zurück.
Das Signal ist verblasst. Das Modell kann sich nicht erinnern. Es
könnte denken, dass sich *it* auf *cat* bezieht, was mit dem Wort
*warm* keinen Sinn ergibt.

Mit Attention kann das Wort *it* auf jedes vorangegangene Wort
zurückblicken. Es sieht *mat* und bemerkt, dass *mat* mit *warm*
verbunden ist. Es sieht *cat* und bemerkt, dass *cat* weniger mit
*warm* verbunden ist. Es legt mehr Gewicht auf *mat*. Das Modell
löst die Bedeutung korrekt auf.

## Wann wurde es erfunden

Attention wurde 2017 in einem Paper namens Attention Is All You
Need eingeführt. Der Titel war kühn. Die Autoren behaupteten, man
brauche keinen anderen Mechanismus. Attention allein reiche aus.
Sie hatten recht. Jedes bedeutende Sprachmodell seit 2018 baut
vollständig auf Attention auf.

## Wie es funktioniert : eine Schritt-für-Schritt-Geschichte

Lass uns ein echtes Beispiel mit drei Wörtern und echten Zahlen
durchgehen. Du kannst mitverfolgen und jede Berechnung
nachvollziehen.

### Die Ausgangslage

Wir haben drei Wörter in unserem Satz. Das Modell hat jedem Wort
einen Vektor aus vier Zahlen zugewiesen. Diese Zahlen repräsentieren
die Bedeutung.

```
Wort 0 ("I"):    [ 0.5,  0.2, -0.3,  0.8]
Wort 1 ("love"): [ 0.1, -0.5,  0.7, -0.2]
Wort 2 ("dogs"): [ 0.9,  0.3, -0.1, -0.5]
```

Das Modell will das Wort *dogs* verstehen. Aber es sollte auch an
*I* und *love* denken, weil sie Kontext liefern. *I love dogs* ist
etwas anderes als *I fear dogs*. Die Wörter rund um *dogs* sind
wichtig.

### Schritt 1 : Erzeuge drei Dinge für jedes Wort

Das Modell nimmt jedes Wort und multipliziert es mit drei gelernten
Matrizen. Das Ergebnis sind drei neue Vektoren namens Query, Key
und Value. Jedes Wort bekommt sein eigenes Q K V.

```
Query = "Wonach suche ich?"
Key   = "Was habe ich zu bieten?"
Value = "Das ist mein tatsächlicher Inhalt"
```

Stell es dir wie eine Dating-App vor. Die Query ist dein Profil,
das sagt, was du willst. Der Key ist das, was alle anderen über
sich sagen, was sie bieten. Wenn deine Query zum Key von jemandem
passt, schenkst du dieser Person Aufmerksamkeit.

Nach der Matrixmultiplikation werden unsere drei Wörter zu:

```
Wort    | Query              | Key                | Value
"I"     | [ 0.8,  0.1]      | [ 0.6, -0.3]      | [ 0.4,  0.9]
"love"  | [-0.2,  0.7]      | [ 0.1,  0.5]      | [-0.3,  0.2]
"dogs"  | [ 0.5, -0.4]      | [-0.4,  0.8]      | [ 0.7, -0.1]
```

Jeder Vektor hat hier nur zwei Zahlen, um es einfach zu halten.
Echte Modelle verwenden 64 Zahlen pro Head.

### Schritt 2 : Bewerte jedes Wortpaar

Jetzt vergleichen wir jede Query mit jedem Key. Wir bilden das
Skalarprodukt (Dot Product). Das Skalarprodukt misst, wie gut sie
zusammenpassen. Ein hoher Score bedeutet, dass die Query wirklich
das will, was dieser Key bietet.

Lass uns berechnen, wie sehr *dogs* jedem Wort Aufmerksamkeit
schenken will.

```
dogs betrachtet I:
  Q_dogs · K_I = (0.5 × 0.6) + (-0.4 × -0.3)
               = 0.30 + 0.12
               = 0.42

dogs betrachtet love:
  Q_dogs · K_love = (0.5 × 0.1) + (-0.4 × 0.5)
                  = 0.05 + (-0.20)
                  = -0.15

dogs betrachtet dogs (sich selbst):
  Q_dogs · K_dogs = (0.5 × -0.4) + (-0.4 × 0.8)
                  = -0.20 + (-0.32)
                  = -0.52
```

Die Scores sagen uns: *dogs* passt am besten zu *I* (der Score ist
0.42, also positiv). *dogs* passt einigermaßen zu *love* (der Score
ist -0.15, also nahe null). *dogs* passt nicht gut zu sich selbst
(der Score ist -0.52, also deutlich negativ).

### Schritt 3 : Skaliere die Scores

Wir teilen jeden Score durch die Quadratwurzel der Dimensionsgröße.
Unsere Vektoren haben zwei Dimensionen. Die Quadratwurzel von zwei
ist etwa 1.4. Diese Skalierung verhindert, dass die Zahlen zu groß
werden.

```
Skalierte Scores: [0.42/1.4, -0.15/1.4, -0.52/1.4]
             = [0.30, -0.11, -0.37]
```

Warum skalieren wir. Ohne Skalierung können die Scores sehr groß
werden, wenn wir größere Vektoren wie 64 Dimensionen verwenden.
Große Scores führen im nächsten Schritt zu extremen Ergebnissen,
bei denen ein Wort die gesamte Attention bekommt und alles andere
null bekommt. Skalierung sorgt für Ausgewogenheit.

### Schritt 4 : Wende die Causal Mask an

Während des Trainings kann jedes Wort nur Wörter sehen, die vor ihm
kamen. Wörter, die danach kommen, sind verborgen. Das ist wie beim
Lesen eines Buchs. Du weißt nicht, was auf der nächsten Seite
steht, bevor du sie umblätterst.

In unserem Beispiel kann *I* nur sich selbst sehen. *love* kann *I*
und sich selbst sehen. *dogs* kann alle drei sehen. Wörter, die
verborgen bleiben sollen, bekommen einen Score von minus unendlich.
Nach dem nächsten Schritt wird minus unendlich zu null. Diese
Wörter werden vollständig ignoriert.

```
I kann sehen:    [I, versteckt, versteckt]
love kann sehen: [I, love, versteckt]
dogs kann sehen: [I, love, dogs]
```

### Schritt 5 : Verwandle Scores in Prozentwerte (Softmax)

Softmax nimmt unsere Scores und verwandelt sie in Prozentwerte, die
sich zu 100 Prozent aufsummieren. Das ergibt unsere
Attention-Gewichte.

```
Für dogs beim Betrachten aller Wörter:
Scores: [0.30, -0.11, -0.37]

Schritt 1 : Erhebe e in die Potenz jedes Scores:
         e^0.30 = 1.35
         e^-0.11 = 0.90
         e^-0.37 = 0.69

Schritt 2 : Addiere sie:
         1.35 + 0.90 + 0.69 = 2.94

Schritt 3 : Teile jeden durch die Summe:
         1.35 / 2.94 = 0.46  (46 Prozent Attention auf I)
         0.90 / 2.94 = 0.31  (31 Prozent auf love)
         0.69 / 2.94 = 0.23  (23 Prozent auf sich selbst)

Finale Attention-Gewichte für dogs: [0.46, 0.31, 0.23]
```

Beim Lesen des Wortes *dogs* schenkt das Modell *I* 46 Prozent
Aufmerksamkeit, *love* 31 Prozent und *dogs* selbst 23 Prozent.

### Schritt 6 : Mische die Values mithilfe der Attention-Gewichte

Jetzt haben wir Gewichte. Wir verwenden sie, um die Value-Vektoren
zu mischen. Wörter, die uns wichtiger sind, bekommen ihre Values
mit einem größeren Gewicht multipliziert.

```
Neue Repräsentation von dogs = (0.46 × Value von I)
                            + (0.31 × Value von love)
                            + (0.23 × Value von dogs)

= 0.46 × [0.4, 0.9] + 0.31 × [-0.3, 0.2] + 0.23 × [0.7, -0.1]

= [0.184, 0.414] + [-0.093, 0.062] + [0.161, -0.023]

= [0.252, 0.453]
```

Der neue Vektor für *dogs* lautet [0.252, 0.453]. Das ist nicht
mehr nur die Bedeutung von *dogs*. Er enthält jetzt Informationen
über *I* und *love*, gewichtet danach, wie wichtig sie sind. Das
Wort *dogs* weiß jetzt über die Wörter um es herum Bescheid. Es hat
Kontext.

## Die vollständige Attention-Matrix

Hier ist, was alle drei Wörter nach der Attention sehen. Jede Zeile
ist ein Wort. Jede Spalte ist das, worauf dieses Wort Attention
richtet.

```
           I       love    dogs
I         1.00    0.00    0.00     ← I kann nur sich selbst sehen
love      0.45    0.55    0.00     ← love sieht I und sich selbst
dogs      0.46    0.31    0.23     ← dogs sieht alle drei
```

Die obere rechte Ecke besteht komplett aus Nullen. Das ist die
Causal Mask am Werk. Kein Wort kann die Zukunft sehen. So lernt das
Modell, das nächste Wort vorherzusagen, ohne zu schummeln.

## Multi-Head Attention

Warum bei einer einzigen Attention-Berechnung aufhören.
Verschiedene Aspekte von Sprache brauchen verschiedene Arten von
Attention. Ein Head könnte sich auf Grammatik konzentrieren. Ein
anderer Head könnte sich auf Bedeutung konzentrieren. Ein weiterer
könnte sich darauf konzentrieren, ob Wörter sich auf dasselbe
beziehen.

In unserem echten Modell führen wir Attention zwölfmal parallel
aus. Jedes Mal mit unterschiedlichen gelernten Matrizen für Q, K
und V. Jeder Head spezialisiert sich auf etwas anderes. Dann
kombinieren wir alle zwölf Heads wieder miteinander.

```
Eingabe → Head 1: Fokus auf Grammatik     ↘
        Head 2: Fokus auf Bedeutung     →  Kombinieren → Ausgabe
        Head 3: Pronomenauflösung ↗
        ...
        Head 12: Fokus auf Position
```

Jeder Head arbeitet in einem kleineren Raum. Wenn das Modell 768
Dimensionen und 12 Heads hat, arbeitet jeder Head mit 64
Dimensionen. Das ist, als hätte man zwölf Experten, die alle
denselben Text durch eine andere Linse betrachten. Ihre
Erkenntnisse werden kombiniert.

## Ein winziges Codebeispiel zum Ausprobieren

```python
import torch
import torch.nn.functional as F
import math

# Drei Wörter mit jeweils vier Dimensionen
words = torch.tensor([[
    [0.5,  0.2, -0.3,  0.8],   # I
    [0.1, -0.5,  0.7, -0.2],   # love
    [0.9,  0.3, -0.1, -0.5],   # dogs
]])

# Erstelle zufällige Q K V Matrizen (normalerweise werden diese gelernt)
d_model = 4
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

# Projiziere auf Q K V
Q = words @ W_q
K = words @ W_k
V = words @ W_v

print("Query vectors:")
print(Q)
print()

# Berechne Attention-Scores
head_dim = 4
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

# Erstelle Causal Mask
seq_len = 3
mask = torch.tril(torch.ones(seq_len, seq_len))
mask = mask.view(1, 1, seq_len, seq_len)
scores = scores.masked_fill(mask == 0, float('-inf'))

# Softmax, um Gewichte zu erhalten
weights = F.softmax(scores, dim=-1)

# Wende Gewichte auf Values an
output = weights @ V

print("Attention weights:")
print(weights)
print()

print("Output (context aware representations):")
print(output)
print()

print("The first row shows Token 0 attending only to itself.")
print("The last row shows Token 2 attending to all three tokens.")
```

## Das große Ganze

Attention ist, als würde man jedem Wort eine Taschenlampe geben.
Das Wort lässt sein Licht auf andere Wörter scheinen. Helleres
Licht bedeutet mehr Attention. Das Modell lernt während des
Trainings, wohin es das Licht richten soll. Nach ausreichend
Training weiß es, dass Pronomen das Nomen beleuchten sollten, auf
das sie sich beziehen. Es weiß, dass Verben ihre Subjekte
beleuchten sollten. Es weiß, dass Adjektive die Nomen beleuchten
sollten, die sie beschreiben.

Deshalb ist Attention die geheime Zutat moderner KI. Sie lässt
Wörter einander verstehen, statt isoliert zu existieren. Ein Satz
wird zu einem Netz aus Verbindungen. Nicht nur eine Liste von
Wörtern in einer Reihenfolge.

## Was du dir merken musst

Jedes Wort erzeugt eine Query, einen Key und einen Value. Die Query
eines Wortes vergleicht sich mit den Keys aller anderen Wörter. Die
Übereinstimmungs-Scores werden zu Attention-Gewichten. Die Gewichte
werden verwendet, um die Values zu mischen. Das Ergebnis ist ein
neues Verständnis jedes Wortes, das über jedes andere wichtige Wort
Bescheid weiß.

Während des Trainings können Wörter die Zukunft nicht sehen. Die
Causal Mask blockiert sie. Das zwingt das Modell, vorherzusagen,
was als Nächstes kommt, und dabei nur zu verwenden, was vorher kam.
Genauso, wie du das Ende eines Satzes vorhersagst, wenn jemand
mitten im Gedanken innehält.

Attention läuft parallel mit mehreren Heads ab. Jeder Head lernt
ein anderes Muster. Ein Head könnte Pronomen mit Nomen verbinden.
Ein anderer könnte Verben mit Subjekten verbinden. Die Heads
sprechen während der Berechnung nie miteinander. Sie werden erst am
Ende kombiniert.
