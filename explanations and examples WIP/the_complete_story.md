# Wie ein GPT wirklich funktioniert: Die vollständige Geschichte

Dies ist die Geschichte eines Sprachmodells. Nicht nur eines Teils. Nicht nur eines
Schritts. Das Ganze. Von einer Textdatei auf einer Festplatte bis zu einer Maschine,
die Gedichte schreiben und Fragen beantworten und Code generieren kann. Jedes
einzelne Teil. Jede einzelne Entscheidung. Jede einzelne Zahl.

Wir werden ein GPT von Grund auf bauen. Wir werden es trainieren. Wir werden zusehen, wie es
lernt. Wir werden es laufen lassen. Am Ende werden Sie jede Zeile
Code in jeder Datei verstehen. Diese Geschichte setzt voraus, dass Sie Python kennen. Sonst nichts.

---

## Teil 1: Was wir bauen

Ein GPT ist ein Vorhersager des nächsten Wortes. Das ist alles. Das ist die ganze Sache.
Man gibt ihm ein paar Wörter. Es rät, welches Wort als Nächstes kommt. Dann nimmt es
diese Vermutung und rät das nächste Wort. Dann das nächste. Irgendwann hat es
einen Absatz oder ein Gedicht oder ein juristisches Dokument oder ein Rezept für
Schokoladenkuchen geschrieben. Aber darunter rät es immer nur ein Wort auf
einmal.

Das Modell hat etwa 150 Millionen Stellschrauben. Jede Stellschraube ist eine Zahl. Training
bedeutet, die richtigen Zahlen für alle 150 Millionen Stellschrauben zu finden, sodass die
Vermutungen des Modells dem entsprechen, was ein Mensch schreiben würde. Sobald diese Zahlen
gefunden sind, kann das Modell Text schreiben, der manchmal nicht von
menschlichem Schreiben zu unterscheiden ist.

Wie finden wir diese Zahlen? Wir zeigen dem Modell Sätze aus dem
Internet. Milliarden von Sätzen. Für jeden Satz verbergen wir das letzte Wort
und bitten das Modell, es zu erraten. Wenn es falsch rät, ermitteln wir, an welchen
Stellschrauben wir drehen müssen und in welche Richtung, damit die Vermutung beim nächsten Mal besser wird.
Das wiederholen wir Milliarden Mal. Die Stellschrauben konvergieren langsam zu Werten,
die die Muster menschlicher Sprache erfassen.

Die Architektur des Modells bestimmt, welche Muster es erfassen kann.
Ein größeres Modell kann mehr Muster erfassen. Eine bessere Architektur kann
mit derselben Anzahl an Stellschrauben mehr Muster erfassen. Unsere Architektur
ist dieselbe, die auch LLaMA 3 und Mistral und Qwen verwenden. Sie stellt das
beste öffentlich dokumentierte Design für Sprachmodelle Stand 2025 dar.

---

## Teil 2: Die Daten

Bevor wir ein Modell trainieren können, brauchen wir Text. Viel Text. Milliarden
Wörter. Für dieses Projekt verwenden wir Wikipedia, weil es frei
verfügbar und gut geschrieben ist und fast jedes Thema abdeckt, über das Menschen
je nachgedacht haben.

Wikipedia kann als einzelne XML-Datei heruntergeladen oder über die
HuggingFace-datasets-Bibliothek abgerufen werden. Die datasets-Bibliothek übernimmt das Herunterladen
und Zwischenspeichern, sodass wir die Rohdateien nicht selbst verwalten müssen.

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
texts = [item["text"] for item in dataset if item["text"].strip()]
```

Der WikiText-103-Datensatz enthält etwa 29000 Wikipedia-Artikel. Sie
wurden leicht bearbeitet, um Markup und Metadaten zu entfernen. Was übrig bleibt,
ist sauberer, flüssiger englischer Text. Genau das, was wir brauchen.

Aber wir können einem neuronalen Netz keinen Rohtext füttern. Neuronale Netze fressen
Zahlen. Wir müssen unseren Text zuerst in Zahlen umwandeln.

---

## Teil 3: Tokenisierung – Text wird zu Zahlen

Die Umwandlung von Text in Zahlen nennt man Tokenisierung. Der
Algorithmus, den wir verwenden, ist Byte Pair Encoding. Kurz BPE. Er wurde
1994 zur Datenkompression erfunden und 2016 für Sprachmodelle wiederverwendet.

Die Idee ist konzeptionell einfach. Beginne mit jedem Zeichen als eigenem
Token. Finde das häufigste Paar benachbarter Token in den Trainingsdaten.
Verschmelze sie zu einem neuen Token. Wiederhole das, bis du 50000 Token hast.

Schauen wir uns an, wie das an einem winzigen Beispiel funktioniert. Stellen wir uns vor, unsere Trainingsdaten
enthalten nur vier Wörter, wobei Leerzeichen als Unterstriche markiert sind.

```
l o w _
l o w e r _
l o w e s t _
l o w e s t _
```

Jeder Buchstabe und Unterstrich ist ein eigenes Token. Wir haben neun Token
insgesamt. Das Alphabet ist klein. Das Modell würde viele Token benötigen, um
selbst einen kurzen Satz darzustellen. Also verschmelzen wir.

Das häufigste Paar ist l und o. Sie treten im Wort low viermal
gemeinsam auf. Wir erzeugen ein neues Token lo. Jetzt ist unser Text kürzer.

```
lo w _
lo w e r _
lo w e s t _
lo w e s t _
```

Wir haben zehn Token. Wir verschmelzen weiter. Das nächsthäufigste Paar ist lo und
w. Sie treten viermal gemeinsam auf. Wir erzeugen low. Jetzt ist unser Text noch
kürzer.

```
low _
low e r _
low e s t _
low e s t _
```

Wir machen weiter. Nach vielen Verschmelzungsrunden enthält unser Vokabular nützliche
Bausteine wie low und er und est und den Leerzeichen-Marker. Jetzt kann das Wort lowest,
das nicht in unseren ursprünglichen Trainingsdaten vorkommt, trotzdem dargestellt werden als
low plus est. Zwei Token statt sechs Zeichen. Kompression und
Generalisierung in einem Schritt.

Echte BPE-Tokenizer wie der von GPT-2 verwenden 50000 Verschmelzungen. Sie
starten mit allen 256 möglichen Byte-Werten als Basisalphabet. Das bedeutet, sie können
jeden Text in jeder Sprache tokenisieren, der als Bytes dargestellt werden kann, was
auf jeden Text zutrifft. Die 50000 Verschmelzungen erfassen die häufigsten Muster über
Milliarden von Wörtern hinweg. Das Ergebnis ist ein Vokabular, das häufige
Wörter als einzelne Token und seltene Wörter als Sequenzen weniger Token und
völlig unbekannte Wörter als Sequenzen einzelner Byte-Token darstellen kann.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "The cat sat on the mat."
tokens = tokenizer.encode(text)
print(tokens)  # [464, 3797, 3332, 319, 262, 2603, 13]
```

Sieben Token. Jedes Token ist eine Ganzzahl zwischen 0 und 50256. Diese
Ganzzahlen sind das Einzige, was das Modell je sieht. Der Rohtext ist verschwunden.
Das Modell lebt in einer Welt aus Ganzzahlen.

Der Tokenizer hat ein besonderes Token, das Aufmerksamkeit verdient. Token
50256 ist der Ende-des-Texts-Marker. Er wird zwischen jedem Dokument
in den Trainingsdaten platziert. Ohne ihn würde das Modell denken, dass der letzte
Satz eines Wikipedia-Artikels nahtlos in den ersten
Satz des nächsten übergeht. Das Ende-des-Texts-Token ist das Signal für das Modell, dass
ein Gedanke geendet hat und ein neuer, unabhängiger Gedanke begonnen hat.

Jedes Token hat eine eindeutige ID. Token 464 ist immer The mit großem T.
Token 3797 ist immer cat. Token 13 ist immer ein Punkt. Diese Zuordnungen
sind fest. Sie ändern sich während des Trainings nie. Der Tokenizer ist nicht
Teil des neuronalen Netzes. Er ist ein Vorverarbeitungsschritt mit einem eigenen,
separaten Algorithmus.

Aber diese Integer-IDs sind nur Bezeichnungen. Die Zahl 3797 hat keine
mathematische Beziehung zur Zahl 2603. Das Modell kann aus diesen rohen
Ganzzahlen keine sinnvollen Muster lernen. Wir müssen jedem
Token eine reichhaltigere Darstellung geben. Wir brauchen Embeddings.

---

## Teil 4: Embeddings – Zahlen werden zu Bedeutung

Ein Embedding ist ein Vektor aus Fließkommazahlen, der die Bedeutung eines Tokens
erfasst. In unserem Modell erhält jedes Token einen Vektor mit 768
Zahlen. Token 3797 bekommt 768 Zahlen. Token 2603 bekommt 768 Zahlen.
Jedes der 50257 Token bekommt seine eigene Zeile in einer riesigen Nachschlagetabelle
mit der Form 50257 mal 768.

```python
embedding_table = torch.nn.Embedding(50257, 768)
```

Diese Tabelle ist einfach eine Matrix. Zeile 3797 ist das Embedding für cat. Zeile 2603
ist das Embedding für mat. Wenn das Modell den Vektor für Token 3797 braucht,
liest es einfach Zeile 3797 aus der Tabelle. Keine Multiplikation. Keine Aktivierungsfunktion.
Nur ein Speicherzugriff.

Bei der Initialisierung wird jede Zeile mit Zufallszahlen gefüllt, die aus einer
Normalverteilung mit Mittelwert 0 und Standardabweichung 0,02 gezogen werden. Das bedeutet,
die meisten Werte liegen zwischen -0,04 und 0,04. In diesem Moment haben cat und dog
keine besondere Beziehung zueinander. Sie sind nur zwei zufällige Zeilen in einer zufälligen
Tabelle. Jedes Token ist gleichermaßen zufällig.

Das Training ändert das. Über Milliarden von Trainingsschritten hinweg werden die Zeilen
aktualisiert. Token, die in ähnlichen Kontexten auftreten, werden zu
ähnlichen Werten hin verschoben. Token, die in unterschiedlichen Kontexten auftreten, werden
auseinandergetrieben. Nach dem Training liegt das Embedding für cat sehr nah am
Embedding für dog. Beide liegen weit entfernt vom Embedding für democracy.
Der Raum organisiert sich selbst in Nachbarschaften der Bedeutung.

```
cat    = [ 0.34, -0.12,  0.78, -0.56, ...]  (768 Zahlen)
dog    = [ 0.31, -0.15,  0.81, -0.52, ...]  (sehr ähnlich zu cat)
democracy = [ 0.89, 0.67, -0.23,  0.91, ...]  (völlig anders)
```

Die Embedding-Tabelle ist die größte einzelne Komponente im Modell. Sie hat
50257 Zeilen mal 768 Spalten, also etwa 38,6 Millionen Zahlen. Das ist
ungefähr ein Viertel aller Parameter in unserem Modell. Diese 38,6
Millionen Zahlen kodieren alles, was das Modell darüber weiß, was Wörter bedeuten.

---

## Teil 5: Positional Encoding – Das Modell lernt Reihenfolge

Der Transformer liest alle Token gleichzeitig. Es gibt keine Verarbeitung von links nach rechts.
Keine schrittweise Rekurrenz. Jedes Token wird gleichzeitig
verarbeitet. Das ist eine Stärke, weil es schnell und parallelisierbar ist.
Aber es ist auch ein Problem, weil das Modell keine Möglichkeit hat zu wissen, welches
Token zuerst und welches zuletzt kam.

Betrachten wir zwei Sätze. The dog bit the man. The man bit the dog. Dieselben
Wörter. Andere Reihenfolge. Völlig andere Bedeutung. Würde das Modell
jedes Token unabhängig behandeln, könnte es diese
Sätze nicht unterscheiden. Die Bedeutung wäre durcheinandergewürfelt.

Wir müssen jedem Token seine Position aufprägen. Dem Modell mitteilen, wo im
Satz dieses Token sitzt. Der moderne Weg, das zu tun, heißt Rotary
Position Embeddings. Kurz RoPE. Es wurde 2021 eingeführt und
2023 von LLaMA übernommen. Jedes größere Modell seitdem verwendet es.

Statt eine Positionszahl zum Embedding zu ADDIEREN, ROTIERT RoPE die
Query- und Key-Vektoren um einen Winkel, der von der Position abhängt. Die
Rotation erhält die Länge des Vektors, verändert also nicht die
Bedeutung des Wortes. Aber die Rotation verändert die Richtung, sodass das
Attention-Skalarprodukt zwischen zwei Wörtern zu einer Funktion ihres
Abstands wird.

```
Wort an Position 1: rotiert um Winkel θ₁
Wort an Position 4: rotiert um Winkel θ₄

Attention-Score zwischen ihnen:
Q₁ · K₄ = original_dot × cos(θ₁ - θ₄) + cross_term × sin(θ₁ - θ₄)

Das Ergebnis hängt von (θ₁ - θ₄) ab, was eine Funktion von (4 - 1) = 3
Schritten Abstand ist. Nicht von den Positionen 1 und 4 selbst. Nur vom Abstand.
```

Das ist die zentrale Einsicht. RoPE macht Attention abhängig von der relativen
Position. Wörter, die drei Schritte voneinander entfernt sind, bekommen immer dieselbe rotatorische
Beziehung, unabhängig davon, ob sie an den Positionen 0 und 3 oder
497 und 500 auftreten. Der Transformer interessiert sich dafür, wie weit zwei
Wörter voneinander entfernt sind, nicht wo sie in absoluten Zahlen stehen.

Die Winkel werden vorab berechnet und gespeichert. Jedes Dimensionspaar rotiert
mit unterschiedlicher Geschwindigkeit. Das erste Dimensionspaar rotiert am schnellsten und
erfasst die lokale Wortreihenfolge. Das letzte Paar rotiert am langsamsten und
erfasst weitreichende Position. Dieser Multi-Skalen-Ansatz bedeutet, dass das Modell sowohl
feingranulare lokale Positionsinformation als auch grobkörnige globale
Positionsinformation besitzt.

```python
# Rotationswinkel für jede Position vorberechnen
dim_indices = torch.arange(0, d_model, 2).float()
inv_freq = 1.0 / (10000.0 ** (dim_indices / d_model))
positions = torch.arange(max_seq_len).float()
freqs = torch.outer(positions, inv_freq)
emb = freqs.repeat_interleave(2, dim=-1)  # [max_seq_len, d_model]
cos_cached = emb.cos()
sin_cached = emb.sin()
```

Während des Forward-Pass schlagen wir die vorberechneten Cosinus- und Sinus-
Werte für jede Position nach und wenden die Rotation an. Die Rotationsformel
für jedes Dimensionspaar (x₀, x₁) an Position p lautet:

```
x₀' = x₀ × cos(θ_p) - x₁ × sin(θ_p)
x₁' = x₀ × sin(θ_p) + x₁ × cos(θ_p)
```

Das ist eine standardmäßige 2D-Rotation. Angewendet auf jedes Paar von Dimensionen in
den Query- und Key-Vektoren. Die Values werden nicht rotiert, weil Positions-
information nur gebraucht wird, um zu entscheiden, WELCHE Token beachtet werden sollen, nicht
für den Inhalt der Token selbst.

---

## Teil 6: Attention – Der zentrale Mechanismus

Attention ist das Herzstück des Transformers. Alles andere ist unterstützende
Infrastruktur. Die Embedding-Schicht speist Attention. Das Feed-Forward-
Netz verfeinert die Ausgabe der Attention. Die Normalisierungsschichten halten
Attention stabil. Aber in Attention versteht das Modell tatsächlich
Beziehungen zwischen Wörtern.

### Die Intuition

Stellen Sie sich vor, Sie lesen einen langen Satz. Manche Wörter sind wichtiger
als andere, um zu verstehen, was passiert. Wenn der Satz lautet The
cat that had been sitting on the mat for three hours finally stretched
and yawned, müssen Sie stretched und yawned mit cat über
zwölf dazwischenliegende Wörter hinweg verbinden. Ihr Gehirn macht das automatisch. Attention
macht das mathematisch.

Für jedes Wort im Satz erzeugt das Modell drei Vektoren. Einen Query-
Vektor, der fragt: Wonach suche ich? Einen Key-Vektor, der sagt: Was habe
ich anzubieten? Einen Value-Vektor, der meinen tatsächlichen Inhalt enthält. Jedes Wort
vergleicht seine Query mit dem Key jedes anderen Wortes. Wörter mit hohen Übereinstimmungs-
werten bekommen mehr Aufmerksamkeit. Ihre Values werden im
Output stärker gewichtet.

### Die Berechnung Schritt für Schritt

Verfolgen wir Attention für einen konkreten Satz. Unser Satz ist
The cat sat on the mat. Sieben Token. Wir betrachten einen Attention-Head
mit einer Head-Dimension von 64.

Der Input für Attention ist eine Matrix der Form 7 mal 768. Sieben Token, jedes
dargestellt durch 768 Zahlen. Wir projizieren diese Matrix in drei neue
Matrizen der Form 7 mal 64. Eine für Query. Eine für Key. Eine für Value.

```
Q = input @ W_q  (7 × 768 @ 768 × 64 = 7 × 64)
K = input @ W_k  (7 × 768 @ 768 × 64 = 7 × 64)
V = input @ W_v  (7 × 768 @ 768 × 64 = 7 × 64)
```

Die Gewichtsmatrizen W_q, W_k und W_v werden während des Trainings gelernt. Sie
sind es, was jeden Attention-Head unterschiedlich macht. Verschiedene Heads lernen
unterschiedliche Projektionen, die unterschiedliche sprachliche Muster erfassen.

Als Nächstes wenden wir RoPE auf die Query- und Key-Vektoren an. Das prägt jeder
Query und jedem Key seine Positionsinformation auf.

```
Q = RoPE(Q, seq_len=7)
K = RoPE(K, seq_len=7)
```

Jetzt berechnen wir die Attention-Scores. Der Score zwischen Token i und
Token j ist das Skalarprodukt von Query i mit Key j.

```
scores = Q @ K^T / sqrt(64)
```

Das Ergebnis ist eine 7-mal-7-Matrix. Jede Zeile ist ein Token in der Rolle der Query. Jede
Spalte ist ein Token in der Rolle des Keys. Der Wert in Zeile i, Spalte j ist, wie stark
Token i Token j beachten möchte.

```
scores-Matrix (vor der Maskierung):

         The    cat    sat    on     the    mat    .
The      0.42   0.15   0.08  -0.03  -0.11   0.02  -0.18
cat      0.38   0.52   0.22   0.01  -0.05   0.09  -0.21
sat      0.21   0.78   0.15   0.31   0.22   0.28   0.05
on      -0.05   0.11   0.45   0.55   0.38   0.42   0.12
the     -0.09   0.03   0.21   0.48   0.61   0.35   0.08
mat     -0.12  -0.01   0.15   0.41   0.52   0.58   0.11
.       -0.22  -0.15  -0.08   0.12   0.18   0.22   0.48
```

Schauen wir uns die Zeile für sat an (Zeilenindex 2). Sie hat einen hohen Score für cat
(0,78) und moderate Scores für on (0,31) und mat (0,28). Sat möchte
sein Subjekt und seine Präpositionalphrase beachten. Es kümmert sich
weniger um sich selbst (0,15) und den Punkt (0,05). Dieses Muster entstand
aus den gelernten Gewichten W_q und W_k und aus der positionalen Rotation.

Jetzt wenden wir die kausale Maske an. Token können die Zukunft nicht sehen. Das
obere rechte Dreieck der Matrix wird auf minus unendlich gesetzt.

```
scores-Matrix (nach der Maskierung):

         The    cat    sat    on     the    mat    .
The      0.42  -inf   -inf   -inf   -inf   -inf   -inf
cat      0.38   0.52  -inf   -inf   -inf   -inf   -inf
sat      0.21   0.78   0.15  -inf   -inf   -inf   -inf
on      -0.05   0.11   0.45   0.55  -inf   -inf   -inf
the     -0.09   0.03   0.21   0.48   0.61  -inf   -inf
mat     -0.12  -0.01   0.15   0.41   0.52   0.58  -inf
.       -0.22  -0.15  -0.08   0.12   0.18   0.22   0.48
```

Nach Softmax wird minus unendlich zu null. Die Scores werden zu
Attention-Gewichten, die sich für jede Zeile zu eins summieren.

```
Attention-Gewichte-Matrix (nach Softmax):

         The    cat    sat    on     the    mat    .
The      1.00   0.00   0.00   0.00   0.00   0.00   0.00
cat      0.47   0.53   0.00   0.00   0.00   0.00   0.00
sat      0.18   0.35   0.10   0.00   0.00   0.00   0.00
on       0.08   0.10   0.22   0.25   0.20   0.15   0.00
the      0.05   0.06   0.10   0.19   0.28   0.17   0.00
mat      0.04   0.05   0.08   0.17   0.24   0.30   0.00
.        0.03   0.04   0.05   0.08   0.11   0.14   0.28
```

Schauen wir uns die Zeile für on an (Zeilenindex 3). Sie beachtet sich zu 25 Prozent
selbst, zu 22 Prozent sat, zu 20 Prozent the und zu 15 Prozent mat. Eine
ausgewogene Verteilung über die vorangehenden Token. Das Wort on ist eine
Präposition, die alles um sich herum verbindet. Es braucht Kontext von
jedem nahegelegenen Wort.

Schauen wir uns die Zeile für The an (Zeilenindex 0). Sie beachtet sich zu 100 Prozent selbst.
Davor gibt es nichts. Das Wort The hat keinen Kontext. Es muss sich vollständig
auf seine eigene Bedeutung verlassen. Das gilt immer für das erste Token in
jeder Sequenz.

Schließlich nutzen wir diese Gewichte, um die Value-Vektoren zu mischen.

```
output = attention_weights @ V

Für Token sat (Zeile 2):
new_sat = 0.18 × V_The + 0.35 × V_cat + 0.10 × V_sat
```

Der neue Vektor für sat enthält jetzt Informationen von The und cat,
gewichtet danach, wie sehr sat sich um sie kümmert. Die ursprüngliche Bedeutung von sat
ist über die Self-Attention (10 Prozent) noch vorhanden, aber sie wurde
mit Kontext aus dem Subjekt des Satzes angereichert.

Diese gesamte Berechnung findet 12-mal parallel für 12 Heads statt. Jeder
Head hat seine eigenen W_q-, W_k- und W_v-Matrizen. Jeder Head lernt andere
Attention-Muster. Nachdem alle Heads ihre Outputs berechnet haben,
verketten wir sie wieder zu einem einzigen 768-dimensionalen Vektor und
projizieren durch eine finale lineare Schicht.

```
all_heads = torch.cat([head_0, head_1, ..., head_11], dim=-1)  # 12 × 64 = 768
output = all_heads @ W_o  # 768 @ 768 = 768
```

Die Output-Projektion W_o mischt Informationen zwischen den Heads. Jeder Head
arbeitete unabhängig. Jetzt teilen sie ihre Erkenntnisse. Der Grammatik-
Head teilt dem Pronomen-Auflösungs-Head mit, was er gefunden hat. Der Positions-Head
teilt dem semantischen Head Wortabstände mit. Der gemischte Output ist reicher
als der Beitrag jedes einzelnen Heads.

---

## Teil 7: RMSNorm – Zahlen unter Kontrolle halten

Vor Attention und vor dem Feed-Forward-Netz normalisieren wir den
Input. Normalisierung hält die Zahlen auf einer konsistenten Skala, während sie
durch Dutzende Schichten fließen.

Wir verwenden RMSNorm. Es ist einfacher und schneller als das ältere LayerNorm. Es
berechnet den quadratischen Mittelwert (Root Mean Square) eines Vektors und teilt jedes Element durch
ihn. Das Ergebnis hat immer einen RMS-Wert von 1,0.

```
rms = sqrt(mean(x²))
output = x / rms × weight
```

Das Gewicht ist ein gelernter Parameter. Ein Gewicht pro Dimension. Es beginnt
bei 1,0 und lernt während des Trainings. Es erlaubt dem Modell, wichtige
Dimensionen zu verstärken und unwichtige zu unterdrücken, während die Gesamt-
größenordnung stabil bleibt.

Ohne Normalisierung würden die Outputs der Attention- und Feed-Forward-Schichten
unbegrenzt wachsen. Nach zwölf Schichten könnten manche Werte
tausendmal größer sein als andere. Der Softmax in der nächsten Attention-
Schicht würde zu einem One-Hot-Vektor werden. Gradienten würden verschwinden. Training
würde scheitern.

Mit Normalisierung erhält jede Schicht saubere, gut skalierte Inputs. Der Turm
aus zwölf Blöcken bleibt gerade. Das Modell trainiert reibungslos.

---

## Teil 8: SwiGLU – Das gegatete Feed-Forward-Netz

Nach Attention hat jedes Token Informationen von allen anderen Token vermischt.
Aber das Mischen war linear. Attention ist nur eine gewichtete Summe. Gewichtete
Summen reichen nicht aus, um die Komplexität von Sprache zu erfassen. Wir brauchen nicht-
lineare Verarbeitung.

Das Feed-Forward-Netz liefert diese Nichtlinearität. Es verarbeitet jedes
Token unabhängig mit denselben gelernten Gewichten. Jedes Token erhält
dieselbe Transformation, angewendet auf seinen eigenen Vektor.

Unser Feed-Forward-Netz verwendet SwiGLU. SwiGLU ist eine gegatete Aktivierung. Es
teilt die Berechnung in zwei Pfade auf. Ein Pfad erzeugt Werte. Der
andere Pfad erzeugt Gates. Die Gates steuern, wie viel von jedem Wert
durchgelassen wird.

```
h = input @ W₁  (768 → 3072)  # Werte-Pfad
g = input @ W₂  (768 → 3072)  # Gate-Pfad
output = (SiLU(h) × g) @ W₃  (3072 → 768)  # kombinieren und projizieren
```

Die Erweiterung von 768 auf 3072 gibt dem Netz Raum, Informationen zu
transformieren. In der breiteren mittleren Schicht kann das Netz komplexere
Muster darstellen. Die Kontraktion zurück auf 768 zwingt es, diese
Muster in eine dichte Darstellung zu komprimieren.

Die SiLU-Aktivierung auf dem Werte-Pfad liefert glatte Nichtlinearität.
Anders als ReLU, das bei null eine scharfe Kante hat, ist SiLU überall glatt.
Das lässt Gradienten während des Trainings besser fließen. Der Gate-Pfad hat keine
Aktivierung. Er kann jede beliebige reelle Zahl ausgeben. Ein Gate von null blockiert die
Information. Ein Gate von eins lässt sie unverändert durch. Ein Gate von zwei
verstärkt sie. Das Modell lernt, welche Inputs verstärkt und
welche unterdrückt werden sollen.

Das Gate lernt kontextabhängige Filterung. Wenn das Token ein Verb ist,
verstärkt das Gate vielleicht Dimensionen, die mit Handlung zusammenhängen, und unterdrückt
Dimensionen, die mit Objekten zusammenhängen. Wenn das Token ein Nomen ist, macht es vielleicht
das Gegenteil. Dieselben Netzgewichte gelten für jedes Token, aber das Verhalten
unterscheidet sich, weil der Vektor jedes Tokens zu unterschiedlichen Gate-Werten führt.

SwiGLU hat drei Gewichtsmatrizen statt der zwei, die ein Standard-
Feed-Forward-Netz hätte. Die zusätzliche Matrix ist für das Gate. Das
fügt unserem Modell etwa 28 Millionen Parameter hinzu, verglichen mit einem Standard-
FFN. Jeder dieser Parameter trägt zu besserer Leistung bei.
Der Gating-Mechanismus ist der Grund, warum SwiGLU ReLU und GELU im großen Maßstab übertrifft.

---

## Teil 9: Die Residual Connection – Die Gradienten-Autobahn

Jede Teilschicht hat eine Residual Connection. Der Attention-Output wird zum
Attention-Input addiert. Der Feed-Forward-Output wird zum Feed-
Forward-Input addiert.

```
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

Diese Pluszeichen sind die wichtigsten Operatoren im gesamten Modell.
Ohne sie können tiefe Transformer nicht trainiert werden. Die Gradienten würden
verschwinden. Die frühen Schichten würden nie lernen.

Hier ist der Grund. Wenn das Modell eine Vorhersage trifft und den Loss berechnet, sendet es
einen Gradienten rückwärts durch das Netz. Dieser Gradient sagt jedem
Gewicht, wie es sich ändern soll, um den Loss zu verringern. Der Gradient fließt rückwärts
durch jede Schicht in umgekehrter Reihenfolge. Bei jeder Schicht wird er mit
der Ableitung der Funktion dieser Schicht multipliziert. Ist die Ableitung kleiner
als eins, schrumpft der Gradient. Nach der Rückpropagierung durch zwölf
Schichten ist der Gradient in der ersten Schicht das Produkt aus elf Zahlen,
die jeweils kleiner als eins sind.

```
gradient_at_layer_1 = gradient_at_layer_12 × d₁ × d₂ × ... × d₁₁

Wenn jede Ableitung 0,5 ist:
gradient_at_layer_1 = gradient_at_layer_12 × 0.5¹¹
                    = gradient_at_layer_12 × 0.0005
```

Der Gradient in Schicht eins ist zweitausendmal kleiner als der
Gradient in Schicht zwölf. Die erste Schicht erhält fast kein Lernsignal.
Ihre Gewichte bleiben zufällig. Das Modell kann nicht trainieren.

Residual Connections beheben das, indem sie einen zweiten Pfad bereitstellen. Der Gradient
kann wie zuvor rückwärts durch die Teilschicht fließen. Oder er kann die
Teilschicht komplett umgehen und direkt zum Input fließen. Der Umgehungspfad hat eine
Ableitung von exakt 1,0. Immer. Der Gradient schrumpft nicht.

```
Mit Residual: output = input + sublayer(norm(input))
Ableitung:    d(output)/d(input) = 1 + d(sublayer)/d(input)
```

Die Gesamtableitung ist 1 plus etwas. Selbst wenn dieses Etwas klein ist,
sorgt die 1 dafür, dass der Gradient nie verschwindet. Nach zwölf Schichten ist der
Gradient in Schicht eins mindestens so groß wie der Gradient in Schicht
zwölf. Jede Schicht kann lernen.

Das ist der Grund, warum wir zwölf Blöcke stapeln können. Oder vierundzwanzig. Oder sechsundneunzig.
Die Gradienten-Autobahn bleibt unabhängig von der Tiefe offen. Die einzige Grenze ist der
Rechenaufwand, nicht die Trainierbarkeit.

---

## Teil 10: Das vollständige Modell – Alles zusammensetzen

Setzen wir jedes Teil zum vollständigen Modell zusammen.

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # Teil 4
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)  # Teile 6-9
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)  # Teil 7
        self.lm_head = nn.Linear(d_model, vocab_size)  # Output-Projektion

    def forward(self, input_ids):
        # Teil 4: Token einbetten
        x = self.token_embedding(input_ids)  # [batch, seq, 768]

        # Teile 6-9: Durch die Transformer-Blöcke verarbeiten
        for layer in self.layers:
            x = layer(x)  # Jeder Block enthält Attention + FFN + Residuals

        # Teil 7: Finale Normalisierung
        x = self.final_norm(x)

        # Output: Auf das Vokabular projizieren
        logits = self.lm_head(x)  # [batch, seq, 50257]
        return logits
```

Das ist das gesamte Modell. Etwa fünfzig Zeilen Code. Jede Komponente, die wir
besprochen haben, steckt in diesen fünfzig Zeilen. Die Embedding-Tabelle aus Teil 4.
Die gestapelten Transformer-Blöcke aus den Teilen 6 bis 9. Die finale
Normalisierung aus Teil 7. Die Output-Projektion, die Hidden States
zurück in Vokabular-Vorhersagen umwandelt.

Das Modell nimmt einen Batch von Token-Sequenzen als Input. Für jede Position
in jeder Sequenz erzeugt es 50257 Scores. Einen Score für jedes mögliche
nächste Token. Das Token mit dem höchsten Score ist die Vorhersage des Modells dafür, welches
Wort als Nächstes kommt.

---

## Teil 11: Der Output – Von Vektoren zu Wörtern

Die finale Schicht des Modells projiziert von 768 Dimensionen auf 50257
Dimensionen. Das ist eine einfache lineare Transformation. Multiplikation mit einer Gewichts-
matrix der Form 768 mal 50257.

```python
logits = x @ W_lm_head  # [batch, seq, 768] @ [768, 50257] = [batch, seq, 50257]
```

Diese 50257 Zahlen heißen Logits. Sie sind unnormalisierte Scores.
Höher bedeutet, das Modell hält dieses Token für wahrscheinlicher. Es sind noch keine
Wahrscheinlichkeiten, weil sie sich nicht zu eins summieren und manche
negativ sein können.

Um Logits in Wahrscheinlichkeiten umzuwandeln, wenden wir Softmax an.

```python
probs = softmax(logits)  # Jede Zeile summiert sich jetzt zu 1.0
```

Jede Zeile der Wahrscheinlichkeitsmatrix summiert sich zu eins. Zeile i, Spalte j ist die
vom Modell geschätzte Wahrscheinlichkeit, dass Token j als Nächstes kommt, gegeben die ersten
i plus 1 Token des Inputs.

Das Modell gibt nicht nur ein Token aus. Es gibt eine Wahrscheinlichkeits-
verteilung über alle 50257 Token aus. Während des Trainings vergleichen wir diese
Verteilung mit dem tatsächlichen nächsten Token. Während der Generierung ziehen wir eine Stichprobe aus
dieser Verteilung, um das nächste Wort auszuwählen.

---

## Teil 12: Der Loss – Falschheit messen

Training braucht eine Zahl, die uns sagt, wie gut die Vorhersagen des Modells
sind. Niedriger ist besser. Diese Zahl nennt man den Loss.

Wir verwenden Cross-Entropy-Loss. Er misst den Unterschied zwischen den
vom Modell vorhergesagten Wahrscheinlichkeiten und den tatsächlichen nächsten Token.

Für eine einzelne Vorhersage, bei der das wahre nächste Token j ist:

```
loss = -log(probs[j])
```

Weist das Modell dem richtigen Token eine Wahrscheinlichkeit von 0,9 zu, ist der Loss
der negative Logarithmus von 0,9, also 0,105. Gut. Das Modell war sich sicher und
hatte recht.

Weist das Modell dem richtigen Token eine Wahrscheinlichkeit von 0,1 zu, ist der Loss
der negative Logarithmus von 0,1, also 2,303. Schlecht. Das Modell war sich sicher über
die falschen Dinge.

Weist das Modell dem richtigen Token eine Wahrscheinlichkeit von 0,01 zu, ist der Loss
der negative Logarithmus von 0,01, also 4,605. Furchtbar. Das Modell hat die
richtige Antwort kaum in Betracht gezogen.

Der Loss ist immer positiv. Er nähert sich null, wenn das Modell perfekt wird.
Er nähert sich unendlich, wenn das Modell völlig falsch liegt.
Ein zufälliges Modell, das allen 50257 Token gleiche Wahrscheinlichkeit zuweist, hätte
einen Loss von negativem Logarithmus von eins durch 50257, also etwa 10,8. Das
ist die Baseline. Jeder Loss über 10,8 bedeutet, das Modell ist schlechter als
Zufall. Jeder Loss unter 10,8 bedeutet, das Modell hat etwas gelernt.

```python
def compute_loss(logits, targets):
    # logits:   [batch, seq, 50257]
    # targets:  [batch, seq]  (um 1 gegenüber dem Input verschoben)
    logits_flat = logits.view(-1, 50257)
    targets_flat = targets.view(-1)
    return F.cross_entropy(logits_flat, targets_flat)
```

Wir berechnen den Loss über alle Positionen in allen Sequenzen im Batch.
Der durchschnittliche Loss über Millionen von Vorhersagen liefert uns eine einzige Zahl,
die die Leistung des Modells misst. Bei jedem Trainingsschritt versuchen wir,
diese Zahl kleiner zu machen.

---

## Teil 13: Backpropagation – Herausfinden, was zu ändern ist

Wir haben einen Loss. Der Loss sagt uns, wie falsch das Modell lag. Aber er sagt
uns nicht, welches der 150 Millionen Gewichte wir ändern sollen oder in welche
Richtung. Backpropagation beantwortet diese Frage.

Backpropagation wendet die Kettenregel aus der Analysis an. Für jedes Gewicht
im Modell berechnet sie die partielle Ableitung des Loss nach
diesem Gewicht. Diese Ableitung sagt uns: Wenn ich dieses Gewicht um eine winzige
Menge erhöhe, wie stark ändert sich der Loss?

```python
loss.backward()  # PyTorch übernimmt die gesamte Rechnerei automatisch
```

Nach diesem Aufruf hat jedes Gewicht im Modell ein .grad-Attribut. Der
Gradient ist ein Tensor derselben Form wie das Gewicht. Jedes Element im
Gradienten ist die Richtung und Größe, in die dieses Gewicht geändert werden sollte, um
den Loss zu verringern.

```
Wenn weight[i,j].grad = 0.003:
  Eine Erhöhung von weight[i,j] lässt den Loss steigen.
  Wir sollten es verringern.

Wenn weight[i,j].grad = -0.005:
  Eine Erhöhung von weight[i,j] lässt den Loss sinken.
  Wir sollten es erhöhen.

Wenn weight[i,j].grad = 0.000:
  Eine Änderung dieses Gewichts wirkt sich nicht auf den Loss aus.
  Wir können es unverändert lassen oder ohne Konsequenz ändern.
```

Die Gradienten fließen rückwärts vom Loss durch die Output-Projektion,
durch die finale Normalisierung, durch jeden Transformer-Block in
umgekehrter Reihenfolge, durch die Embedding-Tabelle und zurück zum Input. Bei
jedem Schritt multipliziert die Kettenregel lokale Ableitungen. Die Residual-
Connections sorgen dafür, dass Gradienten diese Reise überstehen.

---

## Teil 14: Gradient Clipping – Wilde Sprünge verhindern

Manchmal erzeugt ein Batch Text sehr große Gradienten. Ein seltenes Wort-
muster oder eine ungewöhnliche Satzstruktur schickt eine Schockwelle durch die
Gradienten. Würden wir diese großen Gradienten direkt anwenden, würden die Gewichte
des Modells zu einer völlig anderen Konfiguration springen. Das Training
wäre zerstört.

Gradient Clipping verhindert das. Nach dem Backward-Pass prüfen wir die
Gesamtgröße aller Gradienten. Überschreitet sie einen Schwellenwert, verkleinern wir
alle Gradienten proportional, damit sie unter den Schwellenwert passen.

```python
total_norm = sqrt(sum(g.norm(2)² for g in gradients))
if total_norm > 1.0:
    scale = 1.0 / total_norm
    for g in gradients:
        g *= scale
```

Die Richtung des Updates bleibt erhalten. Nur die Schrittgröße wird begrenzt.
Das Modell macht kleine, sichere Schritte statt wilder Sprünge. Der Schwellenwert
von 1,0 ist Standard beim Training von Transformern. Er wurde empirisch
ermittelt. Er fängt gefährliche Spitzen ab, ohne normale Updates zu stören.

---

## Teil 15: AdamW – Die Gewichte aktualisieren

Wir haben Gradienten für jedes Gewicht. Jetzt müssen wir sie anwenden. Der
einfachste Ansatz ist, jedes Gewicht ein kleines Stück in die entgegengesetzte
Richtung seines Gradienten zu bewegen.

```
weight = weight - learning_rate × gradient
```

Das ist stochastischer Gradientenabstieg (Stochastic Gradient Descent). Er funktioniert, ist aber langsam und
instabil. Die Learning Rate ist für jedes Gewicht gleich, unabhängig davon,
wie stark jedes Gewicht sich ändern muss. Verrauschte Gradienten verursachen Zickzack-
Bewegungen. Große Gewichte erhalten keine Regularisierung.

AdamW verbessert alle drei Punkte. Es hält laufende Durchschnitte
vergangener Gradienten und ihrer Größen. Es nutzt diese Durchschnitte, um die
Schrittgröße für jedes Gewicht unabhängig anzupassen. Es wendet Weight Decay
getrennt vom Gradienten-Update an.

```python
# AdamW für ein einzelnes Gewicht
momentum = β₁ × momentum + (1 - β₁) × gradient      # Laufender Durchschnitt der Gradienten
velocity = β₂ × velocity + (1 - β₂) × gradient²      # Laufender Durchschnitt der quadrierten Gradienten

# Bias-Korrektur für frühe Schritte
mom_corrected = momentum / (1 - β₁^step)
vel_corrected = velocity / (1 - β₂^step)

# Entkoppelter Weight Decay
weight = weight × (1 - lr × weight_decay)

# Gradienten-Update
weight = weight - lr × mom_corrected / (sqrt(vel_corrected) + ε)
```

Der Momentum-Term wirkt wie Trägheit. Er glättet Rauschen, indem er einen
laufenden Durchschnitt vergangener Gradienten hält. Zeigt der Gradient
viele Schritte lang in dieselbe Richtung, baut sich Momentum auf und die
Schrittgröße nimmt zu. Oszilliert der Gradient, hebt sich Momentum
auf und die Schrittgröße nimmt ab.

Der Velocity-Term passt die Learning Rate pro Gewicht an. Gewichte, die
große Bewegungen gemacht haben, bekommen kleinere Schritte. Gewichte, die stillgestanden
haben, bekommen größere Schritte. Dieses adaptive Verhalten bedeutet, dass wir
die Learning Rate nicht für jedes Gewicht einzeln anpassen müssen.

Der Weight-Decay-Term schiebt alle Gewichte bei jedem Schritt um einen winzigen Bruchteil
in Richtung null. Das verhindert, dass Gewichte unbegrenzt wachsen. Große
Gewichte sind ein Zeichen von Overfitting. Das Modell ist zu selbstsicher
bezüglich weniger Muster geworden und ignoriert alles andere. Weight Decay zwingt es
dazu, bescheiden zu bleiben.

Der Epsilon-Term verhindert die Division durch null. Er ist winzig und muss nie
angepasst werden.

AdamW ist der Standard-Optimizer für das Training von Sprachmodellen. GPT-3
wurde damit trainiert. LLaMA wurde damit trainiert. Jedes Modell in diesem Leitfaden
trainiert damit. Die konkreten Hyperparameter β₁ von 0,9, β₂ von 0,95 und
Weight Decay von 0,1 sind die LLaMA-Standardwerte. Sie wurden
an Modellen von einer Milliarde bis siebzig Milliarden Parametern validiert.

---

## Teil 16: Cosine Warmup – Der Learning-Rate-Zeitplan

Die Learning Rate bleibt während des Trainings nicht konstant. Sie folgt einem
Zeitplan, der zunächst aufwärmt und dann abfällt (Decay).

Ganz zu Beginn des Trainings sind die Gewichte des Modells zufällig. Die
Gradienten sind groß und verrauscht. Eine hohe Learning Rate würde das
Modell in zufällige Richtungen davonfliegen lassen. Wir starten mit einer Learning Rate
von null und erhöhen sie über mehrere tausend Schritte hinweg linear bis zum
Maximum. Das ist die Warmup-Phase.

```python
if step < warmup_steps:
    lr = max_lr × step / warmup_steps
```

Sobald das Modell stabil ist, können wir mit voller Geschwindigkeit trainieren. Aber je weiter das Training
fortschreitet und das Modell einer guten Lösung näherkommt, müssen wir
vorsichtiger werden. Große Schritte würden über das Minimum hinausschießen. Wir
reduzieren die Learning Rate schrittweise entlang einer Kosinuskurve.

```python
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = min_lr + (max_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

Die Kosinuskurve fällt zunächst langsam, dann schneller in der Mitte, dann
wieder langsam am Ende. Dieser sanfte Abfall ist milder als Step-Decay,
bei dem die Learning Rate in festen Intervallen abrupt abfällt. Abrupte Sprünge
können das Modell stören. Cosine Decay ist kontinuierlich.

Ganz am Ende des Trainings erreicht die Learning Rate ein kleines Minimum.
Das Modell macht winzige Schritte, die seine Gewichte mit Präzision verfeinern. Die
Feinabstimmungsphase.

Alle drei Phasen zusammen machen das Training am Anfang stabil und
am Ende präzise. Jedes moderne Sprachmodell verwendet diesen Zeitplan.

---

## Teil 17: Mixed Precision – Schnelleres Training

Die Gewichte des Modells werden als 32-Bit-Fließkommazahlen gespeichert. Das
ist der Standard für wissenschaftliches Rechnen. Gute Präzision und guter Wertebereich.

Aber die meisten Operationen im Forward-Pass brauchen nicht 32 Bit
Präzision. Die Matrixmultiplikationen in Attention und im Feed-Forward-
Netz funktionieren fast genauso gut mit 16 Bit. 16 Bit statt 32 zu
verwenden, halbiert den Speicherverbrauch und verdoppelt fast die Geschwindigkeit auf modernen GPUs.

Wir verwenden ein Format namens bfloat16. Es hat denselben Wertebereich wie float32, aber
weniger Präzision. Die größte darstellbare Zahl ist in beiden
Formaten gleich. bfloat16 läuft also selbst bei den größten Matrix-
multiplikationen nie über. Der einzige Unterschied ist, dass bfloat16 nur
etwa zwei Dezimalstellen Präzision darstellen kann statt sieben.

Dieser Kompromiss ist perfekt für neuronale Netze. Wir brauchen den Wertebereich, um
Überlauf während Zwischenberechnungen zu verhindern. Aber wir brauchen nicht
sieben Stellen Präzision für jede Aktivierung. Zwei Stellen reichen aus,
damit das Modell effektiv lernt.

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    # Jede Operation hier verwendet bfloat16, wo es sicher ist
    logits = model(input_ids)
    loss = compute_loss(logits, targets)
```

Die Master-Gewichte werden immer in float32 gespeichert. Nur die Forward- und
Backward-Pässe verwenden bfloat16. Die Gewichts-Updates werden in float32 angewendet,
um die Präzision über tausende Trainingsschritte hinweg zu bewahren.

Manche Operationen bleiben in float32, weil sie mehr Präzision brauchen.
Normalisierungsschichten brauchen volle Präzision, um Aktivierungen richtig
skaliert zu halten. Der Softmax in Attention braucht volle Präzision für numerische
Stabilität. Autocast handhabt diese Ausnahmen automatisch. Wir müssen
nicht angeben, welche Operationen umgewandelt werden sollen.

---

## Teil 18: Die Trainingsschleife – Alles zusammensetzen

Wir haben jedes Teil. Das Modell. Die Daten. Den Tokenizer. Den Optimizer.
Den Scheduler. Die Loss-Funktion. Jetzt setzen wir sie zu einer Trainingsschleife
zusammen.

```python
for step in range(max_steps):
    # 1. Einen Batch Text holen
    batch = next(dataloader)
    input_ids, target_ids = batch

    # 2. Forward-Pass
    with autocast(use_amp):
        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)

    # 3. Backward-Pass
    loss.backward()

    # 4. Gradienten clippen
    clip_grad_norm(model.parameters(), max_norm=1.0)

    # 5. Gewichte aktualisieren
    optimizer.step()
    optimizer.zero_grad()

    # 6. Learning Rate aktualisieren
    scheduler.step()

    # 7. Fortschritt protokollieren
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

Sieben Schritte. Tausend- oder millionenfach wiederholt. Bei jeder Wiederholung
wird der Loss etwas kleiner. Das Modell wird etwas besser. Nach
genügend Wiederholungen kann das Modell zusammenhängenden Text generieren.

Die ersten paar hundert Schritte sind chaotisch. Der Loss springt hin und her. Die
Gradienten sind groß. Das Modell sucht. Um Schritt eintausend herum
beginnt der Loss stetig zu sinken. Das Modell hat eine gute Richtung gefunden.
Von da an ist der Fortschritt langsam, aber beständig. Jeder Schritt kratzt einen winzigen
Bruchteil vom Loss ab. Nach fünfzigtausend Schritten ist der Loss von
rund 10,8 auf irgendwo zwischen 2 und 3 gefallen. Das Modell kann
Sätze schreiben, die manchmal grammatikalisch korrekt und manchmal unsinnig sind. Es
weiß, dass Punkte Sätze beenden und dass Großbuchstaben sie beginnen.
Es weiß, dass auf the oft ein Nomen folgt. Es weiß, dass cat und dog
beide sitzen und laufen und schlafen können.

Nach fünfhunderttausend Schritten schreibt das Modell Absätze, die
größtenteils zusammenhängend sind. Es macht immer noch Fehler. Es erfindet Fakten. Es wiederholt
sich. Aber es hat einen bemerkenswerten Teil der Struktur des
Englischen erfasst. Alles durch das milliardenfache Vorhersagen des nächsten Worts.

---

## Teil 19: Textgenerierung – Das Modell spricht

Sobald das Modell trainiert ist, wollen wir es etwas schreiben lassen. Wir geben ihm eine
Startphrase, einen sogenannten Prompt. Das Modell liest den Prompt und sagt
das erste Wort danach voraus. Dann nimmt es den Prompt plus das vorhergesagte
Wort und sagt das zweite Wort voraus. Das wiederholt es, bis es genug
Text generiert hat oder ein Ende-des-Texts-Token vorhersagt.

```python
prompt = "The cat sat on the"
input_ids = tokenizer.encode(prompt)  # [464, 3797, 3332, 319, 262]

for _ in range(50):
    logits = model(input_ids)          # Nächstes Token vorhersagen
    logits = logits[:, -1, :]          # Nur die letzte Position

    probs = softmax(logits / temperature)
    next_token = sample(probs, top_k=50)

    input_ids = append(input_ids, next_token)
```

Die Sampling-Parameter steuern, wie das Modell das nächste Token auswählt.
Ohne jegliche Parameter würde das Modell immer das eine wahrscheinlichste
Token wählen. Der Output wäre deterministisch und oft repetitiv.
Derselbe Prompt würde immer dieselbe Vervollständigung erzeugen. Das Modell
würde sich in gängigen Phrasen wiederholen.

Temperature fügt Zufälligkeit hinzu. Sie teilt die Logits vor dem
Softmax durch eine Zahl. Niedrige Temperature macht die Verteilung schärfer. Das
oberste Token bekommt noch mehr Wahrscheinlichkeit. Der Output ist fokussiert und
vorhersagbar. Hohe Temperature flacht die Verteilung ab. Weniger wahrscheinliche
Token bekommen mehr Chancen. Der Output ist kreativ und unvorhersagbar.

```
Temperature 0.3: "The cat sat on the windowsill gazing at the birds outside."
Temperature 0.8: "The cat sat on the edge of the couch watching me with sleepy eyes."
Temperature 1.5: "The cat sat on the piano keys and composed a midnight melody."
```

Top-k begrenzt die Auswahl auf die k wahrscheinlichsten Token. Alles andere
bekommt die Wahrscheinlichkeit null. Das verhindert, dass das Modell jemals ein
völlig unsinniges Token wählt. Ein Wert von 50 ist üblich. Er eliminiert die
unteren 50207 Token und behält gleichzeitig genug Vielfalt für interessanten Output.

Top-p ist eine adaptive Version von Top-k. Statt immer k
Token zu behalten, behält es die kleinste Menge an Token, deren kumulative Wahrscheinlichkeit
p übersteigt. Ist sich das Modell sehr sicher, behält es vielleicht nur drei
Token. Ist das Modell unsicher, behält es vielleicht fünfhundert. Das
passt sich an die Konfidenz des Modells bei jedem Schritt an.

Zusammen geben uns diese drei Parameter feine Kontrolle über den Output des
Modells. Sie sind der Grund, warum dasselbe Modell sowohl technische
Dokumentation als auch Lyrik schreiben kann. Das Modell liefert die Wahrscheinlichkeiten. Die
Parameter steuern, wie wir aus ihnen samplen.

---

## Teil 20: KV-Cache – Generierung schnell machen

Die naive Generierungsschleife ist langsam. Jedes Mal, wenn wir ein neues Token anhängen,
berechnen wir die gesamte Sequenz von Grund auf neu. Token 500 wurde bereits
499 Mal verarbeitet, bevor wir Token 501 hinzufügen. Der Großteil der
Berechnung ist redundant. Die Key- und Value-Vektoren für die ersten 500
Token ändern sich nicht, wenn wir Token 501 hinzufügen.

Der KV-Cache eliminiert diese Redundanz. Wir speichern die Key- und Value-
Vektoren für jedes Token, das wir bereits verarbeitet haben. Wenn ein neues Token
ankommt, berechnen wir seinen Key und Value und hängen sie an den Cache an. Wir
berechnen nichts für die alten Token neu.

```
Ohne Cache:  Schritt 1 berechnet K und V für 1 Token.
                Schritt 2 berechnet K und V für 2 Token.
                Schritt 3 berechnet K und V für 3 Token.
                Gesamtaufwand: 1 + 2 + 3 + ... + N ≈ N²/2

Mit Cache:     Schritt 1 berechnet K und V für 1 Token.
                Schritt 2 berechnet K und V für 1 neues Token. Nutzt Altes wieder.
                Schritt 3 berechnet K und V für 1 neues Token. Nutzt Altes wieder.
                Gesamtaufwand: N
```

Für eine Generierung von tausend Token ist der KV-Cache etwa tausendmal
schneller. Die Speicherkosten sind für kleine Modelle beherrschbar. Für GPT-2 Small
sind es für den Cache von tausend Token etwa 35 Megabyte. Für GPT-3 Large
wären es etwa 4 Gigabyte. Bei sehr großen Modellen und sehr langen
Kontextlängen kann der Cache zum dominierenden Speicherverbraucher werden.

---

## Teil 21: Was das Modell tatsächlich gelernt hat

Nach dem Training an Milliarden von Wörtern hat das Modell Muster gelernt, die
für das ungeübte Auge unsichtbar sind. Es hat keine Fakten gelernt in der Art, wie
eine Datenbank Fakten speichert. Es hat statistische Regelmäßigkeiten gelernt. Auf die
Wortfolge the cat sat on the folgt fast immer mat oder
floor oder chair oder bed. Auf die Folge the capital of France folgt fast
immer Paris. Das Modell weiß nicht, was France oder Paris
oder capital bedeuten. Es kennt nur die Wahrscheinlichkeitsverteilung über die nächsten
Wörter gegeben alle vorangehenden Wörter.

Die Embeddings haben sich selbst zu einem Raum mit Struktur organisiert.
Der Vektor für king minus der Vektor für man plus der Vektor für woman
liegt sehr nah am Vektor für queen. Das wurde nicht programmiert. Es
ist aus Trainingsdaten entstanden, in denen king und queen in ähnlichen
Kontexten auftraten, aber mit unterschiedlichen geschlechtsspezifischen Pronomen.

Die Attention-Heads haben sich spezialisiert. Manche Heads beachten konsequent
das Subjekt des aktuellen Verbs. Andere beachten kürzlich erwähnte Nomen
im Satz. Andere beachten Interpunktion, um Satzgrenzen zu
verstehen. Diese Spezialisierungen wurden nicht entworfen. Sie sind aus
dem Trainingsziel entstanden, das nächste Wort vorherzusagen.

Die Feed-Forward-Netze sind zu Mustererkennern geworden. Ein Teil des
Netzes aktiviert sich vielleicht stark, wenn es eine Liste von Elementen sieht, weil
Kommas zwischen Elementen weitere Elemente vorhersagen. Ein anderer Teil aktiviert sich vielleicht bei
Daten, weil das Wort in gefolgt von einer Jahreszahl ein bestimmtes
zeitliches Muster vorhersagt. Diese Muster sind über Tausende von
Neuronen verteilt, auf eine Weise, die schwer zu interpretieren, aber mathematisch
optimal für die Vorhersage ist.

---

## Teil 22: Warum das wichtig ist

Eine Maschine, die das nächste Wort mit hoher Genauigkeit vorhersagen kann, ist eine Maschine,
die implizit die Regeln der Sprache gelernt hat. Grammatik. Syntax.
Semantik. Diskursstruktur. Weltwissen. All das ist notwendig,
um genaue Vorhersagen zu treffen. Das Modell muss wissen, dass Verben in Numerus mit
ihren Subjekten übereinstimmen. Es muss wissen, dass Paris in Frankreich liegt und dass
Frankreich in Europa liegt. Es muss wissen, dass ein Satz, der mit
although beginnt, eine gegensätzliche Nebensatzkonstruktion erwarten lässt. Es muss wissen, dass ein Rezept
für einen Kuchen Mehl und Zucker und Eier enthält, nicht Motoröl und Beton.

Das Modell erwirbt all dieses Wissen durch eine einzige Aufgabe: das
nächste Token vorherzusagen. Das ist eine einfache Aufgabe mit tiefgreifenden Implikationen. Ein
System, das vorhersagen kann, was Menschen als Nächstes schreiben werden, ist ein System, das
einen erheblichen Teil menschlichen Wissens in eine Reihe von
Matrixmultiplikationen komprimiert hat.

Die Transformer-Architektur hat das möglich gemacht. Vor den Transformern
konnten Sprachmodelle nur lokale Muster innerhalb weniger Wörter erfassen.
Rekurrente Netze vergaßen Informationen, die mehr als ein paar
Dutzend Wörter zurücklagen. Attention änderte das. Attention lässt jedes Wort
mit jedem anderen Wort interagieren, unabhängig vom Abstand. Ein Wort am
Ende eines Absatzes kann ein Wort am Anfang genauso leicht beachten wie
das Wort direkt daneben.

Der Umfang (Scale) machte das leistungsfähig. GPT-2 mit 1,5 Milliarden Parametern konnte
plausible Absätze schreiben. GPT-3 mit 175 Milliarden Parametern konnte
plausible Aufsätze schreiben und Fragen beantworten und Code generieren. Der Sprung
in der Leistungsfähigkeit kam vollständig durch mehr Daten und mehr Parameter. Die
Architektur blieb fast unverändert.

Die neueste Modellgeneration fügt Instruction Following hinzu. Sie werden
nicht nur trainiert, das nächste Wort vorherzusagen, sondern das nächste Wort
in der Antwort eines hilfreichen und harmlosen Assistenten vorherzusagen. Dieses zusätzliche
Training macht die Modelle als Werkzeuge nützlich statt nur als
Demonstrationen interessant.

Aber unter der Chat-Oberfläche und dem Instruction Tuning und den
Sicherheitsfiltern ist der Kernmechanismus unverändert. Token rein. Attention
hindurch. Feed-Forward durch. Logits raus. Dieselbe Geschichte, die wir
vom Anfang bis zum Ende verfolgt haben. Dieselbe Mathematik. Dieselbe Architektur. Derselbe
Gradientenabstieg, der Schritt für Schritt Cross-Entropy-Loss optimiert.

---

## Epilog: Was Sie als Nächstes bauen können

Sie haben jetzt jedes Teil eines modernen Sprachmodells gesehen. Sie könnten
mit dem Code in diesem Leitfaden eines von Grund auf bauen. Sie könnten es modifizieren.
Mehr Schichten hinzufügen. Einen größeren Datensatz verwenden. Mit anderen
Attention-Mustern experimentieren. SwiGLU gegen eine andere Aktivierung tauschen.

Die hier beschriebene Architektur ist nicht das letzte Wort. Die Forschung
geht weiter. State-Space-Modelle wie Mamba fordern die Dominanz des Transformers
heraus. Mixture of Experts leitet Token durch verschiedene Sub-
Netzwerke, um effizienter zu skalieren. Retrieval Augmented Generation
verbindet Modelle mit externen Wissensdatenbanken. Aber die Kernideen sind
stabil. Embeddings. Attention. Residuals. Normalisierung. Gradienten-
abstieg. Diese werden relevant sein, solange neuronale Netze existieren.

Sie verstehen sie jetzt. Nicht nur was sie sind. Warum sie sind. Jede
Designentscheidung in dieser Architektur wurde getroffen, um ein bestimmtes Problem zu lösen.
Die Residual Connections lösen verschwindende Gradienten. RMSNorm löst
Aktivierungsdrift. SwiGLU löst die Unflexibilität einfacher Aktivierungs-
funktionen. RoPE löst Positional Encoding ohne Parameter. Jedes
Teil erzählt eine Geschichte.

Die Geschichte moderner KI ist die Geschichte vieler Menschen, die über viele Jahre
ein Problem nach dem anderen gelöst und ihre Lösungen zu
etwas Größerem zusammengefügt haben, als die Summe seiner Teile. Sie kennen jetzt jedes Teil.
Sie können einer dieser Menschen sein.
