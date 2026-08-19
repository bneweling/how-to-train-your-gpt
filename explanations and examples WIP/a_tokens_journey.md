# Die Reise eines Tokens: Ein Satz durch das gesamte GPT

Dies ist die Geschichte eines einzigen Satzes. Wir folgen ihm von dem
Moment an, in dem er als Rohtext in das Modell gelangt, bis zu dem
Moment, in dem das Modell das nächste Wort vorhersagt. Jede Zahl ist
real. Jeder Schritt wird erklärt. Am Ende wirst du sehen, wie jedes Teil
der Architektur zusammenwirkt.

## Der Satz

```
"The cat sat on the mat"
```

Sechs Wörter. Ein Punkt. Das ist unser Testobjekt. Wir wollen, dass das
Modell diesen Satz liest und vorhersagt, welches Wort als Nächstes
kommt. Vielleicht *comfortably*. Vielleicht *quietly*. Vielleicht *and*.
Das Modell weiß es noch nicht. Es wird es Schritt für Schritt
herausfinden.

## Schritt 1: Tokenisierung

Das Erste, was das Modell tut, ist, den Satz in Tokens zu zerlegen.
Unser Tokenizer verwendet dasselbe BPE-Vokabular wie GPT-2. Es umfasst
50257 Tokens. Jedes Token ist ein kleines Textstück. Häufige Wörter
bekommen ihr eigenes Token. Satzzeichen bekommen ihr eigenes Token.

```
"The cat sat on the mat."
    ↓  tokenizer
[464, 3797, 3332, 319, 262, 2603, 13]
```

Sieben Tokens für sieben Textstücke. Token 464 ist *The* mit großem T.
Token 3797 ist *cat*. Token 3332 ist *sat*. Token 319 ist *on*. Token
262 ist *the* mit kleinem t. Token 2603 ist *mat*. Token 13 ist der
Punkt. Jedes Token ist nur eine Zahl. Das Modell weiß noch nicht, was
diese Zahlen bedeuten.

## Schritt 2: Embedding-Lookup

Das Modell besitzt eine riesige Lookup-Tabelle. Sie hat 50257 Zeilen.
Jede Zeile ist ein Vektor aus 768 Zahlen. Zeile 3797 ist der Vektor für
*cat*. Zeile 2603 ist der Vektor für *mat*. Das Modell schlägt jede
Token-ID nach und liefert deren Vektor zurück.

```
Token 464 ("The"):  [ 0.023, -0.451,  0.789, ..., -0.102]  (768 Zahlen)
Token 3797 ("cat"): [ 0.019, -0.443,  0.795, ..., -0.098]
Token 3332 ("sat"): [-0.231,  0.567, -0.334, ...,  0.445]
Token 319 ("on"):   [ 0.891,  0.112, -0.334, ...,  0.567]
Token 262 ("the"):  [ 0.773, -0.219,  0.441, ..., -0.332]
Token 2603 ("mat"): [ 0.445, -0.667,  0.223, ..., -0.111]
Token 13 ("."):     [-0.123,  0.456, -0.789, ...,  0.234]
```

Wir haben jetzt eine Matrix der Form 7 mal 768. Sieben Zeilen.
Siebenhundertachtundsechzig Spalten. Diese Matrix ist die Eingabe für
den ersten Transformer-Block.

An diesem Punkt sind die Vektoren zufällig. Das Modell wurde gerade erst
initialisiert. *Cat* und *dog* liegen noch nicht nahe beieinander. Das
Training wird das ändern. Aber schon jetzt kann das Modell sie
verarbeiten. Die Vektoren existieren. Sie haben eine Form. Die
Mathematik kann fließen.

## Schritt 3: Der erste Transformer-Block

Das Modell hat zwölf Transformer-Blöcke. Jeder Block tut dasselbe
Zweierlei. Erstens Attention: Jedes Wort darf mit jedem anderen Wort
sprechen. Zweitens Feed-Forward: Jedes Wort denkt für sich allein
darüber nach, was es gehört hat.

### Schritt 3a: Multi-Head-Attention

Attention ist der Ort, an dem die Magie geschieht. Das Wort *sat* muss
verstehen, was es in diesem Satz tut. Es sollte auf *The* und *cat*
schauen, weil sie ihm sagen, wer sitzt. Es sollte auf *on* und *the* und
*mat* schauen, weil sie ihm sagen, wo. Es sollte auf den Punkt schauen,
weil der ihm sagt, dass der Satz endet.

Der Attention-Mechanismus hat zwölf Heads. Jeder Head betrachtet den
Satz aus einem anderen Blickwinkel. Ein Head könnte sich auf
Subjekt-Verb-Beziehungen konzentrieren. Ein anderer könnte sich auf
Präpositionalphrasen konzentrieren. Wieder ein anderer könnte sich auf
Satzgrenzen konzentrieren.

Für einen Head mit unseren sieben Tokens macht das Modell Folgendes:

Zuerst projiziert es jedes Token in drei Räume. Der Query-Raum fragt:
Wonach suche ich? Der Key-Raum sagt: Was habe ich anzubieten? Der
Value-Raum enthält meinen eigentlichen Inhalt.

```
Für Token "sat" (Position 2):
Q = [ 0.34, -0.12,  0.78, ...]  (64 Zahlen für diesen Head)
K = [-0.23,  0.56, -0.41, ...]  (64 Zahlen)
V = [ 0.67, -0.89,  0.12, ...]  (64 Zahlen)
```

Dann fragt es jedes andere Token: Wie gut passt meine Query zu deinem
Key? Das ist der Attention-Score. Ein hoher Score bedeutet, dass sat
wirklich will, was dieses Token anbietet. Ein niedriger Score bedeutet,
dass es sat egal ist.

Schauen wir uns die berechneten Scores von sat gegenüber jedem Token an.
Vor dem Softmax sind das rohe Zahlen. Größer ist besser.

```
sat achtet auf "The" (Pos. 0): Score = 0.42  (Subjekt des Satzes)
sat achtet auf "cat" (Pos. 1): Score = 0.78  (wer sitzt)
sat achtet auf "sat" (Pos. 2): Score = 0.15  (sich selbst, immer etwas Self-Attention)
sat achtet auf "on"  (Pos. 3): Score = 0.31  (wo das Sitzen stattfindet)
sat achtet auf "the" (Pos. 4): Score = 0.22
sat achtet auf "mat" (Pos. 5): Score = 0.28  (das Objekt unter der Katze)
sat achtet auf "."   (Pos. 6): Score = 0.05  (Satzzeichen, am wenigsten wichtig)
```

Das Wort *cat* erhält den höchsten Score. Das ergibt Sinn. Das Verb
*sat* muss wissen, wer sitzt. *cat* ist das Subjekt. *on* und *mat* sind
ebenfalls relevant, aber weniger entscheidend. Der Punkt ist für das
Verständnis der Handlung irrelevant.

Diese Scores werden durch die Quadratwurzel von 64, also 8, geteilt. Das
verhindert, dass die Zahlen zu groß werden. Anschließend wandelt Softmax
sie in Prozentwerte um.

```
Nach Softmax:
sat achtet auf "The": 0.18  (18 Prozent Attention)
sat achtet auf "cat": 0.35  (35 Prozent)
sat achtet auf "sat": 0.10  (10 Prozent)
sat achtet auf "on":  0.13  (13 Prozent)
sat achtet auf "the": 0.11  (11 Prozent)
sat achtet auf "mat": 0.12  (12 Prozent)
sat achtet auf ".":   0.01  (1 Prozent)
Summe: 1.00 (100 Prozent)
```

Nun mischt das Modell die Value-Vektoren anhand dieser Prozentwerte. Die
neue Repräsentation von *sat* ist eine gewichtete Mischung aller Values.

```
Neues sat = 0.18 × V_The + 0.35 × V_cat + 0.10 × V_sat
          + 0.13 × V_on  + 0.11 × V_the + 0.12 × V_mat
          + 0.01 × V_period
```

Das neue *sat* enthält nun Informationen über *cat* und *on* und *mat*.
Es kennt sein Subjekt und sein Objekt. Es ist nicht mehr nur das Wort
*sat*. Es ist das Wort *sat* im Kontext dieses spezifischen Satzes.

Derselbe Prozess läuft für jedes Token ab. *mat* blickt zurück, sieht
*on* und *the* und erkennt, dass es Teil einer Präpositionalphrase ist.
*The* schaut auf *cat* und erkennt, dass es ein Nomen modifiziert. Jedes
Token gewinnt Kontext von jedem anderen Token.

### Schritt 3b: Das Feed-Forward-Netzwerk

Nach der Attention hat jedes Token Informationen von allen anderen
Tokens vermischt. Jetzt muss jedes Token unabhängig darüber nachdenken,
was es gerade gelernt hat. Das Feed-Forward-Netzwerk verarbeitet jedes
Token einzeln mit denselben Weights.

Das Feed-Forward-Netzwerk erweitert von 768 Dimensionen auf 3072
Dimensionen und dann wieder zurück auf 768. In der Mitte wendet es die
SwiGLU-Aktivierung an. Das Gate entscheidet, welche Informationen
behalten und welche verworfen werden.

```
Für Token "sat":
Eingabe:  [0.45, -0.23, 0.67, ..., -0.11]  (768 Zahlen, durch Attention kontextbewusst)
Erweitern: Multiplikation mit W1 (768 → 3072)
Gate:   Multiplikation mit W2 (768 → 3072), danach Gate anwenden
Kombinieren: SiLU(Erweitern) × Gate  (elementweise Multiplikation)
Projektion: Multiplikation mit W3 (3072 → 768)
Ausgabe: [0.41, -0.28, 0.71, ..., -0.09]  (768 Zahlen, weiter verarbeitet)
```

Bevor die FFN-Ausgabe endgültig übernommen wird, addiert die Residual
Connection die ursprüngliche Eingabe wieder hinzu. Das ist die
Gradienten-Autobahn. Selbst wenn das FFN Unsinn produziert hätte, würde
das ursprüngliche Signal trotzdem hindurchgelangen.

```
Finale Ausgabe dieses Blocks = Eingabe + FFN_output
                              = [0.45, -0.23, ..., -0.11] + [0.41, -0.28, ..., -0.09]
                              = [0.86, -0.51, ..., -0.20]
```

Das Token wurde aktualisiert. Es trägt mehr Informationen als zuvor. Die
ursprüngliche Bedeutung ist noch vorhanden, wurde aber verfeinert.

## Schritt 4: Durch die übrigen Blöcke

Derselbe Prozess wiederholt sich noch elf weitere Male. Mit jedem Block
werden die Tokens ein wenig besser darin, einander zu verstehen. Die
Attention-Muster werden ausgefeilter. Die Feed-Forward-Netzwerke fügen
mehr Nuancen hinzu.

Bis Block zwölf ähnelt die Repräsentation von *sat* in keiner Weise mehr
dem ursprünglichen Embedding. Sie wurde durch Attention auf ihr Subjekt
*cat* geformt. Sie wurde durch Attention auf ihr Objekt *mat* geformt.
Sie wurde durch zwölf Feed-Forward-Netzwerke verarbeitet. Die
ursprünglichen 768 Zahlen wurden in 768 Zahlen umgewandelt, die alles
kodieren, was das Modell über dieses spezifische Vorkommen des Verbs
*sat* weiß.

## Schritt 5: Das nächste Wort vorhersagen

Nach dem letzten Transformer-Block wendet das Modell ein abschließendes
RMSNorm an, um die Repräsentationen zu bereinigen. Anschließend
projiziert es den Vektor des letzten Tokens zurück in den
Vokabularraum.

```
Vektor für "." (Position 6, das letzte Token):
[0.12, -0.34, 0.56, ..., -0.78]  (768 Zahlen)

Projektion in das Vokabular:
Multiplikation mit der LM-Head-Weight-Matrix (768 × 50257)
Ergebnis: 50257 Zahlen. Ein Score für jedes mögliche nächste Token.
```

Diese 50257 Zahlen werden Logits genannt. Der höchste Logit ist die
beste Schätzung des Modells für das nächste Wort. Schauen wir uns die
fünf besten Vorhersagen an.

```
Token 13  ("."):    Logit = -0.23  (das Modell könnte einen weiteren Punkt vorhersagen)
Token 290 (" and"): Logit = 2.34   (und was dann als Nächstes geschah)
Token 3797 ("cat"): Logit = -1.45  (unwahrscheinlich, dass cat hier wiederholt wird)
Token 198 ("\n"):   Logit = 3.12   (einen neuen Absatz beginnen)
Token 50256 ("<|endoftext|>"): Logit = 4.56  (das Dokument beenden)
```

Der höchste Score ist 4.56 für das End-of-Text-Token. Das Modell hält
den Satz für abgeschlossen. Der zweithöchste ist 3.12 für einen
Zeilenumbruch. Der dritte ist 2.34 für das Wort *and*.

Verwenden wir Greedy Sampling, wählen wir den höchsten Wert. Das Modell
gibt das End-of-Text-Token aus. Die Generierung stoppt. Der Satz ist
fertig.

Verwenden wir eine Temperature von 0.8, streuen die Wahrscheinlichkeiten
stärker. Das Modell könnte stattdessen *and* wählen. Die Geschichte geht
weiter.

```
"The cat sat on the mat. And then it stretched and yawned..."
```

Verwenden wir eine Temperature von 1.5 mit Top-k 50, wird das Modell
kreativ.

```
"The cat sat on the mat. Quietly watching the birds through the window..."
```

Die Wahl des nächsten Wortes hängt von den Sampling-Parametern ab. Aber
die rohe Vorhersage des Modells, die Logits, ist immer dieselbe. Das
Modell hält es immer für am wahrscheinlichsten, dass der Satz endet. Die
Sampling-Regler entscheiden, ob diesem Rat gefolgt oder Alternativen
erkundet werden.

## Was gerade passiert ist

Ein Satz. Sieben Tokens. Durch die Tokenisierung. Durch das Embedding.
Durch zwölf Transformer-Blöcke, von denen jeder Attention- und
Feed-Forward-Layer enthält. Durch die finale Normalisierung. Durch die
Ausgabeprojektion. Das Modell hat den Satz gelesen, verstanden und
vorhergesagt, was als Nächstes kommt.

Die Vorhersage wurde durch Attention ermöglicht. Das Verb *sat*
verstand sein Subjekt *cat* und sein Objekt *mat*, weil Attention es
jedes vorangegangene Wort betrachten ließ. Die Feed-Forward-Netzwerke
verfeinerten dieses Verständnis. Die Residual Connections hielten den
Gradienten am Fließen.

Jeder Schritt, den wir nachvollzogen haben, ist derselbe, egal ob das
Modell zwölf Layer hat oder sechsundneunzig. Egal ob die
Embedding-Dimension 768 oder 12288 beträgt. Die Mathematik ändert sich
nicht. Nur die Zahlen werden größer. Dieser Satz. Diese sieben Tokens.
Das ist es, was jedes moderne Sprachmodell milliardenfach jeden Tag tut.

## Was das Modell gelernt hat

Nach dem Training mit Milliarden von Sätzen sind die Embeddings des
Modells nicht mehr zufällig. Der Vektor für *cat* hat sich nahe an
*dog* und *pet* und *feline* bewegt. Der Vektor für *sat* hat sich nahe
an *rested* und *perched* und *settled* bewegt. Der Embedding-Raum hat
sich in Bedeutungsnachbarschaften organisiert.

Die Attention-Muster haben sich spezialisiert. Manche Heads schauen
immer rückwärts, um das Subjekt eines Verbs zu finden. Andere Heads
verfolgen, welche Nomen zuletzt erwähnt wurden. Wieder andere Heads
konzentrieren sich auf Satzzeichen, um Satzgrenzen zu erkennen. Diese
Muster entstehen aus den Trainingsdaten. Niemand hat sie programmiert.
Das Modell hat sie entdeckt, weil sie dabei helfen, das nächste Wort
vorherzusagen.

Die Feed-Forward-Netzwerke sind zu Wissensspeichern geworden. Ein Teil
des Netzwerks erkennt vielleicht, dass *cat* und *mat* häufig zusammen
auftreten. Ein anderer Teil weiß vielleicht, dass Sätze über Katzen oft
mit Sitzen, Schlafen oder Jagen zu tun haben. Diese Assoziationen werden
während des Trainings in die Weights eingebrannt.

## Das Fazit

Ein Sprachmodell ist eine Vorhersagemaschine. Gib ihm Tokens, und es
errät das nächste. Alles andere folgt daraus. Die Architektur ist darauf
ausgelegt, diese Vorhersagen so präzise wie möglich zu machen. Das
Training ist darauf ausgelegt, Muster aus Milliarden von Sätzen zu
extrahieren. Die Inference-Tricks sind darauf ausgelegt, die Generierung
schnell und steuerbar zu machen.

Aber im Kern ist es immer dieselbe Geschichte. Tokens rein. Attention
hindurch. Feed Forward durch. Logits raus. Eins auswählen. Wiederholen.
Das ist das ganze Geheimnis moderner KI.
