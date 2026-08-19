# Perplexity: Die eine Zahl, die dein Modell misst

## Die kurze Antwort

Perplexity ist eine einzelne Zahl, die dir sagt, wie gut ein Sprachmodell
ist. Niedriger ist besser. Eine Perplexity von 10 bedeutet, dass das
Modell bei jedem Schritt so verwirrt ist, als müsste es zwischen 10
gleich wahrscheinlichen Wörtern wählen. Eine Perplexity von 1 bedeutet,
dass das Modell jedes Mal genau weiß, welches Wort als Nächstes kommt.
Reale Sprachmodelle liegen bei echtem Text meist bei einer Perplexity
zwischen 10 und 100.

Perplexity ist nicht abstrakt. Sie ist der exponenzierte Cross-Entropy-
Loss. Wenn dein Trainings-Loss 3.0 beträgt, ist deine Perplexity e hoch
3, also etwa 20. Das Modell ist so verwirrt, als würde es zufällig
unter 20 Optionen wählen.

## Woher die Zahl kommt

Jedes Mal, wenn das Modell das nächste Wort vorhersagt, weist es jedem
Token im Vokabular eine Wahrscheinlichkeit zu. Das korrekte Token
erhält eine Wahrscheinlichkeit P. Das Modell ist sich bei dieser
Vorhersage unsicher. Der Cross-Entropy-Loss misst diese Unsicherheit
als negativen Logarithmus von P. Perplexity ist e hoch dieser Loss.

```
Für eine einzelne Vorhersage:
  Modell sagt P("mat") = 0.25
  Loss = -ln(0.25) = 1.386
  Perplexity = e^1.386 = 4.0

Interpretation: Das Modell war so unsicher, als hätte es zufällig
unter 4 gleich wahrscheinlichen Optionen wählen müssen.
```

Das Besondere an Perplexity ist, dass sie eine abstrakte Loss-Zahl in
etwas Vorstellbares übersetzt. Ein Loss von 1.386 bedeutet den meisten
Menschen nichts. Eine Perplexity von 4 bedeutet, dass das Modell unter
4 Optionen wählt. Das ist greifbar. Du kannst dir vorstellen, zufällig
aus 4 Wörtern zu wählen.

## Perplexity versus Loss

Während des Trainings beobachtest du, wie der Loss sinkt. Der Loss
startet bei etwa 10.8 für ein Modell mit 50257 Vokabular-Tokens. Das
ist nicht interpretierbar. Aber du kannst ihn umrechnen.

```
loss = 10.82:  perplexity = e^10.82 ≈ 50,000  (zufällig. Modell weiß nichts)
loss = 7.0:    perplexity = e^7.0  ≈ 1,100    (lernt Wörterfrequenzen)
loss = 5.0:    perplexity = e^5.0  ≈ 150      (lernt grundlegende Grammatik)
loss = 3.0:    perplexity = e^3.0  ≈ 20       (brauchbares Sprachmodell)
loss = 2.0:    perplexity = e^2.0  ≈ 7.4      (gutes Sprachmodell)
loss = 1.5:    perplexity = e^1.5  ≈ 4.5      (sehr gut)
loss = 1.0:    perplexity = e^1.0  ≈ 2.7      (exzellent)
```

Perplexity gibt dir ein mentales Modell dafür, was der Loss
tatsächlich bedeutet. Wenn dein Loss von 10.8 auf 7.0 sinkt, hast du
dich nicht einfach nur um 3.8 Einheiten verbessert. Du bist von einer
Verwirrung wie bei 50000 Optionen zu einer Verwirrung wie bei 1100
Optionen übergegangen. Das ist eine dramatische Verbesserung.

## Wie man sie berechnet

Im Code ist Perplexity eine einzige Zeile.

```python
import math

loss = 3.0  # Der Cross-Entropy-Loss deines Modells
perplexity = math.exp(loss)

print(f"Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"The model is as uncertain as picking among {perplexity:.0f} options.")
```

Für einen Batch von Vorhersagen berechnest du zuerst den
durchschnittlichen Loss und exponenzierst ihn dann.

```python
total_loss = 0
total_tokens = 0
for input_ids, target_ids in dataloader:
    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += target_ids.numel()

average_loss = total_loss / total_tokens
perplexity = math.exp(average_loss)
print(f"Validation perplexity: {perplexity:.2f}")
```

Berechne Perplexity immer auf Daten, die das Modell während des
Trainings nicht gesehen hat. Die Trainings-Perplexity kann irreführend
niedrig sein, weil das Modell Teile der Trainingsdaten auswendig
gelernt hat. Die Validierungs-Perplexity misst, wie gut das Modell
generalisiert.

## Was unterschiedliche Perplexity-Werte bedeuten

### Perplexity um 50000

Dein Modell ist zufällig. Es weist jedem Token im Vokabular die
gleiche Wahrscheinlichkeit zu. Es hat nichts gelernt. Das ist bei
Schritt null des Trainings normal. Wenn es nach Tausenden von
Schritten immer noch hier steht, ist etwas kaputt. Überprüfe deine
Loss-Funktion und deinen Optimizer.

### Perplexity um 1000

Das Modell hat gelernt, dass manche Wörter häufiger vorkommen als
andere. Es weiß, dass *the* oft vorkommt und *xylophone* selten. Es
nutzt diese Häufigkeiten in seinen Vorhersagen. Aber es versteht noch
nicht Wortreihenfolge, Grammatik oder Bedeutung. Die Ausgabe ist
Kauderwelsch, aber das Kauderwelsch enthält häufige Wörter in ungefähr
den richtigen Proportionen.

### Perplexity um 100

Das Modell hat grundlegende Grammatik gelernt. Es weiß, dass Artikel
vor Substantiven stehen. Es weiß, dass Verben im Numerus mit dem
Subjekt übereinstimmen. Es weiß, dass Sätze mit einem Punkt enden. Die
Ausgabe hat eine erkennbare Satzstruktur, auch wenn der Inhalt
unsinnig ist. Hier stagnieren die meisten kleinen Modelle nach
begrenztem Training.

### Perplexity um 20

Das Modell schreibt zusammenhängenden Text. Sätze haben Subjekt, Verb
und Objekt in der richtigen Reihenfolge. Der Inhalt ist manchmal
faktisch korrekt und manchmal erfunden. Das ist das Niveau von GPT-1
aus dem Jahr 2018. Ein Modell mit 17 Millionen Parametern, trainiert
auf 100 Millionen Tokens, kann dieses Niveau erreichen.

### Perplexity um 10

Das Modell schreibt guten Text. Der Inhalt ist größtenteils faktisch
korrekt. Wenige offensichtliche Fehler. Das ist das Niveau von GPT-2
aus dem Jahr 2019. Ein Modell mit 150 Millionen Parametern, trainiert
auf Milliarden von Tokens, kann das erreichen.

### Perplexity um 5

Das Modell schreibt exzellenten Text. Macht selten faktische Fehler.
Bewältigt komplexes Reasoning. Das ist das Niveau von GPT-3 aus dem
Jahr 2020 und modernen kleinen Modellen wie LLaMA 7B. Das Training
dieser Modelle kostet Millionen von Dollar.

### Perplexity unter 3

Das Modell nähert sich menschlicher Leistung beim Sprachmodellieren
an. Es sagt mit hoher Genauigkeit vorher, was ein Mensch schreiben
würde. Modelle auf diesem Niveau werden anhand schwierigerer Aufgaben
wie Question Answering und Codegenerierung gemessen, weil Perplexity
als Metrik ihre Aussagekraft verliert. Der Unterschied zwischen
Perplexity 2.5 und 2.3 ist kaum spürbar, aber teuer zu erreichen.

## Warum Perplexity nicht alles ist

Perplexity misst, wie gut das Modell das nächste Wort vorhersagt. Sie
misst nicht, ob das Modell hilfreich, wahrheitsgetreu, sicher oder
kreativ ist. Ein Modell kann eine hervorragende Perplexity haben und
trotzdem schädliche Inhalte erzeugen. Es kann eine hervorragende
Perplexity haben und trotzdem Fakten halluzinieren. Es kann eine
hervorragende Perplexity haben und trotzdem langweilig sein.

Perplexity hängt außerdem vom Datensatz ab. Ein Modell, das auf
Kinderbüchern trainiert wurde, hat eine niedrige Perplexity bei
Kinderbüchern und eine hohe Perplexity bei juristischen Dokumenten.
Perplexity ist immer relativ zu den Testdaten. Ein Vergleich der
Perplexity zwischen Modellen ist nur sinnvoll, wenn die Modelle auf
demselben Datensatz mit demselben Tokenizer ausgewertet werden.

Unterschiedliche Tokenizer erzeugen unterschiedliche Perplexity-Werte
für dasselbe Modell auf denselben Daten. Ein Tokenizer mit einem
größeren Vokabular liefert meist eine niedrigere Perplexity, weil
jedes Token mehr Information kodiert und pro Satz weniger Vorhersagen
nötig sind. Deshalb kannst du Perplexity nicht zwischen Modellen
vergleichen, die unterschiedliche Tokenizer verwenden.

## Der Zusammenhang mit Bits pro Zeichen

Perplexity lässt sich in Bits pro Zeichen umrechnen. Bits pro Zeichen
misst, wie viele Bits an Information das Modell im Durchschnitt
braucht, um jedes Zeichen des Textes zu kodieren. Niedriger ist
besser. Der Zusammenhang lautet:

```
bits_per_character = ln(perplexity) / (ln(2) × characters_per_token)
```

Bei GPT-2 deckt jedes Token im Durchschnitt etwa 4 Zeichen ab.

```
perplexity = 20:
  bits_per_char = ln(20) / (0.693 × 4) ≈ 1.08 bits per character

perplexity = 10:
  bits_per_char = ln(10) / (0.693 × 4) ≈ 0.83 bits per character
```

Das zeigt dir, dass das Modell etwa ein Bit an Information pro Zeichen
braucht, um englischen Text zu kodieren. Die theoretische
Mindestentropie des Englischen liegt bei etwa 0.6 bis 1.0 Bits pro
Zeichen. Modelle nähern sich dieser Grenze an. Weitere Verbesserungen
der Perplexity erfordern ein besseres Verständnis von Bedeutung, nicht
nur bessere Statistik.

## Perplexity während des Trainings

Ein guter Trainingslauf sollte zeigen, wie die Perplexity gleichmäßig
sinkt. Die Start-Perplexity sollte nahe an der Vokabulargröße liegen.
Die finale Perplexity hängt von Modellgröße, Datenqualität und
Trainingsdauer ab.

```
Schritt    Loss    Perplexity
0        10.82     50,257     Modell weiß nichts
100       9.23     10,240     Lernt Häufigkeiten
500       7.45      1,720     Lernt Wortmuster
1,000     6.12        455     Entstehende Grammatik
5,000     4.23         69     Zusammenhängende Phrasen
10,000    3.45         31     Brauchbare Sätze
50,000    2.89         18     Gutes Modell
```

Wenn die Perplexity vor Schritt 10000 aufhört zu sinken, schränkt
etwas das Modell ein. Die Learning Rate könnte zu niedrig sein. Die
Modellkapazität könnte ausgeschöpft sein. Die Daten enthalten
vielleicht nicht genug Muster zum Lernen. Versuche, die Learning Rate,
die Modellgröße oder die Datensatzgröße zu erhöhen.

Wenn die Perplexity auf den Trainingsdaten sinkt, aber auf den
Validierungsdaten steigt, overfittet das Modell. Es lernt den
Trainingsdatensatz auswendig, statt allgemeine Muster zu lernen. Die
Trainings- und Validierungskurven laufen auseinander. Füge mehr
Dropout oder Weight Decay hinzu oder nutze Early Stopping.

## Perplexity bekannter Modelle

Alle Zahlen sind Näherungswerte und hängen vom Evaluationsdatensatz ab.

```
Modell                 Params     Perplexity (WikiText-103)
Zufalls-Baseline       :          50,257
GPT-1 (2018)           117M       ~35
GPT-2 Small (2019)     124M       ~19
GPT-2 Medium            350M       ~15
GPT-2 Large             774M       ~12
GPT-3 Small (2020)      125M       ~18
GPT-3 XL               1.3B       ~10
GPT-3 6.7B                       ~8
GPT-3 175B                       ~5
LLaMA 7B (2023)          7B       ~7
LLaMA 13B                        ~6
LLaMA 70B                        ~4
```

Beachte Folgendes. GPT-2 Small erreicht mit 124 Millionen Parametern
eine Perplexity von 19. GPT-3 erreicht mit 125 Millionen Parametern
eine Perplexity von 18. Gleiche Architektur. Gleiche Größe.
Unterschiedliches Training. GPT-3 wurde auf mehr Daten und länger
trainiert. Das zusätzliche Compute verbesserte die Perplexity, ohne
die Modellgröße zu erhöhen. Deshalb sind Datenqualität und
Trainingsdauer genauso wichtig wie die Modellarchitektur.

## Was du dir merken solltest

Perplexity ist der exponenzierte Cross-Entropy-Loss. Eine Perplexity
von N bedeutet, dass das Modell so unsicher ist, als müsste es
zufällig unter N Optionen wählen. Niedriger ist besser. Zufällige
Modelle starten bei etwa der Vokabulargröße. Gute Modelle erreichen
einstellige Werte.

Perplexity übersetzt die abstrakte Loss-Zahl in etwas Anschauliches.
Wenn dein Loss von 10 auf 5 sinkt, ist dein Modell von einer Verwirrung
unter 22000 Optionen zu einer Verwirrung unter 150 Optionen
übergegangen. Das ist der Unterschied zwischen einem Modell, das
nichts weiß, und einem Modell, das etwas weiß.

Perplexity misst nicht Hilfsbereitschaft, Wahrheitsgehalt oder
Sicherheit. Sie misst nur, wie gut das Modell das nächste Wort
vorhersagt. Um zu beurteilen, ob dein Modell nützlich ist, brauchst du
andere Metriken. Aber um zu verfolgen, ob dein Modell lernt, ist
Perplexity die mit Abstand wichtigste Zahl.
