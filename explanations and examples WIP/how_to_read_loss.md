# So liest du eine Loss-Kurve

## Die kurze Antwort

Eine Loss-Kurve ist die wichtigste Diagnose beim Training eines
Sprachmodells. Sie zeigt dir, ob das Modell lernt, ob es gut lernt
und ob etwas kaputt ist. Eine gute Kurve fällt gleichmäßig ab. Das
ist der Idealfall. Alles andere ist ein Problem. Dieser Leitfaden
zeigt dir jedes verbreitete Muster und was dagegen zu tun ist.

## Die Ausgangslage

Du trainierst ein Modell. Alle 100 Schritte zeichnest du den Loss
auf. Du trägst die Schritte auf der x-Achse und den Loss auf der
y-Achse ab. Du willst, dass die Linie fällt. Aber die Form der Linie
ist wichtiger als die Richtung. Zwei Trainingsläufe können beide
einen sinkenden Loss haben, aber der eine ist zum Scheitern verurteilt
und der andere gedeiht.

Alle folgenden Beispiele gehen von einem Start-Loss von etwa 10.8
aus – das ist der Loss eines zufälligen Modells, das gleichverteilt
über GPT-2s Vokabular aus 50257 Token vorhersagt. Der Loss entspricht
dem negativen Logarithmus von eins geteilt durch 50257, was ungefähr
ln(50257) und damit ungefähr 10.82 ergibt.

## Muster 1: Die gute Kurve

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *
      |     **
  5.0 |       ***
      |          ******
  3.0 |                ************
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
                    Schritte
```

Der Loss startet bei etwa 10.8. In den ersten paar tausend Schritten
fällt er schnell. Mit fortschreitendem Training verlangsamt sich der
Abfall. Die Kurve ist glatt. Keine Spikes. Keine Plateaus. So sieht
erfolgreiches Training aus.

Der schnelle Abfall am Anfang zeigt, wie das Modell grundlegende
Muster lernt. Großbuchstaben stehen am Satzanfang. Punkte beenden
Sätze. Häufige Wörter tauchen an vorhersagbaren Positionen auf. Diese
Muster sind leicht zu lernen, weil sie konsistent sind. Der Loss
fällt schnell.

Der langsamere Abstieg später zeigt, wie das Modell subtilere Muster
lernt. Subjekt-Verb-Kongruenz. Pronomenauflösung. Der Unterschied
zwischen den englischen Wörtern „affect“ und „effect“. Diese Muster
sind schwerer, weil sie von weitreichendem Kontext und feinen
Bedeutungsnuancen abhängen.

Die Kurve erreicht nie vollständig ein Plateau, weil das Modell immer
noch etwas dazulernen kann. Es gibt immer einen etwas besseren Satz
von Weights, der das nächste Wort ein kleines Stück genauer
vorhersagt.

## Muster 2: Die flache Linie

```
Loss
 10.8 |********************************************
      |
  8.0 |
      |
  5.0 |
      |
  2.0 |
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
```

Der Loss bewegt sich nicht. Er bleibt für immer bei 10.8. Das Modell
lernt überhaupt nicht. Das ist fast immer ein Bug.

Mögliche Ursachen in der Reihenfolge ihrer Wahrscheinlichkeit.

Die Learning Rate ist null oder der Optimizer führt keinen Schritt
aus. Prüfe, ob `optimizer.step()` aufgerufen wird. Prüfe, ob
`optimizer.zero_grad()` NACH step aufgerufen wird und nicht davor.

Die Gradients sind null. Ein Bug in der Loss-Berechnung. Prüfe, ob
Logits und Targets korrekt ausgerichtet sind. Die Logits sollten die
Vorhersage des nächsten Tokens betreffen. Die Targets sollten die
tatsächlichen nächsten Token sein.

Die Weights des Modells werden nicht aktualisiert. Prüfe `weight.grad`
nach `loss.backward()`. Er sollte für mindestens einige Weights
ungleich null sein.

Die Daten sind kaputt. Vielleicht sind Input und Target identisch.
Vielleicht ist bei allen Targets dasselbe Token. Gib ein paar Batches
aus und untersuche sie.

## Muster 3: Loss-Spikes

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *      /
  7.0 |    *    /
      |     ** /
  5.0 |    /  **
      |   /      ***
  3.0 |  /           ******
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
                    ↑
                   Spike
```

Der Loss ist schön gefallen. Dann ist er um den Faktor zwei oder drei
nach oben gesprungen. Das Modell lernt noch, aber es ist etwas
Schlimmes passiert.

Ursache: Gradient Explosion. Ein seltener Batch der Trainingsdaten
enthielt ein ungewöhnliches Muster. Die Gradients wurden sehr groß.
Die Weights machten einen großen Schritt in eine schlechte Richtung.

Fix: Gradient Clipping hinzufügen. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Rufe dies nach `loss.backward()` und vor `optimizer.step()` auf. Unser Trainingscode macht das bereits.

Wenn Clipping bereits vorhanden ist, ist die Max-Norm vielleicht zu
hoch. Versuche, sie von 1.0 auf 0.5 zu reduzieren. Wenn die Spikes
weiterhin auftreten, senke die Learning Rate.

Ein einzelner Spike ist nicht fatal. Das Modell erholt sich meistens
wieder. Aber wiederholte Spikes sind ein Zeichen dafür, dass
systematisch etwas nicht stimmt. Die Daten könnten beschädigte
Beispiele enthalten. Die Learning Rate könnte zu hoch sein. Die
Architektur könnte einen Bug darin haben, wie sie bestimmte
Sequenzlängen behandelt.

## Muster 4: Das Plateau

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *
      |     *
  6.5 |      ******************************
      |
  3.0 |
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
               ↑
          Plateau beginnt
```

Der Loss fällt auf etwa 6.5 und bleibt dann stehen. Egal wie viele
weitere Schritte du trainierst, der Loss weigert sich, weiter zu
fallen. Das Modell ist an eine Wand gestoßen.

Man nennt das ein Learning-Rate-Plateau. Die Learning Rate ist zu
niedrig, als dass das Modell seinen aktuellen Bereich im Weight-Space
verlassen könnte. Die Gradients sind zu klein, um das Modell in eine
bessere Konfiguration zu bewegen.

Fix: Erhöhe die Learning Rate. Wenn du die Cosine-Schedule
verwendest, versuche, die Peak-Learning-Rate von 3e-4 auf 1e-3 zu
erhöhen. Wenn du bereits am Minimum der Schedule bist, starte mit
einem höheren Peak neu.

Alternative Ursache: Die Kapazität des Modells ist ausgeschöpft. Ein
Modell mit 17 Millionen Parametern, das auf 100 Megabyte Text
trainiert wird, kann nur so viel erfassen. Der Loss kann nicht unter
eine bestimmte Untergrenze fallen, die durch Modellgröße und
Datenkomplexität bestimmt wird. Um niedriger zu kommen, brauchst du
ein größeres Modell oder mehr Daten oder beides.

## Muster 5: Der Trainings-Loss sinkt, aber der Validierungs-Loss steigt

```
Loss
 10.8 |*
  9.0 | *    Training (durchgezogen)
  7.0 |  *         *
      |   *       * *
  5.0 |    *     *   *
      |     *   *     *
  3.0 |      ***       ************* Validierung (gestrichelt)
      |        *                     **************
  1.0 |         *****                               ******
      +-----+-----+-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k   35k   40k
                   ↑
              Overfitting beginnt
```

Der Trainings-Loss sinkt weiter. Der Validierungs-Loss ist auch
gesunken, hat dann aber begonnen zu steigen. Das Modell overfittet.
Es merkt sich die Trainingsdaten, statt allgemeine Muster zu lernen.

Fix: Stoppe das Training an dem Punkt, an dem der Validierungs-Loss
anfängt zu steigen. Das nennt man Early Stopping. Du musst das
Modell nicht reparieren. Du musst nur aufhören, bevor es overfittet.

Vorbeugung: Erhöhe Dropout von 0.1 auf 0.2 oder 0.3. Erhöhe Weight
Decay von 0.1 auf 0.2. Verwende ein größeres und vielfältigeres
Trainings-Dataset. Das Modell kann sich Daten nicht merken, die es
nicht oft genug gesehen hat.

## Muster 6: Loss wird negativ oder NaN

```
Loss
 10.8 |*
      | *
  9.0 |  *
      |   *
  7.0 |    *    |
      |     *   |
  5.0 |      *  |
      |       * |
  3.0 |        *|
      |         *
  0.0 +----------*--------*-------*------
  --------------------------------
  NaN  NaN  NaN  NaN  NaN  NaN  NaN
```

Der Loss ist gesunken. Dann ist er auf null gefallen. Dann wurde er
zu NaN. Das Training ist tot. Das Modell produziert Unendlich- oder
NaN-Werte.

Ursache: numerischer Overflow. Die Logits oder die Loss-Berechnung
haben Zahlen erzeugt, die zu groß waren, um in bfloat16 oder float32
dargestellt zu werden. Das passiert, wenn die Learning Rate zu hoch
ist und die Weights explodieren.

Fix: Senke die Learning Rate drastisch. Füge Gradient Clipping hinzu.
Prüfe, ob du in den Attention-Scores durch `sqrt(head_dim)` teilst.
Ohne diese Division können die Scores groß genug werden, dass
`exp(score)` bei bfloat16 überläuft. Prüfe, ob deine Loss-Berechnung
korrekt ist. Cross Entropy mit Logits von 1000 und Targets erzeugt
NaN.

Wenn du Mixed Precision verwendest, prüfe, ob der Autocast-Kontext
korrekt eingerichtet ist. Manche Operationen wie Softmax und
LayerNorm sollten für numerische Stabilität in float32 bleiben.
Autocast kümmert sich darum, aber nur, wenn das Backend das
unterstützt.

## Muster 7: Die Stufenfunktion

```
Loss
 10.8 |*
      | *
  9.0 |******
      |      ******
  7.0 |           ******
      |                ******
  5.0 |                     ******
      |                          ******
  3.0 |                               ******
      +-----+-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k   30k
```

Der Loss fällt in regelmäßigen Abständen plötzlich ab. Die Abfälle
fallen mit Änderungen der Learning Rate zusammen. Die Learning Rate
wurde um den Faktor 10 reduziert, und der Loss ist nach unten
gesprungen.

Ursache: eine Learning-Rate-Schedule mit Step Decay. Manche
Trainings-Setups nutzen das absichtlich. Aber die plötzlichen
Abfälle können das Momentum des Modells stören. Das Modell hat eine
Weile gebraucht, um sich an die alte Learning Rate anzupassen, und
jetzt hat sich die Rate abrupt geändert.

Empfehlung: Verwende Cosine Decay statt Step Decay. Cosine Decay ist
glatt. Das Modell passt sich kontinuierlich an. Keine plötzlichen
Sprünge. Unser Trainingscode verwendet Cosine Warmup mit Cosine
Decay. Keine Stufenänderungen.

## Muster 8: Der Sägezahn

```
Loss
 10.8 |*  *  *  *
      | *  *  *  *
  9.0 |*  *  *  *
      | *  *  *  *
  7.0 |*  *  *  *  *  *
      | *  *  *  *  *  *
  5.0 |*  *  *  *  *  *  *  *
      | *  *  *  *  *  *  *  *
  3.0 |*  *  *  *  *  *  *  *  *  *
      +-----+-----+-----+-----+-----+
      0    5k   10k   15k   20k   25k
```

Der Loss springt alle paar Schritte auf und ab. Der Gesamttrend ist
fallend, aber das Rauschen ist hoch.

Ursache: Die Batch Size ist zu klein. Jeder Batch ist zu klein, um
repräsentativ für die gesamte Datenverteilung zu sein. Manche
Batches sind einfach und erzeugen einen niedrigen Loss. Andere sind
schwer und erzeugen einen hohen Loss. Das Modell überreagiert auf
jeden einzelnen Batch.

Fix: Erhöhe die Batch Size oder verwende Gradient Accumulation, um
einen größeren Batch zu simulieren. Unser Trainingscode verwendet
Gradient Accumulation mit `grad_accum_steps=2`. Versuche, auf 4 oder
8 zu erhöhen. Die effektive Batch Size steigt, ohne dass mehr
GPU-Speicher verwendet wird.

Der Sägezahn ist nicht zwangsläufig ein Problem. Selbst bei einer
guten Batch Size ist etwas Rauschen normal. Das Modell lernt die
durchschnittliche Gradientenrichtung über viele Batches hinweg.
Einzelne Batches können verrauscht sein, solange der Durchschnitt
stimmt.

## Was du dir merken solltest

Eine gute Loss-Kurve ist glatt und fallend, mit einem steilen Abfall
am Anfang und einem allmählichen Abstieg danach. Keine Spikes. Keine
Plateaus. Kein NaN.

Wenn deine Kurve nicht wie Muster 1 aussieht, stimmt etwas nicht. Die
häufigsten Fixes sind Gradient Clipping, Anpassung der Learning Rate,
Erhöhung der Batch Size und Erhöhung von Dropout. Probiere sie in
dieser Reihenfolge.

Speichere deine Loss-Werte während des Trainings immer ab. Eine
Loss-Kurve ist ein Debugging-Werkzeug. Ohne sie trainierst du blind.
Mit ihr kannst du die meisten Trainingsprobleme in Sekunden
diagnostizieren, indem du das Muster mit einem der acht oben
genannten Muster abgleichst.
