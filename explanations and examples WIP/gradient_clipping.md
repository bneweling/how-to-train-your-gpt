# Gradient Clipping: Trainingsexplosionen verhindern

## Was ist das

Gradient Clipping ist ein Sicherheitsnetz. Während des Trainings
berechnet das Modell, wie jedes Gewicht verändert werden muss, um
den Loss zu verringern. Diese Änderungsanweisungen werden Gradients
genannt. Manchmal wird ein Gradient sehr groß. Ein bestimmtes
Beispiel in den Trainingsdaten schickt eine Schockwelle durch das
Netzwerk. Die Gewichte machen einen gewaltigen Sprung, und das
Modell stürzt von der Klippe in eine Region, in der der Loss
astronomisch hoch ist. Das Training ist ruiniert.

Gradient Clipping sagt: Kein Gradient darf größer sein als eine
bestimmte Grenze. Wenn die Gesamtgröße aller Gradients zu hoch ist,
verkleinern wir sie proportional, bis sie unter diese Grenze
passen. Die Richtung des Updates bleibt dabei gleich. Nur die
Schrittgröße wird begrenzt. Das Modell macht kleine, sichere
Schritte statt wilder Sprünge.

## Wo wird es eingesetzt

Gradient Clipping wird angewendet, unmittelbar bevor der Optimizer
die Gewichte aktualisiert. Die Gradients wurden bereits berechnet.
Sie sind kurz davor, benutzt zu werden, um das Modell zu verändern.
In diesem Moment prüft Gradient Clipping sie und zügelt alle, die
zu groß geworden sind.

```python
loss.backward()  # Gradients berechnen
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # Geclippte Gradients anwenden
```

Es ist ein einziger Funktionsaufruf. Eine Codezeile, die Stunden
verschwendeter Trainingszeit sparen kann.

## Warum wir es brauchen

Sprachmodelle werden auf Text trainiert. Manche Texte sind
ungewöhnlich. Ein Satz könnte ein sehr seltenes Wort enthalten. Das
Modell hat es noch nie zuvor gesehen. Der Loss für diesen Satz ist
sehr hoch. Die Gradients, die von diesem Loss zurückfließen, sind
sehr groß. Wendet der Optimizer diese großen Gradients an, springen
die Gewichte des Modells in eine völlig andere Konfiguration. Alles,
was das Modell in den letzten tausend Schritten gelernt hat, wird
durch einen einzigen ungewöhnlichen Satz ausgelöscht.

Ohne Gradient Clipping weist die Trainings-Loss-Kurve Spitzen auf.
Lange Phasen stetiger Verbesserung werden von plötzlichen Sprüngen
unterbrochen, bei denen sich der Loss verdoppelt oder verdreifacht.
Nach jeder Spitze muss sich das Modell erholen. Manchmal erholt es
sich nie. Die Gradients waren zu groß, und die Gewichte gerieten an
einen Ort, von dem es keine Rückkehr gibt. Von diesem Punkt an
produziert das Modell nur noch Unsinn.

Mit Gradient Clipping ist die Loss-Kurve glatt. Der ungewöhnliche
Satz erzeugt zwar immer noch größere Gradients als normal, aber
diese Gradients werden auf eine sichere Größe geclippt. Das Modell
macht einen etwas größeren als normalen Schritt in die richtige
Richtung, statt eines katastrophalen Sprungs. Das Training läuft
ununterbrochen weiter.

## Wann wurde es erfunden

Gradient Clipping wird bereits seit den frühen Tagen rekurrenter
neuronaler Netze (RNNs) in den 1990er-Jahren eingesetzt. RNNs waren
notorisch anfällig für explodierende Gradients, weil sie Sequenzen
Schritt für Schritt verarbeiteten und sich die Gradients bei jedem
Schritt multiplizierten. Das Problem wurde gelöst, indem Gradients
einfach auf einen Maximalwert begrenzt wurden. Dieselbe Technik
wurde auf Transformer übertragen, obwohl Transformer nicht dasselbe
multiplikative Problem haben. Es stellt sich heraus, dass jedes
tiefe Netzwerk von Gradient Clipping als Sicherheitsmaßnahme
profitiert.

## Wie es funktioniert

Gradient Clipping nach Norm ist die Standardmethode. Statt jeden
Gradient einzeln zu clippen, messen wir die Gesamtgröße aller
Gradients zusammen und clippen sie als Gruppe. Das erhält die
relativen Größenverhältnisse der verschiedenen Gradients. Braucht
ein Parameter ein großes Update und ein anderer ein kleines, bleibt
das Verhältnis zwischen ihnen auch nach dem Clipping erhalten.

### Schritt 1: Die gesamte Gradientengröße messen

Wir berechnen die L2-Norm aller Gradients. Das ist die Quadratwurzel
aus der Summe aller quadrierten Gradients.

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2
total_norm = total_norm ** 0.5
```

Hat das Modell eine Million Parameter mit einem durchschnittlichen
Gradient von 0.01, läge die Gesamtnorm bei etwa 100. Eine
Gesamtnorm von 100 ist beherrschbar. Eine Gesamtnorm von 10000 ist
gefährlich.

### Schritt 2: Bei Bedarf clippen

Überschreitet die Gesamtnorm das erlaubte Maximum, verkleinern wir
jeden Gradient um denselben Faktor.

```python
max_norm = 1.0
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= scale
```

Lag die Gesamtnorm bei 100 und das Maximum bei 1, teilen wir jeden
Gradient durch 100. Die größten Gradients werden zu 0.01. Die
kleinsten Gradients werden noch kleiner. Die Richtung des Updates
bleibt unverändert. Nur die Schrittgröße ändert sich.

### Warum max_norm von 1.0

Der Wert 1.0 ist der Standard für das Training von Transformern. Er
wurde empirisch ermittelt. Kleinere Werte wie 0.1 machen das
Training zu langsam, weil das Modell nur winzige Schritte machen
kann. Größere Werte wie 10.0 bieten wenig Schutz, weil die meisten
Gradientennormen bereits unter 10 liegen. Ein Wert von 1.0 fängt
die gefährlichen Spitzen ab, ohne normale Trainingsschritte zu
beeinträchtigen.

## Ein kleines Codebeispiel

```python
import torch
import torch.nn as nn

# Ein kleines Modell und ein paar fiktive Gradients erstellen
model = nn.Linear(10, 10)
loss = model(torch.randn(1, 10)).sum()
loss.backward()

# Die Gradientennorm vor dem Clipping prüfen
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm before clipping: {total_norm:.4f}")

# Clippen
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Danach prüfen
total_norm_after = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5

print(f"Gradient norm after clipping:  {total_norm_after:.4f}")
print(f"Clipped: {total_norm > 1.0}")
```

## Was ohne es passiert

Sprachmodelle ohne Gradient Clipping zu trainieren bedeutet, mit
dem Feuer zu spielen. Die meisten Schritte werden unproblematisch
sein. Die Gradients werden klein sein, und das Modell wird lernen.
Aber irgendwann wird das Modell auf einen Textbatch stoßen, der
große Gradients erzeugt. Der Loss wird eine Spitze bilden. Hat das
Modell Glück, erholt es sich wieder. Hat es Pech, drängt die Spitze
die Gewichte in eine Region, in der auch jeder nachfolgende Schritt
große Gradients erzeugt. Der Loss divergiert gegen unendlich, und
der Trainingslauf ist verloren.

Gradient Clipping kostet nichts in Bezug auf die Modellqualität. Es
hat keinen Nachteil. Es ist eine reine Sicherheitsmaßnahme, die
einen seltenen, aber katastrophalen Fehlermodus verhindert. Jeder
produktive Trainingslauf nutzt es.

## Was man sich merken muss

Gradient Clipping begrenzt, wie stark sich die Gewichte des Modells
in einem einzigen Trainingsschritt ändern können. Sind Gradients zu
groß, werden sie proportional auf eine maximale Norm
herunterskaliert. Der Standardwert für das Maximum ist 1.0 beim
Training von Transformern.

Ein Funktionsaufruf. Kein Nachteil. Unendlicher Schutz vor einem
trainingszerstörenden Fehlermodus. Immer verwenden.
