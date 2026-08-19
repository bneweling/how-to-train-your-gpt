# Cosine Warmup: Der Learning-Rate-Schedule

## Was ist das

Der Cosine-Warmup-Schedule steuert, wie sich die Learning Rate
während des Trainings verändert. Sie beginnt niedrig und steigt
bis zu einem Höchstwert an. Anschließend fällt sie allmählich
entlang einer Kosinuskurve ab. Am Ende des Trainings ist die
Learning Rate sehr klein, und das Modell pendelt sich in einem
feinen Minimum ein.

Man kann es sich wie das Fahrradfahrenlernen vorstellen. Zu
Beginn fährt man sehr langsam. Man schwankt. Man bekommt ein
Gefühl für das Gleichgewicht. Sobald man etwas stabiler ist,
tritt man kräftiger in die Pedale und wird schneller. Kurz vor
dem Ziel wird man wieder langsamer, um präzise anzuhalten. Man
sprintet nicht von Anfang an los und tritt am Ende abrupt auf
die Bremse.

Cosine Warmup macht beim Training neuronaler Netze genau
dasselbe. Das Modell startet langsam, um sein Gleichgewicht zu
finden. Sobald es stabil ist, beschleunigt es auf volle
Geschwindigkeit. Am Ende verlangsamt es wieder, um sanft bei der
bestmöglichen Lösung zu landen.

## Wo wird es eingesetzt

Der Schedule steuert den Learning-Rate-Parameter innerhalb des
Optimizers. Bei jedem Trainingsschritt berechnet der Schedule
eine neue Learning Rate und weist sie dem Optimizer zu.

```python
scheduler = CosineWarmupScheduler(optimizer, warmup=2000, max_steps=100000)

for step in range(max_steps):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Learning Rate bei jedem Schritt aktualisieren
    optimizer.zero_grad()
```

## Warum wir es brauchen

Eine konstante Learning Rate erscheint einfacher. Warum nicht
einfach einen Wert wählen und den gesamten Weg damit trainieren?
Zwei Gründe.

Erstens: Das frühe Training ist chaotisch. Die Gewichte des
Modells sind zufällig. Die Gradienten sind groß und verrauscht.
Große Learning Rates zu Beginn können das Modell in zufällige
Richtungen davonschießen lassen. Die Warmup-Phase gibt dem
Modell die Möglichkeit, festen Boden unter die Füße zu bekommen,
bevor es große Schritte macht.

Zweitens: Beim späten Training geht es um Präzision. Nach
Tausenden von Schritten ist das Modell nahe an einer guten
Lösung. Große Schritte würden über das Minimum hinausschießen
und ewig darum herumspringen. Die Decay-Phase erlaubt es dem
Modell, winzige, vorsichtige Schritte zu machen, um sich genau
auf der besten Position einzupendeln.

Eine konstante Learning Rate wäre entweder zu groß für den Start
oder zu klein für die Mitte. Nur die Kombination aus Warmup und
Decay ermöglicht sowohl Stabilität am Anfang als auch Präzision
am Ende.

## Wann wurde es erfunden

Learning-Rate-Warmup wurde bereits 2017 beim ursprünglichen
Transformer eingesetzt. Die Autoren stellten fest, dass das
Training in den ersten paar Tausend Schritten ohne Warmup
instabil war. Cosine Decay wurde etwa zur gleichen Zeit
eingeführt, als Alternative zu Step-Decay-Schedules, die die
Learning Rate abrupt zu vorher festgelegten Zeitpunkten senken.
Step Decay funktioniert, aber die plötzlichen Sprünge können das
Modell stören. Cosine Decay verläuft dagegen weich und
kontinuierlich. GPT-3 verwendete Cosine Warmup. LLaMA verwendete
Cosine Warmup. Es ist der Standard für das Training von
Sprachmodellen.

## Wie es funktioniert

Der Schedule besteht aus drei Phasen. Jede Phase ist eine
einfache mathematische Formel.

### Phase 1: Linearer Warmup

Die Learning Rate startet bei null und steigt linear bis zum
Maximalwert an.

```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
```

Beispiel mit warmup_steps = 2000 und max_lr = 0.0003:

```
Step 0:    lr = 0.0003 × (0 / 2000) = 0.0
Step 500:  lr = 0.0003 × (500 / 2000) = 0.000075
Step 1000: lr = 0.0003 × (1000 / 2000) = 0.00015
Step 2000: lr = 0.0003 × (2000 / 2000) = 0.0003
```

Bei jedem Schritt wächst die Learning Rate um denselben winzigen
Betrag. Keine plötzlichen Sprünge. Das Modell hat zweitausend
Schritte Zeit, um stabil zu werden, bevor es die volle
Geschwindigkeit erreicht.

### Phase 2: Cosine Decay

Nach dem Warmup folgt die Learning Rate einer Kosinuskurve, die
vom Maximum bis zu einem Minimum abfällt.

```python
if step < max_steps:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 × (1 + cos(π × progress))
    lr = min_lr + (max_lr - min_lr) × cosine_decay
```

Die Variable progress läuft über die verbleibenden Schritte von
null bis eins. Die Kosinusfunktion erzeugt eine weiche S-förmige
Kurve.

```
Step 2000:  progress = 0.0, cosine = 1.0, lr = 0.0003
Step 25000: progress = 0.23, cosine = 0.75, lr = 0.000225
Step 50000: progress = 0.49, cosine = 0.25, lr = 0.000075
Step 100000: progress = 1.0, cosine = 0.0, lr = 0.00001
```

Die Learning Rate fällt zunächst langsam, dann schneller in der
Mitte und gegen Ende wieder langsam. Das Minimum liegt
üblicherweise bei 0.00001, was dreißigmal kleiner ist als der
Höchstwert. Diese winzige Rate am Ende erlaubt es dem Modell,
seine Gewichte mit extremer Präzision zu verfeinern.

### Phase 3: Minimum

Nach max_steps bleibt die Learning Rate für immer beim Minimum.

```python
lr = min_lr
```

Das Modell lernt weiterhin, aber im Schneckentempo. Jeder
Schritt macht kaum noch einen Unterschied. Das ist beabsichtigt.
Das Modell hat bereits alles gelernt, was es braucht. Die
verbleibenden Schritte dienen nur noch dem Feinschliff.

## Ein kleines Codebeispiel

```python
import math
import matplotlib.pyplot as plt

max_lr = 0.0003
min_lr = 0.00001
warmup_steps = 2000
max_steps = 100000

lrs = []
for step in range(max_steps):
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    elif step < max_steps:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (max_lr - min_lr) * cosine
    else:
        lr = min_lr
    lrs.append(lr)

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Warmup Schedule')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lrs_curve.png', dpi=100)
print("The curve rises from 0 over 2000 warmup steps")
print("then decays along a cosine curve for 98000 steps")
print("and stays at the minimum from step 100000 onward")
```

## Die Form der Kurve

```
Learning Rate
     ^
     |
0.0003 +         ....----....
     |        ..              ..
     |      ..                  ..
     |    ..                      ....
     |  ..                            .............
     |..                                            ...........
0.0  +----+----+----+----+----+----+----+----+----+----+---->
     0   10k  20k  30k  40k  50k  60k  70k  80k  90k  100k
                             Trainingsschritte
```

Die Kurve steigt während des Warmups steil an. Sie bleibt eine
Weile nahe am Höchstwert. Dann beginnt ein sanfter Abstieg, der
sich in der Mitte beschleunigt und gegen Ende abflacht. Das
Minimum wird exakt beim letzten Trainingsschritt erreicht. Nicht
früher. Nicht später.

## Was man sich merken sollte

Cosine-Warmup-Scheduling steuert die Learning Rate über den
gesamten Trainingslauf hinweg. Die Rate startet bei null und
wärmt sich bis zu einem Höchstwert auf. Anschließend fällt sie
entlang einer Kosinuskurve bis zu einem Minimum ab. Der Schedule
verläuft weich und kontinuierlich, ohne plötzliche Einbrüche.

Warmup verhindert Instabilität zu Beginn des Trainings, wenn die
Gradienten chaotisch sind. Decay ermöglicht Präzision am Ende,
wenn das Modell nahe an der Lösung ist. Zusammen machen sie das
Training sowohl stabil als auch präzise. Jedes moderne
Sprachmodell verwendet diesen Schedule.
