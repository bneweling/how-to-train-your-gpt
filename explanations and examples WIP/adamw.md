# AdamW: Der Optimizer, der Sprachmodelle trainiert

## Was ist das

AdamW ist der Algorithmus, der die Gewichte des Modells während
des Trainings aktualisiert. Nachdem berechnet wurde, wie falsch
eine Vorhersage war, und die Richtung bestimmt wurde, in die jedes
Gewicht bewegt werden soll, entscheidet AdamW genau, wie weit es
sich bewegen soll. Das macht es intelligent, basierend auf der
Historie vergangener Gradienten für jeden Parameter.

Stellen Sie es sich vor wie eine Wanderung einen Berg hinab im
Nebel. Sie können den Talboden nicht sehen. Sie können nur
fühlen, in welche Richtung es bergab geht. Sie machen einen
Schritt. Dann fühlen Sie erneut. Ein naiver Wanderer macht immer
Schritte gleicher Größe. Aber manche Teile des Berges sind steil
und brauchen große Schritte. Andere sind flach und brauchen
kleine Schritte. AdamW merkt sich, wie steil jeder Parameter war,
und passt die Schrittgröße entsprechend an. Es merkt sich außerdem
die allgemeine Richtung, um das Momentum aufrechtzuerhalten.

Das W in AdamW steht für entkoppelten Weight Decay (decoupled
weight decay). Dies ist die entscheidende Neuerung gegenüber dem
ursprünglichen Adam-Optimizer. Weight Decay drängt alle Gewichte
langsam in Richtung null, um zu verhindern, dass sie zu groß
werden. Bei AdamW ist dieser Effekt von der Gradientenberechnung
getrennt. Diese Trennung sorgt dafür, dass Weight Decay korrekt
funktioniert.

## Wo wird es eingesetzt

AdamW wird bei jedem Trainingsschritt nach der Backpropagation
aufgerufen. Es nimmt die berechneten und geclippten Gradienten
und wendet sie auf die Gewichte des Modells an.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # AdamW aktualisiert hier die Gewichte
optimizer.zero_grad()
```

## Warum wir es anstelle von einfachem Gradientenabstieg verwenden

Einfacher Gradientenabstieg ist simpel. Jedes Gewicht wird in die
Richtung bewegt, die den Loss verringert. Die Schrittgröße ist
für jedes Gewicht gleich.

```
weight = weight - learning_rate × gradient
```

Das hat drei Probleme.

Erstens ist die Schrittgröße fest. Ein Gewicht, das eine große
Änderung braucht, bekommt denselben Schritt wie ein Gewicht, das
nur eine winzige Änderung braucht. Die Learning Rate muss für die
empfindlichsten Gewichte gewählt werden. Das macht das Training
für alle anderen Gewichte langsam.

Zweitens gibt es kein Momentum. Wenn der Gradient verrauscht ist
und bei jedem Schritt in eine andere Richtung zeigt, zickzackt der
Optimizer hin und her und macht nur langsam Fortschritte. Momentum
glättet das Rauschen, indem es die Richtung vorheriger Schritte
einbezieht.

Drittens gibt es keinen Weight Decay. Ohne Regularisierung können
Gewichte beliebig groß werden. Große Gewichte bedeuten, dass das
Modell bei manchen Mustern übermäßig selbstsicher ist und andere
ignoriert. Das Modell overfittet.

AdamW löst alle drei Probleme.

## Wann wurde es erfunden

Adam wurde 2014 von Diederik Kingma und Jimmy Ba veröffentlicht.
Er wurde schnell zum Standard-Optimizer für Deep Learning. Doch
Forscher bemerkten, dass die Weight-Decay-Implementierung in Adam
mit den adaptiven Learning Rates verflochten war. Das bedeutete,
dass Weight Decay große Gewichte nicht tatsächlich verhinderte.
Es verlangsamte meist nur das Training.

AdamW wurde 2017 von Ilya Loshchilov und Frank Hutter
vorgeschlagen. Sie zeigten, dass die Entkopplung von Weight Decay
und adaptiven Learning Rates das Problem behob. AdamW erzielte mit
denselben Hyperparametern eine bessere Generalisierung als Adam.
Die Korrektur war einfach, aber die Wirkung war groß. GPT-3 wurde
mit AdamW trainiert. LLaMA wurde mit AdamW trainiert. Jedes
moderne Sprachmodell verwendet AdamW.

## Wie es funktioniert

AdamW führt für jeden Parameter zwei laufende Durchschnitte. Der
erste ist das Momentum, das die durchschnittliche Richtung der
letzten Gradienten verfolgt. Der zweite ist die Velocity, die die
durchschnittliche Größenordnung der letzten Gradienten verfolgt.

### Schritt 1: den verrauschten Gradienten berechnen

```python
gradient = compute_gradient(loss, weight)
```

Das ist das rohe Signal aus einem Batch an Daten. Es ist
verrauscht. Ein einzelner Batch kann in eine irreführende Richtung
weisen.

### Schritt 2: das Momentum aktualisieren

```python
momentum = beta1 × momentum + (1 - beta1) × gradient
```

Das Momentum ist ein gewichteter Durchschnitt vergangener
Gradienten. Beta1 liegt üblicherweise bei 0,9. Das bedeutet, dass
aktuelle Gradienten zu neunzig Prozent zählen und ältere
Gradienten verblassen. Das Momentum glättet das Rauschen und
liefert eine stabile Richtung.

### Schritt 3: die Velocity aktualisieren

```python
velocity = beta2 × velocity + (1 - beta2) × gradient²
```

Die Velocity verfolgt, wie stark sich jeder Parameter bewegt hat.
Beta2 liegt üblicherweise bei 0,95. Parameter, die große
Bewegungen gemacht haben, bekommen eine hohe Velocity. Parameter,
die sich kaum bewegt haben, bekommen eine niedrige Velocity.

### Schritt 4: Bias-Korrektur

Sowohl Momentum als auch Velocity starten bei null. In den ersten
Schritten sind sie in Richtung null verzerrt. Die Bias-Korrektur
behebt das.

```python
momentum_corrected = momentum / (1 - beta1^t)
velocity_corrected = velocity / (1 - beta2^t)
```

Dabei ist t die aktuelle Schrittnummer. Nach vielen Schritten wird
die Korrektur vernachlässigbar. Aber in den ersten Schritten
verhindert sie, dass der Optimizer winzige, nutzlose Schritte
macht.

### Schritt 5: entkoppelter Weight Decay

```python
weight = weight × (1 - learning_rate × weight_decay)
```

Das schrumpft jedes Gewicht um einen winzigen Bruchteil. Weight
Decay liegt üblicherweise bei 0,1. Bei einer Learning Rate von
0,0003 wird jedes Gewicht pro Schritt mit 0,99997 multipliziert.
Über Tausende von Schritten hinweg drängt dies die Gewichte sanft
in Richtung null. Nur Gewichte, die kontinuierlich starke
Gradienten erhalten, überleben. Gewichte, die nicht nützlich sind,
verschwinden allmählich.

Beachten Sie, dass dieser Schritt vor dem Gradienten-Update
erfolgt und vollständig unabhängig vom Gradienten ist. Das ist
der entkoppelte Teil von AdamW. Im ursprünglichen Adam war Weight
Decay mit der Gradientenskalierung vermischt, was es unwirksam
machte.

### Schritt 6: den Gradienten anwenden

```python
weight = weight - learning_rate × momentum_corrected / (sqrt(velocity_corrected) + eps)
```

Der Gradientenschritt wird mit der Learning Rate skaliert.
Anschließend wird er durch die Quadratwurzel der Velocity geteilt.
Parameter mit hoher Velocity haben sich stark verändert, daher
machen wir kleinere Schritte. Parameter mit niedriger Velocity
waren stabil, daher können wir größere Schritte machen. Das
Epsilon verhindert eine Division durch null.

## Zwei Parametergruppen

Nicht alle Parameter sollten Weight Decay bekommen. Die Biases
und Normalisierungsgewichte sind eindimensional. Sie passen den
Offset und die Skalierung von Aktivierungen an. Sie in Richtung
null zu drängen würde sie daran hindern, ihre Aufgabe zu
erfüllen. Wir erstellen zwei Gruppen von Parametern mit
unterschiedlichen Weight-Decay-Werten.

```python
def create_optimizer(model, config):
    decay_params = []      # Linear- und Embedding-Gewichte
    no_decay_params = []   # Biases und Normalisierungsgewichte

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or 'norm' in name.lower() or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

Die Decay-Gruppe bekommt einen Weight Decay von 0,1. Die
No-Decay-Gruppe bekommt einen Weight Decay von null. Jede Gruppe
wird vom Optimizer separat behandelt.

## Die Hyperparameter

Jeder Optimizer hat Einstellungen, die Hyperparameter genannt
werden. Bei AdamW sind die wichtigsten:

```
learning_rate = 0.0003  (3e-4)
  Wie groß der Schritt ist, der gemacht wird. Kleiner ist
  sicherer, aber langsamer.

betas = (0.9, 0.95)
  Wie stark vergangenen Gradienten vertraut wird. Höher bedeutet
  glattere Updates.

weight_decay = 0.1
  Wie aggressiv Gewichte in Richtung null gedrängt werden. Höher
  verhindert Overfitting, aber zu hoch lässt das Modell vergessen.

eps = 0.00000001 (1e-8)
  Eine winzige Zahl, um eine Division durch null zu verhindern.
  Muss nie angepasst werden.
```

Diese Werte sind die LLaMA-Standardwerte und wurden an Modellen
von einer Milliarde bis siebzig Milliarden Parametern in der
Praxis erprobt. Wenn Sie nichts Ungewöhnliches vorhaben, gibt es
selten einen Grund, sie zu ändern.

## Was Sie sich merken sollten

AdamW ist der Standard-Optimizer für das Training von
Sprachmodellen. Es kombiniert Momentum für Stabilität, adaptive
Learning Rates für Effizienz und entkoppelten Weight Decay für
Regularisierung. Die drei Mechanismen wirken zusammen, um das
Training schnell, stabil und widerstandsfähig gegen Overfitting
zu machen.

Jedes produktive Sprachmodell wird mit AdamW trainiert. Die
Hyperparameter sind gut etabliert und müssen selten angepasst
werden. Wie Gradient Clipping hat es keinen nennenswerten
Nachteil. Es ist schlicht das richtige Werkzeug für die Aufgabe.
