# Mixed Precision Training: Geschwindigkeit ohne Kompromisse

## Was ist das

Mixed Precision Training verwendet zwei verschiedene Zahlenformate
während des Trainings. Die meisten Operationen verwenden bfloat16,
ein kompaktes Format, das weniger Speicher belegt und schneller
läuft. Einige kritische Operationen verwenden float32, das
Standardformat, das präziser ist. Diese Mischung gibt dir die
Geschwindigkeit des kompakten Formats bei der Genauigkeit des
Standardformats.

Stell es dir vor wie das Tragen von Einkäufen. Du kannst Dinge in
den Händen tragen und hast dabei volle Kontrolle über jedes
einzelne Stück. Das ist float32. Oder du packst alles in eine
Tüte und trägst die Tüte. Das ist bfloat16. Du verlierst etwas
Kontrolle über die einzelnen Gegenstände, kannst dafür aber die
doppelte Menge in einem Durchgang tragen.

Mixed Precision bedeutet, die Tüte für die schwere Arbeit zu
nutzen, aber Dinge herauszunehmen und sorgfältig zu behandeln,
wenn es auf Präzision ankommt.

## Wo wird es eingesetzt

Mixed Precision umschließt den Forward Pass des Modells. Jede
Matrixmultiplikation in den Attention- und Feed-Forward-Layern
läuft in bfloat16. Der Loss wird in float32 berechnet. Die
Gewichts-Updates werden in float32 gespeichert.

```python
with torch.amp.autocast('cuda', enabled=True):
    # Alles hier läuft in bfloat16
    logits, loss = model(input_ids, target_ids)

# Der Loss ist float32 für einen präzisen Backward Pass
loss.backward()
```

Ohne Mixed Precision Training würde ein großes Sprachmodell
doppelt so lange brauchen und doppelt so viel GPU-Speicher
belegen. Mit Mixed Precision kannst du größere Modelle auf
derselben Hardware trainieren.

## Warum wir es brauchen

Der Flaschenhals beim Training neuronaler Netze ist die
Matrixmultiplikation. Der Attention-Layer berechnet Q mal K
transponiert. Der Feed-Forward-Layer berechnet Input mal
Gewichtsmatrix. Diese Operationen dominieren die Trainingszeit.
Wenn wir sie doppelt so schnell machen können, halbieren wir die
Trainingszeit.

Kleinere Zahlenformate machen die Matrixmultiplikation schneller,
weil jede Zahl weniger Speicher belegt. Bei float32 belegt jede
Zahl vier Byte. Bei bfloat16 belegt jede Zahl zwei Byte. Du kannst
doppelt so viele Zahlen im selben Speicher unterbringen. Die GPU
kann sie in breiteren Batches verarbeiten. Das Ergebnis ist etwa
die doppelte Geschwindigkeit.

Aber warum nicht überall float16 verwenden und noch mehr
Geschwindigkeit gewinnen? Das Problem ist der Wertebereich.
Float16 kann nur Zahlen bis etwa fünfundsechzigtausend darstellen.
Während des Trainings können Zwischenwerte dieses Limit
überschreiten. Die Zahl läuft über und wird zu unendlich. Das
Modell produziert Unsinn. Deshalb sind frühe Versuche mit Half
Precision Training gescheitert.

Bfloat16 löst dieses Problem, indem es denselben Wertebereich wie
float32 beibehält. Der maximal mögliche Wert liegt für beide
Formate bei etwa drei Komma vier mal zehn hoch achtunddreißig.
Bfloat16 kann nicht überlaufen. Es hat innerhalb dieses Bereichs
nur weniger Präzision. Für das Training neuronaler Netze ist
dieser Kompromiss perfekt. Wir brauchen den Wertebereich für
Zwischenwerte, aber wir brauchen nicht sieben Dezimalstellen
Präzision für jede Aktivierung.

## Wann wurde es erfunden

Bfloat16 wurde 2017 von Google speziell für ihre TPU-Hardware
entwickelt. Es wurde von Grund auf für das Training neuronaler
Netze konzipiert. NVIDIA übernahm es 2020 in ihren A100-GPUs.
Heute unterstützt jede größere GPU bfloat16 nativ. Mixed Precision
Training mit bfloat16 ist der Standard für das gesamte
produktive Training von Sprachmodellen.

## Die drei Zahlenformate im Vergleich

```
Float32:    32 Bit gesamt
            1 Bit für Vorzeichen
            8 Bit für Exponent (Wertebereich)
            23 Bit für Mantisse (Präzision)
            Wertebereich: bis zu ±3,4 × 10³⁸
            Präzision: 7 Dezimalstellen

Bfloat16:   16 Bit gesamt
            1 Bit für Vorzeichen
            8 Bit für Exponent (Wertebereich)
            7 Bit für Mantisse (Präzision)
            Wertebereich: bis zu ±3,4 × 10³⁸ (wie float32!)
            Präzision: 2 Dezimalstellen

Float16:    16 Bit gesamt
            1 Bit für Vorzeichen
            5 Bit für Exponent (Wertebereich)
            10 Bit für Mantisse (Präzision)
            Wertebereich: bis zu ±65504 (kann überlaufen!)
            Präzision: 3 Dezimalstellen
```

Beachte, dass bfloat16 und float32 denselben Wertebereich haben.
Der einzige Unterschied ist die Präzision. Bfloat16 hat sieben
Bit für die Mantisse, während float32 dreiundzwanzig hat. Das
bedeutet, bfloat16 kann Zahlen mit etwa zwei Dezimalstellen
Genauigkeit darstellen. Float32 kann sieben darstellen. Für das
Training neuronaler Netze reichen zwei Stellen aus. Die
Gradienten und Aktivierungen brauchen keine extreme Präzision.
Sie brauchen nur konsistente Größenordnungen.

Float16 hat eine bessere Präzision als bfloat16, aber einen
miserablen Wertebereich. In der Praxis läuft float16 bei langen
Trainingsläufen über. Bfloat16 nicht. Deshalb hat sich bfloat16
durchgesetzt.

## Wie es in der Praxis funktioniert

Die Trainingsschleife besteht aus drei Teilen, und jeder nutzt
eine andere Precision-Strategie.

### Teil 1: der Forward Pass

Die meisten Operationen laufen in bfloat16 innerhalb eines
Autocast-Kontexts.

```python
with torch.amp.autocast('cuda', enabled=True):
    logits, loss = model(input_ids, target_ids)
```

Der Autocast-Kontextmanager konvertiert Operationen automatisch
zu bfloat16, wo immer es sicher ist. Matrixmultiplikationen und
Convolutions werden konvertiert. Normalisierungsoperationen wie
RMSNorm bleiben in float32, weil sie mehr Präzision brauchen. Der
Entwickler muss nicht manuell festlegen, welche Operationen
konvertiert werden. Autocast übernimmt das.

### Teil 2: der Backward Pass

Der Loss liegt für die Genauigkeit in float32 vor. Der Backward
Pass berechnet die Gradienten. Manche Gradienten liegen in
float32 vor, manche in bfloat16. Ein Gradient Scaler achtet auf
Underflow.

Der Gradient Scaler multipliziert den Loss vor dem Backward Pass
mit einer großen Zahl. Das schiebt kleine Gradienten in den
darstellbaren Bereich von bfloat16. Nach dem Backward Pass
dividiert der Scaler die Gradienten vor dem Weight Update wieder
herunter. Das verhindert, dass winzige Gradienten in bfloat16 zu
null werden.

```python
scaler = torch.amp.GradScaler('cuda', enabled=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Teil 3: das Weight Update

Die Master-Gewichte werden immer in float32 gespeichert. Nach dem
Backward Pass werden die Gradienten auf die float32-Gewichte
angewendet. Das stellt sicher, dass die Gewichte selbst über
viele Updates hinweg nie an Präzision verlieren, auch wenn der
Forward Pass in bfloat16 rechnet. Man nennt das Mixed Precision,
weil der Forward Pass mit halber Precision arbeitet und die
Gewichtsspeicherung mit voller Precision.

```python
# Master-Gewichte sind immer float32
optimizer.step()  # Wendet float32-Gradienten auf float32-Gewichte an
```

## Ein kleines Codebeispiel

```python
import torch

# Prüfen, ob deine GPU bfloat16 unterstützt
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    print(f"GPU compute capability: {capability}")
    bf16_ok = capability[0] >= 8
    print(f"Bfloat16 supported: {bf16_ok}")
else:
    print("No GPU available. Mixed precision requires CUDA.")
    print("CPU training runs in float32 only.")

# Einfacher Geschwindigkeitsvergleich
if torch.cuda.is_available():
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # Float32
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c32 = a @ b
    end.record()
    torch.cuda.synchronize()
    t32 = start.elapsed_time(end)

    # Bfloat16
    start.record()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        c16 = a @ b
    end.record()
    torch.cuda.synchronize()
    t16 = start.elapsed_time(end)

    print(f"\nMatrix multiply {size}x{size}:")
    print(f"Float32 time:  {t32:.1f} ms")
    print(f"Bfloat16 time: {t16:.1f} ms")
    print(f"Speedup:       {t32/t16:.1f}x")
```

## Wann Mixed Precision nicht verwendet werden sollte

Manche Operationen verschlechtern sich bei reduzierter Precision.
Normalisierungslayer wie RMSNorm sollten in float32 bleiben. Der
Softmax in der Attention sollte für die Stabilität in float32
laufen. Die Loss-Berechnung muss für präzise Gradienten in
float32 erfolgen. Autocast übernimmt das meiste davon automatisch.

Wenn du auf der CPU trainierst, bringt Mixed Precision keinen
Vorteil. Die CPU hat keine native bfloat16-Unterstützung. Die
Konvertierungen würden nur Overhead ohne Geschwindigkeitsgewinn
hinzufügen. Unser Code prüft die CUDA-Verfügbarkeit und aktiviert
Mixed Precision nur auf der GPU.

## Was du dir merken solltest

Mixed Precision Training führt die meisten Operationen zugunsten
der Geschwindigkeit in bfloat16 aus, während kritische Werte für
die Genauigkeit in float32 gehalten werden. Das bfloat16-Format
hat denselben Wertebereich wie float32, sodass es nie überläuft.
Es hat weniger Präzision, aber neuronale Netze brauchen nicht für
jede Zahl sieben Dezimalstellen.

Das Ergebnis ist etwa die doppelte Geschwindigkeit und der halbe
Speicherverbrauch, ohne messbaren Qualitätsverlust des Modells.
Jedes produktive Sprachmodell wird mit Mixed Precision trainiert.
Das ist keine optionale Optimierung. Es ist die Standardmethode
zu trainieren.
