# Vollständiges Fine-Tuning

## Kurz gesagt

Vollständiges Fine-Tuning aktualisiert jedes Gewicht im Modell. Nichts wird
eingefroren. Jeder Parameter wird auf die neue Aufgabe trainiert. Das ist
der mächtigste Ansatz, erfordert aber auch die meiste Hardware. Er ist
außerdem der teuerste und am wenigsten flexible Ansatz, weil du für jede
Aufgabe, für die du ein Fine-Tuning durchführst, eine vollständige Kopie
des Modells speichern musst.

## Wo es einzuordnen ist

Vollständiges Fine-Tuning nimmt den vortrainierten Checkpoint und setzt
das Training fort. Der einzige Unterschied zum Pretraining sind die Daten.
Statt rohem Internettext sieht das Modell Instruction-Response-Paare. Die
Trainingsschleife ist identisch. Forward Pass. Loss berechnen. Backward
Pass. Gewichte aktualisieren. Derselbe Code, der das Basismodell trainiert
hat, trainiert auch das fine-tuned Modell.

## Warum du es vielleicht brauchst

LoRA funktioniert für die meisten Aufgaben gut. Aber manche Aufgaben
erfordern eine Veränderung des grundlegenden Verhaltens des Modells auf
eine Weise, die Low-Rank-Updates nicht erfassen können. Einem Modell eine
neue Sprache beizubringen, die nicht in seinen Pretraining-Daten enthalten
war. Das Ausgabeformat oder den Stil radikal zu verändern. Systematische
Verzerrungen oder Fehler im Basismodell zu korrigieren. Diese Fälle können
vollständiges Fine-Tuning rechtfertigen.

Vollständiges Fine-Tuning liefert außerdem die qualitativ besten
Ergebnisse, gemessen an Standard-Benchmarks. Der Abstand zwischen
vollständigem Fine-Tuning und LoRA wird kleiner, je besser die
Basismodelle werden, aber bei den anspruchsvollsten Anwendungen liegt
vollständiges Fine-Tuning noch immer knapp vorn.

## Warum du es wahrscheinlich nicht brauchst

Die meisten Anwendungsfälle aus der Praxis brauchen kein vollständiges
Fine-Tuning. Instruction Following. Chat-Verhalten. Zusammenfassung.
Klassifikation. Übersetzung. All das funktioniert gut mit LoRA. Die
zusätzlichen Kosten des vollständigen Fine-Tunings rechtfertigen selten
die geringfügige Qualitätsverbesserung.

Vollständiges Fine-Tuning bringt außerdem operative Komplexität mit sich.
Jede fine-tuned Variante ist ein eigener, mehrere Gigabyte großer
Checkpoint. Das Speichern, Bereitstellen und Versionieren dieser
Checkpoints ist teuer. Mit LoRA speicherst du ein Basismodell und viele
kleine Adapter-Dateien.

## Die Hardware-Anforderung

Vollständiges Fine-Tuning benötigt genug GPU-Speicher, um die
Modellgewichte, die Optimizer-States und die Aktivierungen für einen
Trainings-Batch zu speichern. Die Optimizer-States verursachen die
größten Kosten, weil AdamW zwei Werte pro Parameter speichert.

```
Speicherbedarf für ein Modell mit 7 Milliarden Parametern beim
vollständigen Fine-Tuning:

Modellgewichte (bfloat16):        14 GB  (7B × 2 Byte)
Optimizer-States (float32):       56 GB  (7B × 2 × 4 Byte)
Gradienten (float32):             28 GB  (7B × 4 Byte)
Aktivierungen (bfloat16, Batch=1): ~2 GB (variiert je nach Sequenzlänge)
-------------------------------------------------
Gesamt:                          ~100 GB
```

Das erfordert mehrere GPUs. Eine RTX 4090 hat 24 Gigabyte. Du bräuchtest
vier oder fünf davon, nur um das Modell und den Optimizer zu laden.
Deshalb findet vollständiges Fine-Tuning in Rechenzentren statt und nicht
auf dem Desktop-Rechner.

## Der Ablauf

Vollständiges Fine-Tuning folgt denselben Schritten wie das Pretraining,
nur mit anderen Hyperparametern.

```
Learning rate:   1e-5 bis 5e-5  (deutlich kleiner als beim Pretraining)
Warmup steps:    100 bis 500     (weniger Warmup nötig)
Total steps:     1 bis 5 Epochen  (weniger Epochen, da weniger Daten)
Batch size:      So groß wie es in den Speicher passt
Weight decay:    0.1            (wie beim Pretraining)
```

Die Learning Rate ist zehn- bis dreißigmal kleiner als beim Pretraining.
Das Modell ist bereits nah an einer guten Lösung. Große Schritte würden
das vortrainierte Wissen zerstören. Kleine Schritte verfeinern das
Verhalten, ohne das Fundament zu erschüttern.

## Katastrophales Vergessen

Vollständiges Fine-Tuning birgt ein Risiko, das LoRA vermeidet. Wenn sich
jedes Gewicht verändern kann, kann das Modell seine vortrainierten
Fähigkeiten verlernen. Ein Modell, das ausschließlich auf medizinische
Fragen fine-tuned wurde, könnte die Fähigkeit verlieren, über Sport,
Politik oder Geschichte zu sprechen. Die Gewichte driften von der
vortrainierten Konfiguration weg.

Es gibt Gegenmaßnahmen. Allgemeine Daten in den Fine-Tuning-Datensatz
mischen. Eine sehr kleine Learning Rate verwenden. Das Training stoppen,
sobald sich der Validierungs-Loss nicht mehr verbessert. Aber selbst mit
Gegenmaßnahmen ist ein gewisses Vergessen unvermeidlich. Das Modell
tauscht einen Teil seines breiten Wissens gegen tiefes Wissen über die
Fine-Tuning-Aufgabe ein.

LoRA vermeidet das, indem es die Basisgewichte einfriert. Das
ursprüngliche Wissen bleibt exakt erhalten, weil sich die ursprünglichen
Gewichte nie ändern. Nur die Adapter lernen die neue Aufgabe. Das ist ein
grundlegender architektonischer Vorteil von LoRA gegenüber vollständigem
Fine-Tuning für die meisten Anwendungen.

## Wann es sich lohnt

Vollständiges Fine-Tuning ergibt in einigen wenigen konkreten Szenarien
Sinn. Wenn du ein Foundation Model baust, das andere fine-tunen werden.
Wenn die Aufgabe eine Änderung des Wissens des Modells erfordert und
nicht nur seines Verhaltens. Wenn sich die geringfügige
Qualitätsverbesserung durch vollständiges Fine-Tuning in echtem
geschäftlichem Nutzen niederschlägt. Wenn du das Hardware-Budget und die
operative Infrastruktur hast, um große Modelldateien zu verwalten.

Für alle anderen ist LoRA oder QLoRA die richtige Wahl. Weniger Hardware.
Weniger Speicherplatz. Weniger Zeit. Nahezu die gleiche Qualität. Der
Abstand wird immer kleiner, je besser die Basismodelle werden und je
besser Adapter darin werden, aufgabenspezifisches Verhalten zu erfassen.

## Was du dir merken solltest

Vollständiges Fine-Tuning aktualisiert jedes Gewicht im Modell anhand
aufgabenspezifischer Daten. Es erfordert teure Hardware, weil die
Optimizer-States das Vierfache des Speicherbedarfs des Modells selbst
verbrauchen. Es birgt das Risiko von katastrophalem Vergessen, bei dem
das Modell seine allgemeinen Fähigkeiten verliert. Die
Qualitätsverbesserung gegenüber LoRA ist bei den meisten Aufgaben
geringfügig. Vollständiges Fine-Tuning ist nur dann das richtige
Werkzeug, wenn du Foundation Models baust oder unabhängig von den Kosten
die absolut maximale Performance benötigst.

## Eine Trainingsschleife für vollständiges Fine-Tuning

So sieht eine Trainingsschleife für vollständiges Fine-Tuning aus.
Vergleiche sie mit dem LoRA-Notebook. Die einzigen Unterschiede sind,
dass jeder Parameter trainierbar ist, nicht nur ein paar Adapter, und
dass die Learning Rate kleiner ist.

```python
import torch
from torch.utils.data import DataLoader

model = load_pretrained_model()
model.train()

# Beim vollständigen Fine-Tuning sind alle Parameter trainierbar
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,           # Kleinere LR als bei LoRA
    weight_decay=0.1,
    betas=(0.9, 0.95),
)

# Trainingsschleife identisch zum Pretraining
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, target_ids = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

# Den vollständigen Modell-Checkpoint speichern
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "fine_tuned_model.pt")
```

Achte auf die Learning Rate. Zwei e minus fünf statt drei e minus vier
wie beim Pretraining. Das Modell ist nah an einer guten Lösung. Große
Schritte würden sein vortrainiertes Wissen zerstören. Kleine Schritte
verfeinern sein Verhalten.

Achte außerdem darauf, dass jeder Parameter trainierbar ist. Der
Optimizer speichert zwei States pro Parameter. Bei einem Modell mit
sieben Milliarden Parametern sind das allein für den Optimizer
sechsundfünfzig Gigabyte. Deshalb braucht vollständiges Fine-Tuning
mehrere GPUs.

Vergleiche das mit dem LoRA-Notebook, in dem nur 0,19 Prozent der
Parameter trainierbar sind. Die Trainingsschleife ist dieselbe. Der
Optimizer ist derselbe. Der einzige Unterschied ist, wie viele Parameter
aktualisiert werden. LoRA aktualisiert wenige. Vollständiges Fine-Tuning
aktualisiert alle.
