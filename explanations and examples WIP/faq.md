# FAQ und Fehlerbehebung

## Training

### Q: Mein Loss bleibt nach Tausenden von Schritten bei 10.8. Was ist falsch?

10.8 ist der Loss eines zufällig initialisierten Modells, das
gleichverteilt über das Vokabular vorhersagt. Er entspricht ln(50257).
Wenn dein Loss bei 10.8 bleibt, lernt dein Modell nicht. Mögliche
Ursachen.

Die Learning Rate ist zu niedrig, um die Weights zu bewegen. Versuche
testweise, sie von 3e-4 auf 1e-3 zu erhöhen, und beobachte, ob sich der
Loss bewegt.

Der Optimizer führt keinen Schritt aus. Prüfe, dass `optimizer.step()`
aufgerufen wird und dass `optimizer.zero_grad()` danach aufgerufen
wird.

Die Gradients sind null. Das deutet auf einen Bug in der
Loss-Berechnung oder im Backward Pass hin. Prüfe, dass
`loss.backward()` aufgerufen wird und dass Gradients fließen. Gib
`model.layers[0].attention.qkv_proj.weight.grad` nach dem Backward
Pass aus. Er sollte ungleich null sein.

Die Daten sind fehlerhaft. Vielleicht sind Input und Target identisch,
oder die Targets bestehen alle aus demselben Token. Gib ein paar
Samples aus.

### Q: Mein Loss ist NaN. Was ist passiert?

NaN steht für „not a number“ (keine Zahl). Es bedeutet, dass eine Zahl
übergelaufen ist oder durch null geteilt wurde. Die Ursache ist fast
immer eine zu hohe Learning Rate oder fehlendes Gradient Clipping.

Lösung: Senke die Learning Rate um den Faktor 10. Füge Gradient
Clipping mit max_norm 1.0 hinzu. Prüfe, ob deine Loss-Funktion korrekt
rechnet. Gib die Logits vor der Loss-Berechnung aus. Enthalten sie
NaN, liegt das Problem im Forward Pass des Modells. Sind sie
unauffällig, liegt das Problem in der Loss-Berechnung.

### Q: Mein Loss sinkt eine Weile und schnellt dann plötzlich hoch

Das ist eine Gradient Explosion. Ein seltener Batch mit ungewöhnlichen
Daten verursacht sehr große Gradients, die die Gewichte des Modells in
einen Bereich schießen, in dem der Loss riesig ist.

Lösung: Füge Gradient Clipping hinzu.
`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
Dies sollte nach `loss.backward()` und vor `optimizer.step()`
aufgerufen werden.

### Q: Das Training ist auf der CPU extrem langsam. Was kann ich tun?

Training auf der CPU ist 10x bis 50x langsamer als auf der GPU.
Optionen:

Verwende die Tiny-Konfiguration (d_model=256, 4 Layer). Sie trainiert
auf der CPU in Minuten. Die Small-Konfiguration (d_model=768, 12
Layer) braucht Tage.

Verwende Gradient Accumulation, um größere Batches zu simulieren, ohne
mehr Speicher zu benötigen. Das beschleunigt das Training aber nicht.
Es erlaubt dir nur eine größere effektive Batch Size.

Verwende eine Cloud-GPU. Google Colab stellt eine kostenlose T4-GPU
bereit. Führe das Colab-Notebook für Training per Ein-Klick aus.

Verwende Apple MPS, wenn du einen Mac hast. Main.py erkennt MPS
inzwischen automatisch und aktiviert Mixed Precision.

### Q: Muss ich alle 50,000 Schritte trainieren?

Nein. Du kannst früher aufhören. 500 Schritte mit der
Tiny-Konfiguration ergeben einen Loss von etwa 6 bis 7. Das Modell
wird nicht gut sein, aber es beweist, dass der Code funktioniert. Bei
der Small-Konfiguration zeigen 5,000 Schritte deutliches Lernen.
50,000 Schritte sind für Produktionsqualität gedacht. Höre auf, sobald
du mit dem generierten Text zufrieden bist.

### Q: Woran erkenne ich, dass mein Modell overfittet?

Wenn der Trainings-Loss weiter sinkt, das Modell aber sich
wiederholenden oder unsinnigen Text erzeugt, overfittet es. Das Modell
hat die Trainingsdaten auswendig gelernt, statt allgemeine Muster zu
lernen.

Lösung: Erhöhe den Dropout (versuche 0.2 oder 0.3). Erhöhe den Weight
Decay (versuche 0.2). Reduziere die Anzahl der Trainingsschritte.
Verwende ein größeres und vielfältigeres Dataset.

## Generierung

### Q: Das Modell erzeugt Kauderwelsch

Das ist normal für ein zufällig initialisiertes Modell oder ein
Modell, das nur für sehr wenige Schritte trainiert wurde. Selbst 500
Schritte mit der Tiny-Konfiguration erzeugen größtenteils Kauderwelsch.
Das Modell braucht Tausende von Schritten, um zusammenhängenden Text
zu erzeugen.

Wenn das Modell für viele Schritte trainiert wurde und trotzdem
Kauderwelsch erzeugt, prüfe, ob der Tokenizer derselbe ist, der auch
beim Training verwendet wurde. Die Verwendung eines anderen Tokenizers
bei der Generierung als beim Training erzeugt Datenmüll, weil die
Token-IDs dann etwas anderes bedeuten.

### Q: Das Modell wiederholt immer wieder dieselbe Phrase

Das ist ein bekanntes Problem namens repetitive degeneration
(repetitive Entartung). Das Modell lernt, dass die Wiederholung seiner
selbst eine sichere Vorhersage ist, weil sich wiederholende Muster in
Texten häufig vorkommen.

Lösung: Erhöhe die Temperature auf 0.8 oder 1.0. Verwende
Top-k-Sampling mit k=50. Verwende Top-p-Sampling mit p=0.9. Das
verhindert, dass das Modell immer das wahrscheinlichste Token wählt,
was oft eine Wiederholung ist.

### Q: Wie mache ich das Modell kreativer?

Erhöhe die Temperature auf 1.2 oder 1.5. Entferne top_k oder erhöhe es
auf 100. Setze top_p auf 0.95. Das Modell wählt dann öfter weniger
wahrscheinliche Tokens, was zu abwechslungsreicherer Ausgabe führt.

### Q: Wie mache ich das Modell faktentreuer?

Senke die Temperature auf 0.3 oder 0.5. Setze top_k auf 20. Verwende
top_p von 0.5 bis 0.7. Das Modell hält sich dann an seine sichersten
Vorhersagen. Genauer, aber weniger interessant.

### Q: Was ist das `<|endoftext|>`-Token, das ich in meiner Ausgabe sehe?

Das ist der Textende-Marker. Das Modell wurde mit diesem Token
zwischen Dokumenten trainiert. Bei der Generierung sagt das Modell
dieses Token manchmal voraus, was bedeutet, dass es den Text für
beendet hält. Du kannst es herausfiltern oder die Generierung stoppen,
sobald es erscheint.

## Architektur

### Q: Warum RoPE statt gelernter Positional Embeddings?

Gelernte Positional Embeddings können keine Sequenzen verarbeiten, die
länger sind als die Trainingslänge. Wurde mit 1024 Tokens trainiert,
kann das Modell keine 2048 Tokens verarbeiten. RoPE erfasst relative
Position und generalisiert daher auf beliebige Längen. RoPE hat
außerdem keine gelernten Parameter. Eine Verbesserung zum Nulltarif.

### Q: Warum RMSNorm statt LayerNorm?

RMSNorm ist mathematisch einfacher und etwa 15 Prozent schneller. Es
verzichtet auf die Zentrierung um den Mittelwert und den Bias, die
sich in Experimenten als unnötig erwiesen haben. Jedes moderne Modell
verwendet RMSNorm.

### Q: Warum SwiGLU statt ReLU oder GELU?

SwiGLU verfügt über einen Gating-Mechanismus. Er lernt, welche
Informationen durchgelassen und welche blockiert werden. ReLU und
GELU behandeln jeden Input gleich. Das Gate verleiht SwiGLU pro
Parameter mehr Ausdruckskraft. Bei großem Maßstab führt das zu
besserer Performance.

### Q: Warum Weight Tying?

Der Embedding Layer und der Output Layer führen inverse Operationen
aus. Embeddings bilden Token-IDs auf Vektoren ab. Der Output Layer
bildet Vektoren auf Token-Wahrscheinlichkeiten ab. Das gemeinsame
Nutzen der Matrix spart 30 Prozent der Parameter und verbessert das
Training, weil jedes Token-Embedding Gradientensignale aus beiden
Richtungen erhält.

### Q: Verwendet unser Modell FlashAttention?

Nein. FlashAttention ist ein optimierter CUDA-Kernel, der Attention um
das 2- bis 4-Fache beschleunigt. Er ändert nichts an der Mathematik.
Unsere Implementierung verwendet Standard-PyTorch-Operationen, die
langsamer, aber nachvollziehbarer sind. Für den Produktiveinsatz würde
man FlashAttention einsetzen.

### Q: Verwendet unser Modell Grouped Query Attention?

Nein. Grouped Query Attention reduziert die Anzahl der Key- und
Value-Heads im Verhältnis zu den Query-Heads. Das spart Speicher im
KV-Cache während der Inference. Unser Modell verwendet
Standard-Multi-Head-Attention, bei der Q, K und V alle dieselbe Anzahl
an Heads haben.

## Hardware

### Q: Kann ich auf meinem Laptop trainieren?

Ja. Die Tiny-Konfiguration (256 Dimensionen, 4 Layer, 17M Parameter)
trainiert auf einer modernen Laptop-CPU in 2 bis 5 Minuten. Die
Small-Konfiguration (768 Dimensionen, 12 Layer, 152M Parameter)
braucht auf der CPU Stunden bis Tage. Eine GPU macht einen riesigen
Unterschied.

### Q: Welche GPU brauche ich?

Tiny-Konfiguration: beliebige GPU oder CPU.
GPT-2 Small (152M): mindestens 4GB VRAM. 8GB komfortabel.
GPT-2 Medium (350M): mindestens 8GB VRAM. 12GB komfortabel.
GPT-3 1.3B: mindestens 12GB VRAM. 16GB komfortabel.
GPT-3 6.7B: mindestens 24GB VRAM.
GPT-3 175B: 8x A100 80GB.

### Q: Wie viel Speicher benötigt mein Modell?

Faustregel: Jeder Parameter benötigt 2 Bytes (bfloat16) für die
Weights plus 8 Bytes (float32) für die Optimizer-States während des
Trainings. Insgesamt sind das etwa 10 Bytes pro Parameter für das
Training.

```
17M Parameter:  170 MB Trainingsspeicher
152M Parameter: 1.5 GB Trainingsspeicher
7B Parameter:   70 GB Trainingsspeicher (benötigt mehrere GPUs)
```

## Bugs

### Q: PyTorch 2.6+ kann meinen Checkpoint nicht laden

PyTorch 2.6 hat den Standardwert von `torch.load` auf
`weights_only=True` geändert. Übergib `weights_only=False`, um
Checkpoints zu laden, die eigene Klassen wie GPTConfig enthalten.
Unser Code berücksichtigt das bereits. Falls du ein älteres Notebook
ohne diesen Fix verwendest, füge `weights_only=False` zu deinem
Load-Aufruf hinzu.

### Q: Ich bekomme einen Shape-Mismatch-Fehler in der Attention

Die häufigste Ursache ist, dass Sequenzlänge und Anzahl der Heads
vertauscht sind. Unsere Attention erwartet den Input als [batch,
seq_len, d_model], was intern zu [batch, num_heads, seq_len,
head_dim] umgeformt wird. Wenn dein Input an der Stelle, an der
seq_len stehen sollte, num_heads enthält, schlägt das Broadcasting
fehl.

### Q: Mein Loss wird als 0.0000 ausgegeben

Der Loss wird wahrscheinlich als 0 geteilt durch irgendetwas
berechnet. Prüfe, ob logits und targets die richtigen Shapes haben und
ob cross_entropy korrekt aufgerufen wird. Cross Entropy erwartet
Logits der Form [N, vocab_size] und Targets der Form [N] mit
ganzzahligen Klassenindizes.

### Q: Ich bekomme `RuntimeError: expected scalar type Float but found Half`

Das passiert, wenn Mixed Precision aktiviert ist, eine Operation aber
float32 erwartet. Verwende `torch.amp.autocast(device_type,
enabled=True)` für den Forward Pass und behalte die Loss-Berechnung in
float32 bei. Der autocast-Context-Manager übernimmt die
Dtype-Umwandlung für die meisten Operationen.

### Q: Die Attention-Weights sind nach wenigen Trainingsschritten alle NaN

Das liegt meist daran, dass die Attention-Scores vor dem Softmax zu
groß werden. Prüfe, ob du bei der Berechnung der Attention-Scores
durch `sqrt(head_dim)` teilst. Ohne diese Division können die Scores
groß genug werden, dass `exp(score)` gegen Unendlich überläuft.

### Q: Ich habe import tiktoken ausgeführt und einen Fehler bekommen

Installiere tiktoken: `pip install tiktoken`. Das ist der Tokenizer,
der von GPT-2 und GPT-3 verwendet wird. Er ist in Rust geschrieben und
sehr schnell. Falls bei der Installation ein Kompilierungsfehler
auftritt, versuche `pip install tiktoken --no-binary tiktoken` oder
aktualisiere dein pip.
