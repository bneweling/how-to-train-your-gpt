# QLoRA: Quantisiertes LoRA

## Die kurze Antwort

QLoRA ist LoRA plus Quantization. Das Basismodell wird auf vier Bit pro
Weight komprimiert, bevor die LoRA-Adapter angewendet werden. Das
reduziert den Speicherbedarf um das Vier- bis Achtfache im Vergleich
zum Modell in voller Präzision. Ein Modell mit sieben Milliarden
Parametern, das normalerweise vierzehn Gigabyte benötigt, läuft damit
in unter vier Gigabyte. Fine-Tuning wird so auf einem Laptop möglich.

## Wo die Quantization stattfindet

Die Quantization betrifft ausschließlich das eingefrorene Basismodell.
Die LoRA-Adapter-Matrizen bleiben in voller Präzision. Während des
Trainings werden die 4-Bit-Weights für die Forward- und
Backward-Passes vorübergehend zurück auf 16 Bit dequantisiert. Die
Dequantisierung geschieht innerhalb der Berechnung. Man sieht sie
nicht. Das Modell liegt in vier Bit vor, rechnet aber in sechzehn Bit.

```
Basismodell-Weights (eingefroren):
  Gespeichert als 4-Bit-Integers
  Für die Berechnung zu bfloat16 dequantisiert
  Nach der Berechnung requantisiert (Gradienten fließen nicht durch sie)

LoRA-Adapter (trainiert):
  Gespeichert und berechnet in bfloat16
  Gradienten fließen normal durch sie
```

## Warum vier Bit

Die Standarddarstellung float32 verwendet 32 Bit pro Weight. Bfloat16
verwendet 16 Bit. Die 4-Bit-Quantization verwendet 4 Bit. Jede
Reduzierung halbiert sowohl den Speicherbedarf als auch die Zeit, die
zum Lesen der Weights aus dem Speicher benötigt wird.

```
Float32-Modell (7B Parameter):    28 GB  (7B × 4 Byte)
Bfloat16-Modell (7B Parameter):   14 GB  (7B × 2 Byte)
4-Bit-quantisiert (7B Parameter):   3.5 GB (7B × 0.5 Byte)
Plus LoRA-Adapter:           ~0.1 GB (vernachlässigbar)
```

Die Speicherersparnis ist dramatisch. Ein Modell, das in bfloat16
nicht auf eine RTX 3090 mit 24 Gigabyte passen würde, passt in 4-Bit
problemlos hinein – mit noch Raum übrig für Trainings-Batches.

## Wie die Quantization funktioniert

Quantization bildet Fließkommawerte auf eine kleine Menge von
Integer-Werten ab. Die einfachste Form der Quantization ist das Runden
auf die nächste ganze Zahl. Doch die Weights in einem neuronalen Netz
sind nicht gleichmäßig verteilt. Manche Layer haben Weights, die sich
nahe null häufen. Andere haben Weights, die über einen weiten Bereich
verstreut sind.

QLoRA verwendet eine Technik namens NormalFloat4. Sie geht davon aus,
dass die Weights einer Normalverteilung folgen, und weist mehr
Quantization-Level in der Nähe von null zu, wo die meisten Weights
liegen, und weniger Level an den Extremen, wo nur wenige Weights
liegen.

```
Standard-Quantization mit gleichmäßigen Bins (2-Bit-Beispiel):
  0.00 ... 0.25 ... 0.50 ... 0.75 ... 1.00
  Jeder Bin hat die gleiche Breite.

NormalFloat-Quantization (2-Bit-Beispiel):
  0.00 ......... 0.10 . 0.40 ......... 1.00
  Mehr Bins nahe null. Weniger Bins an den Extremen.
```

Das Ergebnis ist ein geringerer Quantization-Fehler bei gleicher
Anzahl von Bit. Die meisten Weights liegen nahe null und erhalten eine
hohe Präzision. Die wenigen Ausreißer-Weights an den Extremen erhalten
eine geringere Präzision. Das entspricht der tatsächlichen Verteilung
der Weights vortrainierter Modelle.

## Ein kleines Code-Beispiel

Hier ist eine vereinfachte Version davon, wie QLoRA Weights während
des Trainings quantisiert und dequantisiert.

```python
import torch

def quantize_4bit(weights, block_size=64):
    """
    Quantisiert eine Weight-Matrix auf 4 Bit.
    Arbeitet blockweise über die Weights für eine bessere Genauigkeit.
    """
    out_features, in_features = weights.shape
    quantized = torch.zeros(out_features, in_features // 2, dtype=torch.uint8)

    for i in range(0, out_features):
        for j_start in range(0, in_features, block_size):
            j_end = min(j_start + block_size, in_features)
            block = weights[i, j_start:j_end]

            # Min und Max dieses Blocks ermitteln
            w_min = block.min()
            w_max = block.max()

            # Auf 4 Bit quantisieren (16 Level)
            scale = (w_max - w_min) / 15.0
            zero_point = -w_min / scale

            quantized_block = torch.clamp(
                torch.round(block / scale + zero_point), 0, 15
            ).to(torch.uint8)

            # Zwei 4-Bit-Werte in ein 8-Bit-Byte packen
            for j in range(0, len(quantized_block), 2):
                byte_idx = j_start // 2 + j // 2
                high = quantized_block[j] << 4
                low = quantized_block[j + 1] if j + 1 < len(quantized_block) else 0
                quantized[i, byte_idx] = high | low

    return quantized


def dequantize_4bit(quantized, original_shape, block_size=64):
    """
    Rekonstruiert angenäherte Fließkommawerte aus 4-Bit-quantisierten Weights.
    Diese vereinfachte Version speichert keine Scales pro Block.
    Im echten QLoRA werden Scale und Zero Point zusätzlich gespeichert.
    """
    out_features, packed_in_features = quantized.shape
    in_features = packed_in_features * 2
    reconstructed = torch.zeros(out_features, min(in_features, original_shape[1]))

    for i in range(out_features):
        for j in range(0, min(in_features, original_shape[1]), 2):
            byte_idx = j // 2
            packed = quantized[i, byte_idx].item()
            high = (packed >> 4) & 0xF
            low = packed & 0xF
            if j < original_shape[1]:
                reconstructed[i, j] = float(high)
            if j + 1 < original_shape[1]:
                reconstructed[i, j + 1] = float(low)

    return reconstructed


# Beispiel: eine kleine Weight-Matrix quantisieren und dequantisieren
weights = torch.randn(4, 16) * 0.5
quantized = quantize_4bit(weights, block_size=8)
reconstructed = dequantize_4bit(quantized, weights.shape, block_size=8)

# Im echten QLoRA werden Scale und Offset pro Block gespeichert,
# sodass rekonstruierte Werte den Originalen deutlich näher kommen
print(f"Original shape: {weights.shape}")
print(f"Quantized shape: {quantized.shape}")
print(f"Compression ratio: {weights.numel() * 4 / quantized.numel():.1f}x")
```

In einer echten QLoRA-Implementierung werden Scale und Zero Point
zusammen mit den quantisierten Werten für jeden Block gespeichert. Bei
der Dequantisierung werden die Integer-Werte anhand ihres
blockspezifischen Scale und Zero Point zurückkonvertiert. Das liefert
eine deutlich höhere Genauigkeit als die vereinfachte Version oben,
die die Metadaten pro Block verliert.

## Doppelte Quantization

QLoRA wendet eine zweite Runde Quantization auf die
Quantization-Konstanten selbst an. Jeder Block von Weights hat einen
Skalierungsfaktor, der zwischen der Integer-Darstellung und der
Fließkomma-Darstellung umrechnet. Diese Skalierungsfaktoren werden
standardmäßig als float32 gespeichert. Die doppelte Quantization
komprimiert sie auf acht Bit.

Die Ersparnis durch die doppelte Quantization ist gering. Im Schnitt
etwa 0,13 Bit pro Parameter. Bei einem Modell mit sieben Milliarden
Parametern sind das aber rund 120 Megabyte. Genug, um einen
Unterschied zu machen, wenn ein Modell auf eine GPU mit knappem
Speicher passen muss.

## Paged Optimizers

Das Training mit QLoRA verwendet einen Optimizer, der den GPU-Speicher
so verwaltet, wie ein Betriebssystem den RAM verwaltet. Wenn der
GPU-Speicher ausgeht, lagert der Optimizer einige Optimizer-States in
den CPU-RAM aus. Werden diese States wieder gebraucht, werden sie
zurück in den GPU-Speicher geladen.

Das verhindert Out-of-Memory-Fehler während des Trainings. Ohne Paging
könnte eine Speicherspitze durch eine große Gradientenberechnung das
Training zum Absturz bringen. Mit Paging verschiebt der Optimizer
Daten zwischen GPU und CPU hin und her, um die Spitze abzufangen.

Die Technik ist von Betriebssystemen entlehnt, wo virtueller Speicher
es Prozessen erlaubt, mehr Speicher zu nutzen, als physisch verfügbar
ist, indem Pages auf die Festplatte ausgelagert werden. Die gleiche
Idee wird hier auf die Verwaltung des GPU-Speichers während des
Trainings angewendet.

## Der Kompromiss

QLoRA ist wegen des Dequantisierungs-Overheads etwas langsamer als
LoRA. Jeder Forward- und Backward-Pass muss die Weights des
Basismodells von vier Bit auf sechzehn Bit umwandeln und wieder
zurück. Das kostet zusätzliche Rechenzeit. Aber die Speicherersparnis
überwiegt in der Regel den Geschwindigkeitsnachteil. Ein Modell, das
sich in bfloat16 überhaupt nicht trainieren lässt, kann mit QLoRA
erfolgreich trainiert werden.

Die Qualität ist etwas geringer als bei LoRA. Die Quantization führt
Rauschen ein. Bei der Umwandlung auf vier Bit geht etwas Information
verloren. Der Verlust ist jedoch gering. Bei den meisten Aufgaben
liegt der Unterschied zwischen QLoRA und LoRA bei Standard-Benchmarks
im Bereich weniger Prozent.

## Was sich wo ausführen lässt

```
16 GB GPU (RTX 4080 Laptop):
  LoRA: Fine-Tuning von Modellen bis ~7B
  QLoRA: Fine-Tuning von Modellen bis ~13B

24 GB GPU (RTX 3090/4090):
  LoRA: Fine-Tuning von Modellen bis ~13B
  QLoRA: Fine-Tuning von Modellen bis ~34B

12 GB GPU (RTX 3060/4060):
  LoRA: Fine-Tuning von Modellen bis ~3B
  QLoRA: Fine-Tuning von Modellen bis ~7B

8 GB GPU (RTX 2070):
  LoRA: Fine-Tuning von Modellen bis ~1B
  QLoRA: Fine-Tuning von Modellen bis ~3B
```

## Ein echtes QLoRA-Trainings-Snippet

Unter Verwendung der beliebten bitsandbytes-Bibliothek, die QLoRA
implementiert.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-Bit-Quantization konfigurieren
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # Berechnung in bfloat16
    bnb_4bit_use_double_quant=True,        # Doppelte Quantization aktivieren
)

# Ein 7B-Modell in 4-Bit laden (~4 GB statt ~14 GB)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA obendrauf hinzufügen (mit der peft-Bibliothek)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 6,746,726,400 || trainable%: 0.12%

# Jetzt wie gewohnt trainieren. Nur die LoRA-Parameter werden aktualisiert.
# Das 4-Bit-Basismodell bleibt eingefroren.
```

## Was man sich merken sollte

QLoRA komprimiert das eingefrorene Basismodell auf vier Bit. Die
LoRA-Adapter bleiben in voller Präzision. Die Kombination ermöglicht
Fine-Tuning für Modelle, die sonst nicht in den GPU-Speicher passen
würden. Die NormalFloat4-Quantization liefert bei gleichem Bit-Budget
eine höhere Genauigkeit als eine gleichmäßige Quantization. Die
doppelte Quantization holt noch etwas mehr Speicher aus den
Quantization-Konstanten heraus. Paged Optimizers verhindern
Out-of-Memory-Abstürze. QLoRA ist der Standardansatz für Fine-Tuning
auf Consumer-Hardware.
