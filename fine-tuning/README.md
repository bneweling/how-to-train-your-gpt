# Fine-Tuning: Das Modell nützlich machen

Dieser Ordner behandelt alles rund um die Anpassung eines vortrainierten
Sprachmodells für eine bestimmte Aufgabe. Pretraining bringt dem Modell
Sprache bei. Fine-Tuning bringt ihm bei, hilfreich zu sein.

## Inhalt

| Datei | Thema |
|---|---|
| [01_what_is_finetuning.md](01_what_is_finetuning.md) | Das Konzept. Warum wir fine-tunen. Typen: Full, LoRA, QLoRA. |
| [02_lora_explained.md](02_lora_explained.md) | LoRA im Detail. Low-Rank-Zerlegung. Die Mathematik einfach erklärt. |
| [03_qlora_explained.md](03_qlora_explained.md) | QLoRA. Quantization plus LoRA. Läuft auf einem Laptop. |
| [04_data_preparation.md](04_data_preparation.md) | Wie man Daten für Instruction Tuning formatiert. Chat-Templates. |
| [05_full_finetune.md](05_full_finetune.md) | Vollständiges Fine-Tuning. Wann und warum nicht. |
| [06_dpo_explained.md](06_dpo_explained.md) | DPO. Präferenzoptimierung ohne RL. |
| [07_prompt_vs_finetune.md](07_prompt_vs_finetune.md) | Wann Prompt Engineering sinnvoll ist und wann Fine-Tuning. |

## Notebook

[`notebooks/lora_finetune.ipynb`](notebooks/lora_finetune.ipynb). Ein
lauffähiges Notebook, das ein kleines Modell mit LoRA auf einem einfachen
Instruction-Datensatz fine-tunt. Läuft auf einer einzelnen Consumer-GPU.

## Empfohlene Lesereihenfolge

Beginne mit 01 für den Überblick. Dann 02, um LoRA zu verstehen, das von
fast allen verwendet wird. 03 fügt Quantization für noch kleinere GPUs
hinzu. 04 zeigt dir, wie du deine Daten vorbereitest. 05 behandelt den
vollständigen Ansatz. 06 erklärt DPO, die einfachere Alternative zu RLHF.
07 hilft dir, dich zwischen Prompting und Fine-Tuning für deinen eigenen
Anwendungsfall zu entscheiden.
