# 🧠 Wie man seinen GPT trainiert

> *Eine Anleitung, um ein Weltklasse-Sprachmodell von Grund auf selbst zu bauen. Erklärt, als wärst du fünf Jahre alt. Gebaut, als wärst du Ingenieur.*
>
> *Ich habe das mit dem Ziel gemacht, etwas zu lernen, das ich nicht vollständig verstanden hatte. Insbesondere den Attention-Teil. Ich nutze KI häufig, um zentrale Konzepte zu verstehen und zu überprüfen.*

<p align="center">
  <img src="https://img.shields.io/badge/chapters-12-blue" alt="12 Kapitel">
  <img src="https://img.shields.io/badge/lines-7%2C500%2B-green" alt="7.500+ Zeilen">
  <img src="https://img.shields.io/badge/topics_explained-28-teal" alt="28 Themen-Erklärungen">
  <img src="https://img.shields.io/badge/code%20commented-100%25-brightgreen" alt="100 % kommentiert">
  <img src="https://img.shields.io/badge/prerequisite-python%20basics-orange" alt="Nur Python-Grundkenntnisse">
  <img src="https://img.shields.io/badge/architecture-LLaMA%203%20style-purple" alt="LLaMA-3-Stil">
  <img src="https://img.shields.io/badge/purpose-learning%20only-lightgrey" alt="Nur zum Lernen">
  <a href="https://colab.research.google.com/github/raiyanyahya/how-to-train-your-gpt/blob/master/notebooks/colab_train.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="In Colab öffnen" height="25">
  </a>
</p>

---

## 📖 Was ist das?

Dies ist ein **interaktives Lehrbuch mit 12 Kapiteln und über 7.500 Zeilen**, das dir zeigt, wie du ein modernes Sprachmodell von Grund auf baust, trainierst und ausführst. Dieselbe Architekturfamilie, die hinter ChatGPT, Claude, LLaMA und Mistral steckt.

Neben den Kapiteln gibt es **28 eigenständige Themen-Erklärungen**, die jede Technik im Detail behandeln. RoPE, Attention, RMSNorm, SwiGLU, KV-Cache, AdamW, Mixed Precision und mehr. Dazu zwei erzählerische Walkthroughs, die einen einzelnen Satz Schritt für Schritt durch das gesamte Modell verfolgen. Jede Datei folgt demselben Stil: kindgerechte Sprache, kein Fachjargon, ein Codebeispiel zum Ausprobieren.

Du wirst nicht nur über Transformer lesen. Du wirst **jede Zeile selbst schreiben**: Tokenizer, Embeddings, Attention, Trainingsschleife, Inferenz-Engine. Jede einzelne Zeile ist kommentiert, um zu erklären, **was** sie tut und **warum** sie da ist.

---

## 🤔 Warum es das gibt

Die meisten ML-Tutorials tappen in eine von zwei Fallen:

| ❌ Zu oberflächlich | ❌ Zu akademisch | ✅ Diese Anleitung |
|---|---|---|
| `model = GPT().fit(data)` | 40-seitige Paper, dichte Notation | Analogien für Fünfjährige → vollständig funktionierender Code |
| Du lernst, APIs aufzurufen | Setzt einen Doktortitel in ML voraus | Keine ML-Erfahrung nötig |
| Kein Verständnis der internen Abläufe | Keine durchgerechneten Beispiele | Jede Zeile kommentiert mit WAS & WARUM |

**Das Ziel:** Am Ende weißt du nicht nur, dass Attention „funktioniert". Du verstehst das Varianz-Argument hinter `1/√d_k`. Wie RoPE relative Position durch Rotation erfasst. Warum Pre-Norm bei tiefen Netzwerken besser ist als Post-Norm. Und genau, wohin jeder Gradient während der Backpropagation fließt.

---

## 👥 Für wen ist das?

| 🧑‍💻 Du bist... | 📚 Du brauchst... |
|---|---|
| Ein Python-Entwickler, der neugierig ist, wie ChatGPT tatsächlich funktioniert | Grundlegendes Python (Funktionen, Klassen, Listen). Keine ML-Erfahrung |
| Ein Student, der Transformer wirklich tiefgehend verstehen möchte | Bereitschaft, ~3.500 Zeilen kommentierten Code zu lesen |
| Ein Ingenieur, der LLM-Architekturen bewertet | Verständnis von Kompromissen (RoPE vs. gelernt, RMSNorm vs. LayerNorm) |
| Jemand, der bei „Attention" in anderen Tutorials den Faden verloren hat | Party-Analogie + durchgerechnetes numerisches Beispiel mit echten Zahlen |

**🔧 Voraussetzungen:** Python-Grundlagen (Variablen, Funktionen, Klassen, `pip install`). Das war's. Keine Analysis, keine lineare Algebra, keine PyTorch-Erfahrung nötig. Wir vermitteln das nebenbei.

---

## 🗺️ Kapitel

| Kapitel | Was du lernst |
|---|---|
| **[0: Überblick](chapters/00_overview.md)** | Was ist ein GPT? Der große Zusammenhang |
| **[1: Setup](chapters/01_setup.md)** | Tools installieren, GPU vs. CPU, venv, PyTorch-Grundlagen |
| **[2: Tokenisierung](chapters/02_tokenization.md)** | BPE-Walkthrough: Wie „unbelievably" zu Token wird |
| **[3: Embeddings](chapters/03_embeddings.md)** | Wie aus Zahlen Bedeutung wird. king − man + woman = queen |
| **[4: Positionscodierung](chapters/04_positional_encoding.md)** | RoPE: Warum LLaMA Vektoren rotiert statt Zahlen zu addieren |
| **[5: Attention](chapters/05_attention.md)** | ⭐ DER KERN. Q, K, V, Skalierung, Causal Mask, 8-Schritte-Walkthrough |
| **[6: Transformer-Block](chapters/06_transformer_block.md)** | RMSNorm, SwiGLU, Residual Connections, Pre-Norm vs. Post-Norm |
| **[7: Vollständiges GPT-Modell](chapters/07_gpt_model.md)** | 151-Mio.-Parameter-Modell (mit SwiGLU), Weight Tying, Logits erklärt |
| **[8: Trainings-Pipeline](chapters/08_training.md)** | Cross-Entropy, Backpropagation, AdamW, Cosine Warmup, Mixed Precision |
| **[9: Inferenz](chapters/09_inference.md)** | KV-Cache, Temperature, Top-k/p, Beam Search, Repetition Penalty |
| **[10: Vollständiges Skript](chapters/10_full_script.md)** | Lauffähige `main.py`: alles in einer Datei |
| **[11: Glossar](chapters/11_glossary.md)** | Tabelle zur Architektur-Herkunft, Parameteraufschlüsselung |

> ⭐ **Beginne mit [Kapitel 0](chapters/00_overview.md) und lies der Reihe nach.** Jedes Kapitel baut auf dem vorherigen auf.

---

## 🏗️ Was du bauen wirst

| 🧩 Komponente | 📝 Zeilen | 💡 Was du verstehen wirst |
|---|---|---|
| **BPE-Tokenizer** | ~60 | Wie GPT-4 „unbelievably" in „un" + „believ" + „ably" aufteilt |
| **Embeddings** | ~30 | Wie „cat" und „dog" im 768-dimensionalen Raum nah beieinander landen |
| **RoPE** | ~70 | Warum LLaMA Vektoren rotiert, statt Positionszahlen zu addieren |
| **Multi-Head Attention** | ~120 | Die exakte 8-Schritte-Berechnung hinter jedem modernen LLM |
| **Transformer-Block** | ~50 | Warum Residual Connections die „Gradienten-Autobahn" sind |
| **Vollständiges GPT-Modell** | ~200 | 151-Mio.-Parameter-Modell mit SwiGLU, Weight Tying und Pre-Norm |
| **Trainings-Pipeline** | ~250 | AdamW, Cosine Warmup, Mixed Precision, Gradient Accumulation |
| **Inferenz-Engine** | ~80 | KV-Cache, Temperature, Top-k/p, Beam Search |

> 💎 **~860 Zeilen Kern-Modellcode, ~2.600 Zeilen Erklärungen und Diagramme**

---

## 🏛️ Architektur

Diese Anleitung implementiert den **aktuellsten öffentlich dokumentierten** reinen Decoder-Transformer:

| 🧬 Technik | 📦 Quellmodell | ⚡ Warum es wichtig ist |
|---|---|---|
| **RoPE** | LLaMA, Mistral, Qwen | Relative Position ohne gelernte Parameter |
| **RMSNorm** | LLaMA, Mistral, Gemma | 15 % schneller als LayerNorm, gleich effektiv |
| **SwiGLU** | PaLM, LLaMA, Gemini | Lernt, welche Information durchgelassen oder blockiert wird |
| **Pre-Norm** | GPT-3, alle modernen | Stabiles Training bei 100+ Layers |
| **AdamW** | GPT-3+ | Bessere Generalisierung als reines Adam |
| **BPE** | GPT-2/3/4 | Verarbeitet jeden Text. Auch unbekannte Wörter und Emojis |
| **Weight Tying** | GPT-2/3 | Spart 30 % Parameter, verbessert das Trainingssignal |
| **Mixed Precision** | Alle produktiven LLMs | 2× Geschwindigkeit, halber Speicherbedarf, gleiche Qualität |

> ℹ️ Die Architekturen von GPT-4 und Claude sind proprietär/nicht offengelegt. Diese Anleitung vermittelt die beste öffentlich bestätigte Architektur: das, was LLaMA 3, Mistral und Qwen 2.5 verwenden.

---

## 🚀 Schnellstart

```bash
# 1. Klonen
git clone https://github.com/raiyanyahya/how-to-train-your-gpt.git
cd how-to-train-your-gpt

# 2. Umgebung erstellen
python -m venv gpt_env
source gpt_env/bin/activate          # Mac/Linux
# gpt_env\Scripts\activate           # Windows

# 3. Abhängigkeiten installieren (CPU-Version. Für GPU siehe unten)
pip install torch tiktoken datasets numpy matplotlib --index-url https://download.pytorch.org/whl/cpu

# Oder die requirements-Datei verwenden
pip install -r requirements.txt

# 4. GPU überprüfen (optional, aber empfohlen)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Leg los mit dem Lesen!
open chapters/00_overview.md
```

Führe das Trainingsskript aus:

```bash
python main.py
```

Standardmäßig wird die winzige Konfiguration verwendet (d_model=256, 4 Layer). Das Training dauert auf der CPU nur wenige Minuten. Für die GPT-2-Größenkonfiguration (151 Mio. Parameter, 768 Dimensionen, 12 Layer) bearbeite die Konfiguration in main.py und kommentiere die größere Konfiguration ein.

> 💻 Die Standardkonfiguration verwendet ein winziges Modell (d_model=256, 4 Layer, 17 Mio. Parameter), das in wenigen Minuten auf der CPU läuft. Für die volle GPT-2-Größe (151 Mio. Parameter, 768 Dimensionen, 12 Layer) bearbeite die Konfiguration in `main.py` und kommentiere die größere Konfiguration ein. Dafür brauchst du eine GPU.

---

## 📓 Jupyter-Notebooks

Neben dem Lehrbuch hat jedes Kapitel ein begleitendes Notebook, das du live ausführen kannst. Diese verzichten auf die Erklärungen und geben dir puren, sauberen Code, der von oben nach unten durchläuft. Wenn dir das Lehrbuch das Warum beibringt, zeigen dir die Notebooks, wie es tatsächlich passiert.

Wir führen das gesamte Projekt auf einem sehr kleinen Datensatz aus, damit du das Training in Minuten statt Wochen beobachten kannst. Jedes Notebook ist eigenständig. Öffne es, führe alle Zellen aus und du siehst, wie das Modell in Echtzeit lernt.

```bash
# Installiere alles, was du brauchst
pip install jupyter tiktoken torch numpy datasets matplotlib --index-url https://download.pytorch.org/whl/cpu

# Starte mit Kapitel 2 (Tokenisierung)
jupyter notebook notebooks/02_tokenization.ipynb
```

Die Notebooks liegen im Verzeichnis `notebooks/`, eines pro Kapitel. Öffne eines davon und klicke auf **Cell → Run All**.

---

## 📚 Themen-Erklärungen

Jedes Konzept in dieser Anleitung hat eine eigene ausführliche Vertiefung im Verzeichnis `explanations and examples WIP/`. Diese sind in der einfachstmöglichen Sprache geschrieben. Kein Fachjargon. Keine Formeln vor Analogien. Jede Erklärung deckt das Was, Wo, Warum, Wann und Wie ab, mit einem Codebeispiel zum Ausprobieren.

Die letzten beiden Dateien sind erzählerische Walkthroughs. A Token's Journey verfolgt einen Satz durch das gesamte Modell. The Complete Story deckt jede Komponente in 22 Teilen ab. Lies diese nach den Kapiteln, um zu sehen, wie alles zusammenhängt.

| Thema | Datei | Was es abdeckt |
|---|---|---|
| RoPE | [rope.md](explanations%20and%20examples%20WIP/rope.md) | Wie die Wortreihenfolge durch Rotation codiert wird |
| Attention | [attention.md](explanations%20and%20examples%20WIP/attention.md) | Schritt für Schritt mit einem durchgerechneten 3-Token-Beispiel |
| BPE-Tokenisierung | [bpe_tokenization.md](explanations%20and%20examples%20WIP/bpe_tokenization.md) | Wie aus Text Token werden |
| Embeddings | [embeddings.md](explanations%20and%20examples%20WIP/embeddings.md) | Wie aus Zahlen Bedeutung wird |
| RMSNorm | [rmsnorm.md](explanations%20and%20examples%20WIP/rmsnorm.md) | Einfachere, schnellere Normalisierung |
| SwiGLU | [swiglu.md](explanations%20and%20examples%20WIP/swiglu.md) | Die gegatete Aktivierung, die ReLU übertraf |
| Causal Masking | [causal_masking.md](explanations%20and%20examples%20WIP/causal_masking.md) | Kein Blick in die Zukunft |
| Residual Connections | [residual_connections.md](explanations%20and%20examples%20WIP/residual_connections.md) | Die Gradienten-Autobahn |
| KV-Cache | [kv_cache.md](explanations%20and%20examples%20WIP/kv_cache.md) | Wie Generierung schnell wird |
| Sampling | [sampling.md](explanations%20and%20examples%20WIP/sampling.md) | Temperature, Top-k, Top-p |
| Mixed Precision | [mixed_precision.md](explanations%20and%20examples%20WIP/mixed_precision.md) | Geschwindigkeit ohne Kompromisse |
| AdamW | [adamw.md](explanations%20and%20examples%20WIP/adamw.md) | Der Optimizer, der LLMs trainiert |
| Weight Tying | [weight_tying.md](explanations%20and%20examples%20WIP/weight_tying.md) | Zwei Aufgaben, eine Matrix |
| Gradient Clipping | [gradient_clipping.md](explanations%20and%20examples%20WIP/gradient_clipping.md) | Trainingsexplosionen verhindern |
| Cosine Warmup | [cosine_warmup.md](explanations%20and%20examples%20WIP/cosine_warmup.md) | Der Learning-Rate-Schedule |
| Pre-Norm | [pre_norm.md](explanations%20and%20examples%20WIP/pre_norm.md) | Wo normalisiert wird |
| Grouped Query Attention | [grouped_query_attention.md](explanations%20and%20examples%20WIP/grouped_query_attention.md) | MHA vs. GQA vs. MQA erklärt |
| Flash Attention | [flash_attention.md](explanations%20and%20examples%20WIP/flash_attention.md) | Wie Flash Attention das Training 4× schneller macht |
| Loss-Kurven | [how_to_read_loss.md](explanations%20and%20examples%20WIP/how_to_read_loss.md) | Trainingsprobleme anhand der Loss-Kurve diagnostizieren |
| Mixture of Experts | [mixture_of_experts.md](explanations%20and%20examples%20WIP/mixture_of_experts.md) | Wie MoE Modelle mit spärlichem Routing skaliert |
| Speculative Decoding | [speculative_decoding.md](explanations%20and%20examples%20WIP/speculative_decoding.md) | 2-3× schnellere Generierung mit einem Draft Model |
| Perplexity | [perplexity.md](explanations%20and%20examples%20WIP/perplexity.md) | Die eine Zahl, die dein Modell bewertet |
| Beam Search | [beam_search.md](explanations%20and%20examples%20WIP/beam_search.md) | Präziseren Text mit mehreren Kandidaten generieren |
| Cheatsheet | [cheatsheet.md](explanations%20and%20examples%20WIP/cheatsheet.md) | Jede Formel und jeder Hyperparameter an einem Ort |
| FAQ | [faq.md](explanations%20and%20examples%20WIP/faq.md) | Häufige Probleme beheben |
| Encoder vs. Decoder | [encoder_decoder_architectures.md](explanations%20and%20examples%20WIP/encoder_decoder_architectures.md) | GPT vs. BERT vs. T5 erklärt |
| 📖 **A Token's Journey** | [a_tokens_journey.md](explanations%20and%20examples%20WIP/a_tokens_journey.md) | Verfolge einen Satz durch jeden Layer |
| 📖 **The Complete Story** | [the_complete_story.md](explanations%20and%20examples%20WIP/the_complete_story.md) | Die vollständige Erzählung: 22 Teile, 7800 Wörter |

---

## 📖 So liest du

Jedes Kapitel folgt derselben **4-Schritte-Struktur**:

| Schritt | Format | Zweck |
|---|---|---|
| 1️⃣ **Analogie** | Einfache Sprache, Niveau eines Fünfjährigen | Intuition aufbauen, bevor die Mathematik kommt |
| 2️⃣ **Durchgerechnetes Beispiel** | Echte Zahlen werden Schritt für Schritt verfolgt | Genau sehen, was passiert |
| 3️⃣ **Kommentierter Code** | Jede Zeile: `WAS` + `WARUM` | Jede Entscheidung verstehen |
| 4️⃣ **Diagramm** | Mermaid-Flowchart oder ASCII | Datenfluss visualisieren |

> 💡 **Tipp:** Im Code verloren? Spring zurück zur Analogie. Von der Mathematik verwirrt? Spring zum durchgerechneten Beispiel.

---

## ✨ Was diese Anleitung anders macht

| Aspekt | 😴 Typisches Tutorial | 🔥 Diese Anleitung |
|---|---|---|
| **Erklärungstiefe** | „Attention hilft dem Modell, sich zu fokussieren" | 8-Schritte-Beispiel mit echten Zahlen + Varianz-Mathematik + Visualisierung der Causal Mask |
| **Code-Kommentare** | Wenige oder keine | Jede einzelne Zeile: WAS + WARUM |
| **Moderne Techniken** | GPT-2-Stil (2019) | LLaMA-3-Stil (2024): RoPE, RMSNorm, SwiGLU |
| **Training** | Nutzt den HuggingFace Trainer | Vollständig eigene Schleife: AdamW, Cosine Warmup, Mixed Precision, Grad Accumulation |
| **Inferenz** | `model.generate()` | Temperature, Top-k, Top-p, Beam Search, KV-Cache erklärt |
| **Zielgruppe** | ML-Ingenieure | Python-Entwickler ohne ML-Erfahrung |
| **Diagramme** | Keine | Mermaid-Flowcharts + ASCII-Matrizen + durchgerechnete Beispiele |

---

## 🎯 Fähigkeiten, die du gewinnst

- ✅ Erklären, wie GPT-4 Text mittels BPE tokenisiert
- ✅ Verstehen, warum RoPE, RMSNorm und SwiGLU ältere Techniken abgelöst haben
- ✅ Attention-Scores für einen 3-Token-Satz manuell berechnen
- ✅ Eine Transformer-Trainingsschleife debuggen (Loss-Spitzen, flache Verläufe, Overfitting)
- ✅ Sampling-Parameter (temperature, top_k, top_p) für verschiedene Anwendungsfälle wählen
- ✅ Verstehen, warum KV-Caching für den Produktivbetrieb der Inferenz entscheidend ist
- ✅ Moderne ML-Paper selbstbewusst lesen (du erkennst jede Komponente wieder)

---

## 🔮 Nächste Schritte nach dem Abschluss

| Experiment | Was du änderst | Was du lernst |
|---|---|---|
| **Größeres Modell** | `num_layers` 12 → 24 | Wie Tiefe das Schlussfolgern verbessert |
| **Mehr Daten** | BookCorpus, C4, The Pile hinzufügen | Einfluss von Datenqualität und -vielfalt |
| **Flash Attention** | `flash-attn` installieren, Attention austauschen | 2-5× schnelleres Training, längerer Context |
| **Grouped Query Attention** | `num_kv_heads` < `num_heads` setzen | Wie Mistral effiziente Inferenz erreicht |
| **LoRA-Fine-Tuning** | Low-Rank-Adapter-Layer hinzufügen | Modelle anpassen, ohne sie vollständig neu zu trainieren |
| **RLHF / DPO** | Reward-Model-Training hinzufügen | Wie ChatGPT lernt, Anweisungen zu befolgen |
| **KV-Cache** | Persistenten Key-Value-Speicher implementieren | 500× schnellere Textgenerierung |
| **Mixture of Experts** | Token durch verschiedene FFN-Experts routen | Wie GPT-4 auf Billionen von Parametern skaliert |

---

## 📁 Dateistruktur

```
📦 how-to-train-your-gpt/
├── 📄 README.md              ← Du bist hier
├── 🐍 main.py                ← Lauffähiges Trainingsskript (klonen & ausführen)
├── 📋 requirements.txt       ← Installation mit einem Befehl
├── 📂 chapters/
│   ├── 🏠 00_overview.md     ← Was ist ein GPT? Warum eins bauen?
│   ├── 🔧 01_setup.md        ← Tools installieren, GPU vs. CPU, venv-Grundlagen
│   ├── 🔪 02_tokenization.md ← BPE-Walkthrough, EOS-Token, Umgang mit Emojis
│   ├── 🧊 03_embeddings.md   ← Wie aus Zahlen Bedeutung wird, king − man + woman
│   ├── 📍 04_positional_encoding.md ← RoPE-Mathematik, numerisches Beispiel, Theta
│   ├── 🧠 05_attention.md    ← ⭐ DER KERN (713 Zeilen). Q, K, V, Skalierung, Causal Mask
│   ├── 🧱 06_transformer_block.md ← RMSNorm, SwiGLU, Residual Connections, Pre-Norm vs. Post
│   ├── 🏗️ 07_gpt_model.md    ← Vollständiges 151-Mio.-Modell, Weight Tying, Logits erklärt
│   ├── 🏋️ 08_training.md     ← Cross-Entropy, Backpropagation, AdamW, Cosine Warmup
│   ├── 🎤 09_inference.md    ← KV-Cache, Temperature, Top-k/p, Beam Search
│   ├── 📜 10_full_script.md  ← Über main.py
│   └── 📊 11_glossary.md     ← Architektur-Herkunft, Parameteraufschlüsselung
├── 📓 notebooks/             ← Jupyter-Notebooks (eines pro Kapitel)
│   ├── 🎨 attention_visualized.ipynb ← Attention-Gewichte live beobachten
│   └── ☁️ colab_train.ipynb  ← Cloud-Training mit einem Klick auf Colab
├── 🎯 fine-tuning/           ← Fine-Tuning-Anleitung: LoRA, QLoRA, Datenaufbereitung
│   ├── 📄 README.md
│   ├── 01_what_is_finetuning.md
│   ├── 02_lora_explained.md
│   ├── 03_qlora_explained.md
│   ├── 04_data_preparation.md
│   ├── 05_full_finetune.md
│   └── 📓 notebooks/lora_finetune.ipynb
├── 📚 explanations and examples WIP/ ← Eigenständige Erklärungen (28 Themen)
└── 📄 CONTRIBUTING.md
```

---

<p align="center">
  <i>"Jede hinreichend erklärte Technologie ist von Magie nicht zu unterscheiden. Bis du sie selbst gebaut hast."</i>
</p>

<p align="center">
  <sub>⭐ Gib diesem Repo einen Stern, wenn es dir geholfen hat | 🐛 Issues & PRs willkommen | 📖 Viel Spaß beim Lernen!</sub>
</p>
