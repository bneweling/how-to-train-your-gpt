# Encoder, Decoder und Encoder-Decoder: Die drei Transformer-Familien

## Die kurze Antwort

Es gibt drei Wege, einen Transformer zu bauen. Decoder-only-Modelle wie
GPT erzeugen Text Token für Token. Sie können nur sehen, was vorher kam.
Encoder-only-Modelle wie BERT betrachten die gesamte Eingabe auf einmal
in beide Richtungen. Sie verstehen Text, können ihn aber nicht generieren.
Encoder-Decoder-Modelle wie T5 lesen die vollständige Eingabe mit einem
Encoder und erzeugen die Ausgabe mit einem Decoder. Sie sind für Aufgaben
gebaut, die einen Textabschnitt in einen anderen transformieren.

Dieser Guide baut ein Decoder-only-Modell. Dieselbe Architektur wie GPT,
LLaMA und Mistral. Diese Datei erklärt, warum, und was die anderen
Optionen leisten.

## Decoder-Only (GPT-Familie)

### So sieht es aus

```
Eingabe: "The cat sat on the"
  → Token-Embedding
    → Causal Attention (kann nur rückwärts schauen)
      → Feed-Forward
        → N-mal wiederholen
          → Ausgabeprojektion
            → Nächstes Token vorhersagen: "mat"
```

Das zentrale Merkmal ist die Causal Mask. Jedes Token kann nur auf
Token achten, die davor kamen. Token 5 kann Token 0 bis 4 sehen. Token
5 kann Token 6 nicht sehen, weil Token 6 noch nicht geschrieben wurde.

### So wird trainiert

Decoder-only-Modelle werden mit Next-Token-Prediction trainiert. Man
zeigt dem Modell eine Sequenz von Token. Man lässt es jedes nächste
Token vorhersagen. Das Modell lernt zu erraten, was als Nächstes kommt.
Das nennt man autoregressives Training.

```
Trainingsbeispiel:
  Eingabe: [The, cat, sat, on, the]
  Ziel:    [cat, sat, on, the, mat]

  Das Modell sieht "The" und muss "cat" vorhersagen
  Das Modell sieht "The cat" und muss "sat" vorhersagen
  Das Modell sieht "The cat sat" und muss "on" vorhersagen
  ... und so weiter
```

Jede Vorhersage wird nur anhand vergangener Token getroffen. Das Modell
sieht die Zukunft nie. Die Causal Mask erzwingt das während des
Trainings. Bei der Generierung wird keine Maske benötigt, weil zukünftige
Token schlicht noch nicht existieren.

### Worin es gut ist

Textgenerierung. Geschichten schreiben. Fragen beantworten. Gespräche
führen. Code vervollständigen. Jede Aufgabe, bei der neuer Text Token
für Token erzeugt wird.

Decoder-only-Modelle sind das universelle Werkzeug. Mit ausreichender
Skalierung und den richtigen Trainingsdaten können sie fast alles. GPT-3
hat das 2020 gezeigt. Das Modell konnte übersetzen, zusammenfassen und
Fragen beantworten, obwohl es nur darauf trainiert war, das nächste Wort
vorherzusagen. Es hat diese Fähigkeiten implizit gelernt, weil die
Trainingsdaten Beispiele für Übersetzung, Zusammenfassung und
Frage-Antwort-Aufgaben enthielten.

### Warum wir uns dafür entschieden haben

Decoder-only-Modelle sind am einfachsten zu bauen und zu trainieren.
Eine Aufgabe. Das nächste Token vorhersagen. Eine Architektur. Causal
Attention, deren Ausgabe durch ein Feed-Forward-Netzwerk läuft. Kein
separater Encoder. Keine Cross-Attention zwischen Encoder und Decoder.
Ein Stapel identischer Blöcke.

Sie skalieren außerdem am besten. Jeder große Fähigkeitssprung von GPT-2
über GPT-3 bis GPT-4 kam von Decoder-only-Modellen. Die Einfachheit der
Architektur bedeutet, dass alle Ressourcen darin fließen, das Modell
größer und die Daten besser zu machen. Es wird kein Komplexitätsbudget
für zusätzliche Komponenten verbraucht.

### Die Einschränkung

Decoder-only-Modelle können die vollständige Eingabe nicht bidirektional
betrachten. Token 5 kann keine Information von Token 10 nutzen, weil
Token 10 noch nicht existiert. Das ist für die Generierung in Ordnung,
aber suboptimal für Verständnisaufgaben, bei denen die gesamte Eingabe
von Anfang an verfügbar ist.

Bei Aufgaben wie Klassifikation oder Named Entity Recognition, bei denen
die vollständige Eingabe vorliegt, kann ein bidirektionales Modell
Kontext aus beiden Richtungen erfassen. Ein Decoder-only-Modell kann nur
Kontext von links erfassen. In der Praxis spielt das eine geringere
Rolle, als man annehmen könnte. Mit ausreichender Skalierung lernt ein
Decoder-only-Modell, den fehlenden rechten Kontext auszugleichen, indem
es reichhaltige Repräsentationen aufbaut, die vorwegnehmen, was als
Nächstes kommt.

## Encoder-Only (BERT-Familie)

### So sieht es aus

```
Eingabe: "The cat sat on the [MASK]"
  → Token-Embedding
    → Bidirektionale Attention (kann überallhin schauen)
      → Feed-Forward
        → N-mal wiederholen
          → Ausgabeprojektion
            → Maskiertes Token vorhersagen: "mat"
```

Das zentrale Merkmal ist bidirektionale Attention. Jedes Token kann auf
jedes andere Token achten, unabhängig von der Position. Es gibt keine
Causal Mask. Token 5 kann Token 0 und Token 10 gleichermaßen sehen.

### So wird trainiert

Encoder-only-Modelle werden mit Masked Language Modeling trainiert. Ein
bestimmter Prozentsatz der Eingabe-Token wird zufällig verborgen. Das
Modell wird gebeten vorherzusagen, was verborgen wurde.

```
Trainingsbeispiel:
  Original:   "The cat sat on the mat"
  Maskiert:   "The cat [MASK] on the [MASK]"
  Ziel:       "sat" und "mat"

  Das Modell sieht den gesamten Satz einschließlich der Wörter nach der Maskierung.
  Es nutzt Kontext aus beiden Richtungen, um die verborgenen Wörter vorherzusagen.
```

Das Modell sieht die gesamte Eingabe auf einmal. Es kann Information
aus Wörtern vor UND nach dem maskierten Token nutzen. Das unterscheidet
sich grundlegend vom Training von Decoder-only-Modellen, bei dem das
Modell blind für die Zukunft ist.

### Worin es gut ist

Verständnisaufgaben. Klassifikation. Named Entity Recognition.
Frage-Antwort-Aufgaben, bei denen die Antwort im bereitgestellten Text
steht. Sentiment-Analyse. Jede Aufgabe, bei der die Eingabe vollständig
ist und die Ausgabe ein Label oder eine Textspanne ist statt einer
generierten Sequenz.

BERT-Embeddings wurden zum Standard für die Repräsentation von Text.
Jahrelang bestand der beste Ansatz für jede NLP-Aufgabe darin, ein
vortrainiertes BERT-Modell zu nehmen und einen kleinen, aufgabenspezifischen
Head obendrauf zu setzen. Ein paar Epochen fine-tunen. Der Ansatz
funktionierte, weil BERTs bidirektionales Verständnis reichhaltige
Repräsentationen der Wortbedeutung im Kontext einfing.

### Die Einschränkung

Encoder-only-Modelle können nicht autoregressiv Text generieren. Sie
haben keine Causal Mask. Sie haben keinen Mechanismus, um Token für
Token bedingt auf vorherige Ausgaben zu erzeugen. Man kann BERT nicht
verwenden, um eine Geschichte zu schreiben oder ein Gespräch zu führen.

Encoder-only-Modelle sind außerdem durch ihr Trainingsziel eingeschränkt.
Masked Language Modeling bringt dem Modell bei, Lücken zu füllen. Es
bringt dem Modell nicht bei, kohärente Sequenzen zu erzeugen. Man kann
Text erzeugen, indem man iterativ maskiert und vorhersagt, aber die
Ausgabe ist typischerweise schlechter als das, was ein Decoder-only-Modell
produziert.

## Encoder-Decoder (T5-Familie)

### So sieht es aus

```
Eingabe: "Translate to French: The cat sat on the mat"
  → Encoder (bidirektionale Attention)
    → Verborgene Repräsentation der gesamten Eingabe
      → Decoder (Causal Attention + Cross-Attention)
        → Ausgabe: "Le chat s'est assis sur le tapis"
```

Der Encoder liest die gesamte Eingabe bidirektional. Er erzeugt eine
dichte Repräsentation der Eingabe. Der Decoder generiert die Ausgabe
autoregressiv Token für Token. Der Decoder verfügt sowohl über Causal
Self-Attention wie ein GPT als auch über Cross-Attention, die auf die
Ausgabe des Encoders schaut.

Die Cross-Attention ist der entscheidende Unterschied zu
Decoder-only-Modellen. Bei jedem Generierungsschritt kann der Decoder auf
die vollständige, encodierte Eingabe zurückblicken. Das gibt dem Decoder
direkten Zugriff auf die Eingaberepräsentation, ohne sie im
autoregressiven Zustand codieren zu müssen.

### So wird trainiert

Encoder-Decoder-Modelle werden auf Sequence-to-Sequence-Aufgaben
trainiert. Man zeigt dem Modell eine Eingabesequenz und eine
Ziel-Ausgabesequenz. Der Encoder verarbeitet die Eingabe. Der Decoder
erzeugt die Ausgabe Token für Token.

```
Trainingsbeispiel:
  Eingabe: "Summarize: The cat sat on the mat for three hours..."
  Ziel:    "A cat stayed on a mat for a long time."

  Der Encoder liest die gesamte Eingabe bidirektional.
  Der Decoder erzeugt "A", dann "cat", dann "stayed" und so weiter.
  Bei jedem Schritt kann der Decoder eine Cross-Attention auf die Ausgabe des Encoders anwenden.
```

Das Training verwendet Teacher Forcing. Dem Decoder werden während des
Trainings die korrekten vorherigen Token gegeben. Das Modell lernt, das
nächste Token anhand der Eingabe und der korrekten Historie zu erzeugen.

### Worin es gut ist

Sequence-to-Sequence-Aufgaben. Übersetzung. Zusammenfassung. Jede
Aufgabe, bei der Eingabe und Ausgabe beide Text sind, aber unterschiedliche
Längen oder Strukturen haben.

Encoder-Decoder-Modelle trennen die Zuständigkeiten. Der Encoder
konzentriert sich auf das Verstehen der Eingabe. Der Decoder konzentriert
sich auf das Erzeugen der Ausgabe. Diese Arbeitsteilung kann effizienter
sein als ein Decoder-only-Modell, das beides in einem einzigen Stapel von
Layern erledigen muss.

### Die Einschränkung

Encoder-Decoder-Modelle sind komplexer. Zwei separate Stapel von Layern.
Cross-Attention zwischen ihnen. Mehr Parameter für dieselbe Qualität bei
allgemeinen Sprachaufgaben. Die Architektur ist auf
Sequence-to-Sequence-Aufgaben spezialisiert und weniger flexibel für
offene Generierung.

Der Aufstieg der Decoder-only-Modelle hat die Popularität von
Encoder-Decoder-Architekturen verringert. Ein ausreichend großes
Decoder-only-Modell kann implizit die Trennung leisten, die ein
Encoder-Decoder-Modell explizit macht. GPT-3 hat das für Übersetzung und
Zusammenfassung gezeigt. Das Decoder-only-Modell lernte, die Eingabe zu
verstehen und die Ausgabe in einem einzigen Stapel von Layern zu
erzeugen.

## Warum dieser Guide Decoder-only lehrt

Decoder-only-Modelle sind das Fundament moderner KI. ChatGPT ist ein
Decoder-only-Modell. Claude ist ein Decoder-only-Modell. LLaMA und
Mistral sind Decoder-only-Modelle. Zu verstehen, wie sie funktionieren,
bedeutet, die Architektur hinter den fähigsten je gebauten KI-Systemen zu
verstehen.

Die Architektur ist zudem die einfachste. Ein Stapel von Blöcken. Ein
Attention-Muster mit Causal Masking. Ein Trainingsziel. Next-Token-
Prediction. Die Einfachheit macht sie zum besten Ausgangspunkt für das
Lernen. Sobald man den Decoder-only-Transformer verstanden hat, kann man
jede Transformer-Variante verstehen.

Encoder-only-Modelle werden weiterhin breit eingesetzt. BERT und seine
Varianten treiben Suchmaschinen, Klassifikationssysteme und Information
Retrieval an. Aber sie können keinen Text generieren. Sie zu verstehen
ist nützlich für spezialisierte Anwendungen, aber nicht essenziell für
den Bau generativer KI.

Encoder-Decoder-Modelle werden seltener. Der Leistungsunterschied
zwischen Decoder-only- und Encoder-Decoder-Modellen hat sich verringert.
Für die meisten praktischen Zwecke erreicht oder übertrifft ein großes
Decoder-only-Modell ein Encoder-Decoder-Modell bei derselben Aufgabe. Die
zusätzliche Komplexität lässt sich schwerer rechtfertigen.

## Wann man was verwendet

```
Müssen Sie neuen Text Token für Token generieren?
  → Decoder-only (GPT, LLaMA, Mistral)

Müssen Sie Text verstehen und ein Label oder eine Klassifikation erzeugen?
  → Encoder-only (BERT, RoBERTa, DeBERTa)

Müssen Sie Text von einer Form in eine andere transformieren und wollen
Sie die bestmögliche Qualität für eine bestimmte Aufgabe?
  → Encoder-Decoder (T5, BART)

Wollen Sie eine Architektur, die alles einigermaßen gut kann
und einfach zu verstehen und zu bauen ist?
  → Decoder-only
```

## Was Sie sich merken müssen

Decoder-only-Modelle erzeugen Text Token für Token, nur unter
Verwendung von vergangenem Kontext. Sie werden mit Next-Token-Prediction
trainiert. Das ist die GPT-Familie und das, was dieser gesamte Guide
lehrt.

Encoder-only-Modelle verstehen Text bidirektional unter Verwendung des
vollständigen Kontexts. Sie werden mit Masked Language Modeling
trainiert. Das ist die BERT-Familie. Sie können keinen Text generieren.

Encoder-Decoder-Modelle kombinieren beides. Ein Encoder liest die
Eingabe bidirektional. Ein Decoder erzeugt die Ausgabe autoregressiv mit
Cross-Attention zum Encoder. Das ist die T5-Familie. Sie sind für
Sequence-to-Sequence-Aufgaben gebaut.

Alle drei verwenden dieselben Bausteine. Attention. Feed-Forward-
Netzwerke. Residual Connections. Normalisierung. Die einzigen
Unterschiede sind das Attention-Mask-Muster und das Trainingsziel. Wer
die Decoder-only-Architektur beherrscht, hat das Fundament aller
modernen Sprachmodelle gemeistert.
