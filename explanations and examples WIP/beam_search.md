# Beam Search: Generierung präziser machen

## Kurz gesagt

Beam Search hält bei jedem Schritt mehrere Kandidaten-Sequenzen
gleichzeitig am Leben, anstatt nur ein einzelnes Token auszuwählen. An
jeder Position werden die K besten Möglichkeiten betrachtet, die
bisherige Sequenz fortzusetzen. Die Beam-Breite K liegt typischerweise
bei 3 bis 5. Das Ergebnis ist präziser, aber weniger kreativer Text.
Beam Search ist der Standard für Übersetzung und Zusammenfassung, wo
die richtige Antwort wichtiger ist als Originalität.

Man kann es sich wie Einparken vorstellen. Greedy Search nimmt die
erste verfügbare Parklücke und parkt dort ein. Sampling könnte
irgendwo auf dem Parkplatz parken. Beam Search setzt zurück und
probiert mehrere Einparkversuche gleichzeitig aus und wählt am Ende
die beste Endposition.

## Wo es ansetzt

Beam Search ersetzt das Token-für-Token-Sampling in der
Generierungsschleife. Es verändert weder das Modell noch das Training.
Es verändert nur, wie die Vorhersagen des Modells genutzt werden, um
die Ausgabesequenz aufzubauen.

```
Generierungsstrategien:
  Greedy:   Wählt das einzelne Token mit der höchsten Wahrscheinlichkeit. Schnell. Repetitiv.
  Sampling: Wählt zufällig aus der Wahrscheinlichkeitsverteilung. Kreativ. Variabel.
  Beam:     Hält K Sequenzen. Prüft alle K×V nächsten Tokens. Behält die besten K.
            Präzise. Deterministisch. Langsamer.
```

## Wie es Schritt für Schritt funktioniert

Stellen wir uns vor, wir bitten das Modell, *The cat sat on the* mit
einer Beam-Breite von 3 zu vervollständigen.

### Schritt 1: Start

Wir haben eine leere Sequenz. Der Beam enthält ein einziges Element,
nämlich den Prompt. Wir wollen 4 weitere Tokens generieren.

```
Beam (Größe 1): ["The cat sat on the"]
```

### Schritt 2: Erstes Token

Das Modell verarbeitet den Prompt und erzeugt Logits für das nächste
Token. Anstatt nur eines auszuwählen, betrachten wir die Top 3.

```
Wahrscheinlichste Tokens und ihre Wahrscheinlichkeiten:
  "mat"   : logit = 4.2, prob = 0.45
  "floor" : logit = 3.1, prob = 0.22
  "table" : logit = 2.5, prob = 0.12
  "chair" : logit = 1.8, prob = 0.08
  "rug"   : logit = 1.2, prob = 0.05
  ... (insgesamt 50257)
```

Wir erweitern den Beam um die Top 3.

```
Beam (Größe 3):
  Sequenz A: "The cat sat on the mat"     (score = 0.45)
  Sequenz B: "The cat sat on the floor"   (score = 0.22)
  Sequenz C: "The cat sat on the table"   (score = 0.12)
```

### Schritt 3: Zweites Token

Jetzt fragt jede der 3 Sequenzen das Modell nach ihrem nächsten Token.
Für jede Sequenz erzeugt das Modell 50257 Wahrscheinlichkeiten. Wir
kombinieren jeden Kandidaten mit dem kumulativen Score seiner
übergeordneten Sequenz.

```
Für Sequenz A ("... mat"):
  "and"    : prob = 0.35  → cumulative = 0.45 × 0.35 = 0.158
  ","      : prob = 0.25  → cumulative = 0.45 × 0.25 = 0.113
  "because": prob = 0.18  → cumulative = 0.45 × 0.18 = 0.081
  ...

Für Sequenz B ("... floor"):
  "and"    : prob = 0.30  → cumulative = 0.22 × 0.30 = 0.066
  ","      : prob = 0.20  → cumulative = 0.22 × 0.20 = 0.044
  ...

Für Sequenz C ("... table"):
  "and"    : prob = 0.28  → cumulative = 0.12 × 0.28 = 0.034
  ...
```

Wir haben 3 × 50257 Kandidaten. Wir wählen die Top 3 nach kumulativem
Score aus.

```
Beam (Größe 3):
  "The cat sat on the mat and"      (score = 0.158)
  "The cat sat on the mat,"         (score = 0.113)
  "The cat sat on the mat because"  (score = 0.081)
```

Eine Beobachtung: Alle drei Top-Sequenzen beginnen mit *mat*. Der Beam
ist auf ein einziges Präfix konvergiert. Das ist bei Beam Search
üblich. Die Top-Kandidaten teilen sich oft ein Präfix. Der Beam
besteht nicht aus drei unabhängigen Suchen. Es ist eine einzige Suche,
die drei Kandidaten-Fortsetzungen gleichzeitig verfolgt.

### Schritt 4 und darüber hinaus

Wir fahren fort, bis wir die maximale Länge erreichen oder alle
Sequenzen mit dem End-of-Text-Token enden. Am Ende wählen wir die
Sequenz mit dem höchsten kumulativen Score.

```
Finaler Beam (nach 4 Tokens):
  "The cat sat on the mat and then"          (score = 0.098)
  "The cat sat on the mat and waited"        (score = 0.072)
  "The cat sat on the mat and the"           (score = 0.054)

Bestes Ergebnis:
  "The cat sat on the mat and then"
```

## Warum wir Log-Wahrscheinlichkeiten verwenden

Man beachte, dass sich die Scores bei jedem Schritt multiplizieren.
0.45 mal 0.35 ergibt 0.158. Bei Schritt 10 könnte der kumulative Score
bei 0.000001 liegen. Diese winzigen Zahlen sind schwer zu vergleichen,
weil die Fließkommagenauigkeit begrenzt ist.

Die Lösung nutzt Log-Wahrscheinlichkeiten. Statt zu multiplizieren,
addieren wir. Das hält alle Zahlen in einem handhabbaren Bereich.

```
Cumulative score = prob_1 × prob_2 × prob_3 × ...
Cumulative log score = log(prob_1) + log(prob_2) + log(prob_3) + ...

Für mat → and:
  log score = log(0.45) + log(0.35) = -0.798 + (-1.050) = -1.848
  Score = e^(-1.848) = 0.158  (wie zuvor)
```

Da die Log-Wahrscheinlichkeiten immer negativ sind, wird der
kumulative Score bei jedem Schritt negativer. Das bedeutet, kürzere
Sequenzen haben höhere Scores, einfach weil weniger negative Terme
addiert wurden. Beam Search bevorzugt von Natur aus kürzere Ausgaben,
sofern wir nicht normalisieren.

Die Lösung ist Längennormalisierung. Man teilt den kumulativen
Log-Score durch die Anzahl der generierten Tokens.

```
Normalized score = cumulative_log_score / length^α

α = 0: keine Normalisierung (bevorzugt kurze Sequenzen)
α = 1: vollständige Normalisierung (bevorzugt lange Sequenzen)
α = 0.6 bis 0.8: Standard (ausgewogen)
```

Ohne Normalisierung könnte das Modell bereits nach einem Token einen
Punkt ausgeben, weil jede Fortsetzung den Score verschlechtert. Mit
Normalisierung wird das Modell dazu ermutigt, vollständige Antworten
zu erzeugen.

## Beam Search versus Greedy versus Sampling

```
Prompt: "The cat sat on the"

Greedy (beam=1):
  "The cat sat on the mat and then the cat sat on the mat and then..."

Sampling (T=0.8, top_k=50):
  "The cat sat on the windowsill watching birds flutter past the garden."

Beam search (beam=5):
  "The cat sat on the mat and waited patiently for its owner to come home."
```

Greedy wiederholt sich, weil es immer das einzelne wahrscheinlichste
Token wählt. Die wahrscheinlichste Fortsetzung nach *the cat sat on
the mat* ist wieder *the cat sat on the mat*. Aus dieser
Wahrscheinlichkeitsfalle gibt es kein Entkommen.

Sampling entkommt der Falle, indem es gelegentlich weniger
wahrscheinliche Tokens wählt. Das Ergebnis ist interessanter, aber
manchmal unsinnig. Die Qualität variiert von Generierung zu
Generierung.

Beam Search erkundet mehrere Pfade und wählt die beste vollständige
Sequenz. Es vermeidet die Wiederholungsfalle, weil *and waited
patiently* eine höhere kumulative Wahrscheinlichkeit haben könnte als
die Rückkehr zu *the*. Das Ergebnis ist präzise und kohärent, aber
weniger kreativ.

## Eine vereinfachte Beam-Search-Implementierung

```python
import torch
import torch.nn.functional as F

def beam_search_generate(model, input_ids, max_new_tokens, beam_width=5,
                         length_penalty=0.7, temperature=1.0):
    """
    Generiert Text mittels Beam Search.
    Gibt die einzelne beste Sequenz zurück.
    """
    model.eval()
    device = input_ids.device

    # Jeder Beam ist eine Sequenz. Wir halten beam_width Sequenzen.
    beams = [(input_ids.clone(), 0.0)]  # (Sequenz, kumulativer Log-Score)
    finished_beams = []

    for step in range(max_new_tokens):
        candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                finished_beams.append((seq, score / (len(seq[0]) ** length_penalty)))
                continue

            with torch.no_grad():
                logits, _ = model(seq[:, -max_seq_len:])
                logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(
                    logits, beam_width * 2, dim=-1
                )
                log_probs = F.log_softmax(top_k_logits, dim=-1)

            for j in range(beam_width * 2):
                token = top_k_indices[0, j:j+1]
                new_seq = torch.cat([seq, token.unsqueeze(0)], dim=1)
                new_score = score + log_probs[0, j].item()
                candidates.append((new_seq, new_score))

        # Behalte die besten beam_width Kandidaten nach Score
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if len(finished_beams) >= beam_width:
            break

    # Verbleibende Beams als abgeschlossen hinzufügen
    for seq, score in beams:
        finished_beams.append(
            (seq, score / (len(seq[0]) ** length_penalty))
        )

    # Bestes abgeschlossenes Beam auswählen
    finished_beams.sort(key=lambda x: x[1], reverse=True)
    return finished_beams[0][0]


# Verwendung
prompt_ids = tokenizer.encode("The cat sat on the", return_tensors="pt")
output_ids = beam_search_generate(model, prompt_ids, max_new_tokens=30, beam_width=5)
text = tokenizer.decode(output_ids[0])
```

Das entscheidende Detail steht in Zeile 33. Wir betrachten
`beam_width × 2` Kandidaten pro Sequenz, nicht nur `beam_width`. Der
Grund ist, dass sich manche Kandidaten ein Präfix teilen und im
nächsten Schritt verschmelzen würden. Doppelt so viele Kandidaten pro
Ausgangssequenz zu betrachten gibt dem Beam mehr Vielfalt zur Auswahl.

## Wann welche Strategie einzusetzen ist

```
Greedy (beam=1):     Übersetzung. Codegenerierung. Aufgaben, bei denen
                     die Ausgabe die einzelne wahrscheinlichste Antwort
                     sein soll. Schnell, kann aber in Schleifen hängen.

Beam search (3-5):   Zusammenfassung. Bildbeschreibung. Spracherkennung.
                     Aufgaben, bei denen Genauigkeit wichtiger ist als
                     Kreativität. Gut für strukturierte Ausgaben mit
                     klar richtigen Antworten.

Sampling (T=0.8):    Kreatives Schreiben. Dialoge. Story-Generierung.
                     Aufgaben, bei denen Vielfalt und Interessantheit
                     zählen. Jedes Mal anders. Weniger repetitiv als
                     Greedy.

Beam + sampling:     Manche Produktivsysteme kombinieren Beam Search
                     mit leichtem Sampling, um etwas Abwechslung
                     hinzuzufügen und gleichzeitig die Qualität zu
                     erhalten. Selten nötig.
```

Beam Search ist etwa um den Faktor der Beam-Breite langsamer als
Greedy. Eine Beam-Breite von 5 bedeutet 5 Forward-Passes pro Schritt
statt 1. Mit KV-Caching lässt sich dieser Mehraufwand reduzieren, weil
die Beams sich dasselbe Präfix teilen und die Keys und Values für das
gemeinsame Präfix gecacht werden.

## Das Wiederholungsproblem

Beam Search kann trotzdem repetitive Ausgaben erzeugen. Der Beam
konvergiert auf ein einziges Präfix und beginnt dann, sich zu
wiederholen. Die Wiederholung ist subtiler als beim Greedy-Decoding,
aber weiterhin vorhanden.

Zu den Lösungen gehören n-Gram-Blocking, das verhindert, dass dasselbe
n-Gram zweimal in einem Beam auftaucht, und Repetition Penalty, das
die Wahrscheinlichkeit bereits aufgetretener Tokens verringert. Diese
werden zusätzlich zu Beam Search angewendet und sind in
Produktivsystemen Standard.

## Was man sich merken sollte

Beam Search hält mehrere Kandidaten-Sequenzen gleichzeitig und wählt
am Ende die beste aus. Es ist präziser als Greedy-Decoding und
konsistenter als Sampling. Es nutzt Log-Wahrscheinlichkeiten, um
Fließkomma-Unterlauf zu vermeiden, und Längennormalisierung, um kurze
Sequenzen nicht zu bevorzugen.

Eine Beam-Breite von 5 ist für die meisten Aufgaben Standard. Höhere
Beam-Breiten sind langsamer und bringen abnehmenden Zusatznutzen.
Niedrigere Beam-Breiten nähern sich dem Verhalten von Greedy an.

Beam Search eignet sich am besten für Aufgaben mit einer klar
richtigen Antwort, etwa Übersetzung und Zusammenfassung. Für kreatives
Schreiben, bei dem Vielfalt wichtiger ist als Genauigkeit, ist es
nicht ideal.
