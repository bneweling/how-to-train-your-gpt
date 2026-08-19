# Speculative Decoding: Generierung 3× schneller machen

## Die kurze Antwort

Speculative Decoding nutzt zwei Modelle, um Text schneller zu
generieren, als es jedes der beiden Modelle allein könnte. Ein
kleines, schnelles Modell namens Draft Model generiert schnell
mehrere Kandidaten-Tokens. Ein großes, langsames Modell namens
Target Model prüft sie alle in einem einzigen Durchgang. Akzeptierte
Tokens werden übernommen. Abgelehnte Tokens werden vom großen
Modell neu generiert. Das Ergebnis ist Text, der identisch zu dem
ist, was das große Modell allein produziert hätte, aber zwei- bis
dreimal schneller generiert wurde.

Man kann es sich wie einen Senior-Engineer und einen Junior-Engineer
vorstellen. Der Junior schreibt schnell einen ersten Entwurf. Der
Senior prüft ihn und übernimmt die Teile, die korrekt sind. Die
Teile, die falsch sind, schreibt der Senior neu. Das fertige
Dokument hat die Qualität des Senior-Engineers, wird aber deutlich
schneller erstellt, weil der Senior nur redigieren und nicht von
Grund auf neu schreiben musste.

## Wo es einzuordnen ist

Speculative Decoding ist eine reine Inferenz-Technik. Sie verändert
nicht das Training. Sie verändert nicht die Modellgewichte. Sie
verändert nur die Generierungsschleife. Man kann sie auf jedes
autoregressive Modell anwenden, ohne es neu zu trainieren.

```
Ohne Speculative Decoding:
  Das große Modell generiert ein Token nach dem anderen
  Jedes Token erfordert einen vollständigen Forward Pass
  100 Tokens = 100 Forward Passes des großen Modells

Mit Speculative Decoding:
  Das kleine Modell entwirft K Tokens pro Schritt
  Das große Modell verifiziert K Tokens in einem Forward Pass
  Akzeptiert einige. Verwirft und regeneriert andere.
  100 Tokens ≈ 100/K Forward Passes des großen Modells
                + 100 Forward Passes des kleinen Modells
  Bei K=4 und einem 10× schnelleren kleinen Modell: ~2-3× Gesamt-Speedup
```

Der Speedup entsteht dadurch, dass Forward Passes des großen
Modells in Forward Passes des kleinen Modells umgewandelt werden.
Das kleine Modell ist deutlich schneller, sodass schon der Ersatz
einiger Forward Passes des großen Modells durch die des kleinen
Modells die Gesamtzeit reduziert.

## Wie es Schritt für Schritt funktioniert

### Schritt 1: die Draft-Phase

Das Target Model hat alles bis Position t generiert. Wir wollen das
nächste Token. Statt das große Modell laufen zu lassen, lassen wir
das kleine Modell K-mal hintereinander laufen, um K Kandidaten-Tokens
zu erzeugen.

```
Aktuelle Tokens: "The cat sat on"

Das Draft Model generiert:
  Schritt 1: "the"     (aus "The cat sat on")
  Schritt 2: "mat"     (aus "The cat sat on the")
  Schritt 3: "and"     (aus "The cat sat on the mat")
  Schritt 4: "then"    (aus "The cat sat on the mat and")

Draft-Vorschlag: ["the", "mat", "and", "then"]
```

Das Draft Model ist schnell. Die Generierung von vier Tokens mit
einem Modell, das 10 Prozent der Größe hat, dauert etwa 40 Prozent
der Zeit eines Forward Pass des großen Modells. Das Entwerfen von
vier Tokens kostet also weniger als die Hälfte eines Tokens des
großen Modells.

### Schritt 2: die Verifikationsphase

Das große Modell nimmt die ursprüngliche Sequenz plus die
Draft-Tokens und führt einen Forward Pass aus.

```
Eingabe für das große Modell: "The cat sat on the mat and then"

Das Modell verarbeitet alle vier neuen Tokens in einem einzigen
Forward Pass. Für jede Position berechnet es die Wahrscheinlichkeit
des Draft-Tokens, das an dieser Position stand.

Ausgabe: Wahrscheinlichkeit(korrekt für Position t+1)
        Wahrscheinlichkeit(korrekt für Position t+2)
        Wahrscheinlichkeit(korrekt für Position t+3)
        Wahrscheinlichkeit(korrekt für Position t+4)
```

Ein Forward Pass prüft vier Tokens. Wenn alle korrekt sind, haben
wir gerade drei Forward Passes des großen Modells eingespart.

### Schritt 3: die Akzeptanzphase

Für jede Position berechnet das große Modell eine Wahrscheinlichkeit
für das vorgeschlagene Draft-Token. Außerdem berechnet es seine
eigene Top-Vorhersage. Die Akzeptanzregel vergleicht die
Wahrscheinlichkeiten beider Modelle.

```
Für Position t+1 (wo der Draft "the" vorgeschlagen hat):
  Wahrscheinlichkeit Draft Model: 0.85
  Wahrscheinlichkeit Target Model: 0.82

  Da die Wahrscheinlichkeit des Target Model nahe an der des Draft
  Model liegt, wird das Token akzeptiert. Das bedeutet, die beiden
  Modelle stimmen überein. "the" bleibt.

Für Position t+2 (wo der Draft "mat" vorgeschlagen hat):
  Wahrscheinlichkeit Draft Model: 0.72
  Wahrscheinlichkeit Target Model: 0.35

  Das Target Model widerspricht deutlich. Das
  Wahrscheinlichkeitsverhältnis ist zu niedrig. Das Token wird
  verworfen. Alle Tokens nach dieser Position werden ebenfalls
  verworfen.

Ergebnis: "the" wird akzeptiert. "mat" "and" "then" werden
verworfen. Das Target Model generiert EIN neues Token an
Position t+2.
```

Die Akzeptanzregel basiert auf Wahrscheinlichkeitsverhältnissen.
Wenn die Wahrscheinlichkeit des Target Model für das Draft-Token
ähnlich hoch oder höher ist als die eigene Wahrscheinlichkeit des
Draft Model, wird das Token akzeptiert. Wenn die Wahrscheinlichkeit
des Target Model deutlich niedriger ist, wird das Token verworfen.

Das garantiert, dass die Ausgabeverteilung identisch zu der ist,
die das große Modell allein produziert hätte. Die Fehler des
kleinen Modells werden erkannt und korrigiert. Der finale Text hat
die Qualität des großen Modells.

### Schritt 4: weiter

Nach Akzeptanz und Ablehnung wurde die Sequenz um eine gewisse
Anzahl an Tokens erweitert. Akzeptierte Tokens bleiben erhalten.
Abgelehnte Positionen werden vom großen Modell aufgefüllt. Der
Prozess wiederholt sich ab dem neuen Ende der Sequenz.

```
Sequenz nach einem Zyklus: "The cat sat on the shelf"

Erneuter Draft: "and" "stared" "out" "the"
Verifizieren. "and" akzeptieren. "stared" und alles danach verwerfen.
Neu generieren: "looked"

Sequenz: "The cat sat on the shelf and looked"

Weiter, bis genügend Tokens generiert wurden.
```

## Die Mathematik der Akzeptanz

Das Besondere an Speculative Decoding ist, dass die Ausgabe
mathematisch identisch zu der ist, die das Target Model generiert
hätte. Das ist keine Näherung. Es ist exakt.

Der Akzeptanztest für jedes Draft-Token x an Position p lautet:

```
1. Wahrscheinlichkeit des Target Model berechnen:  P_target(x | context)
2. Wahrscheinlichkeit des Draft Model berechnen:   P_draft(x | context)
3. Akzeptieren, wenn: P_target(x) ≥ P_draft(x)
   Falls P_target(x) < P_draft(x): mit Wahrscheinlichkeit
   P_target(x) / P_draft(x) akzeptieren
```

Wenn das Target Model das Token für wahrscheinlicher hält als das
Draft Model, wird es immer akzeptiert. Wenn das Target Model es für
weniger wahrscheinlich hält, wird es proportional zum Verhältnis
akzeptiert. Dieses Sampling-Verfahren garantiert, dass die Ausgabe
exakt der Verteilung des Target Model folgt.

Der Beweis umfasst nur wenige Zeilen Wahrscheinlichkeitstheorie.
Aber man muss den Beweis nicht verstehen, um die Technik
einzusetzen. Die zentrale Erkenntnis ist, dass Speculative Decoding
exakt die Qualität des großen Modells liefert. Keine Näherung.
Keine Distillation. Exakt.

## Was ein gutes Draft Model ausmacht

Das Draft Model sollte deutlich kleiner als das Target Model sein,
aber denselben Tokenizer verwenden. Es sollte auf ähnlichen Daten
trainiert sein, damit seine Vorhersagen mit denen des Target Model
korrelieren.

Gute Draft Models:
- Eine kleinere Version derselben Architektur (7B-Draft für 70B-Target)
- Eine distillierte Version des Target Model
- Dasselbe Modell mit weniger Layern oder kleineren Dimensionen
- Ein völlig anderes, schnelles Modell mit demselben Tokenizer

Das Draft Model muss nicht von sich aus gut im Textgenerieren sein.
Es muss nur genug Tokens richtig treffen, damit die Akzeptanzrate
angemessen ist. Selbst bei 60 Prozent Akzeptanz ist der Speedup
erheblich, denn wenn drei von fünf Tokens akzeptiert werden,
bedeutet das drei eingesparte Forward Passes des großen Modells für
die Kosten von fünf Forward Passes des kleinen Modells.

```
Akzeptanzrate 50% bei K=5 Draft-Tokens:
Alt: 100 Forward Passes des großen Modells
Neu: 100/2.5 = 40 Forward Passes des großen Modells + 100 Forward Passes des kleinen Modells
Speedup: ~2.0x (wenn das kleine Modell 10% der Kosten des großen Modells verursacht)

Akzeptanzrate 80% bei K=5 Draft-Tokens:
Alt: 100 Forward Passes des großen Modells
Neu: 100/4 = 25 Forward Passes des großen Modells + 100 Forward Passes des kleinen Modells
Speedup: ~3.5x
```

## Eine vereinfachte Implementierung

```python
def speculative_generate(target_model, draft_model, tokenizer,
                         prompt, max_new_tokens=100, K=5):
    """
    Generiert Text mithilfe von Speculative Decoding.
    Die Ausgabe ist identisch zu target_model.generate(), aber schneller.
    """
    input_ids = tokenizer.encode(prompt)

    while len(input_ids) < max_new_tokens:
        # Phase 1: K Tokens mit dem kleinen Modell entwerfen
        draft_ids = input_ids.copy()
        draft_probs = []
        for _ in range(K):
            logits, _ = draft_model(draft_ids[-max_seq_len:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            draft_probs.append(probs[0, next_token].item())
            draft_ids.append(next_token)

        # Phase 2: Mit dem großen Modell in einem Pass verifizieren
        full_sequence = draft_ids[-max_seq_len:]
        target_logits, _ = target_model(full_sequence)
        target_probs = F.softmax(target_logits, dim=-1)

        # Phase 3: Akzeptieren oder verwerfen
        accepted = 0
        for i in range(K):
            pos = len(input_ids) - K + i
            draft_token = draft_ids[pos]
            target_prob = target_probs[0, pos, draft_token].item()
            draft_prob = draft_probs[i]

            if target_prob >= draft_prob:
                accepted += 1
            elif random.random() < target_prob / draft_prob:
                accepted += 1
            else:
                break

        # Akzeptierte Tokens behalten
        input_ids = draft_ids[:len(input_ids) - K + accepted]

        # Falls kein Token akzeptiert wurde, eines vom Target Model samplen
        if accepted == 0:
            target_probs = F.softmax(target_logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(target_probs, num_samples=1).item()
            input_ids.append(next_token)

    return tokenizer.decode(input_ids)
```

Das entscheidende Detail: Das Target Model verarbeitet alle K
Draft-Tokens in einem einzigen Forward Pass (Zeile 26). Genau daher
kommt der Speedup. Ein Forward Pass des Target Model prüft K
Draft-Tokens. Wenn die meisten akzeptiert werden, sparen wir K
minus eins Forward Passes des Target Model.

## Warum das in der Praxis wichtig ist

Ein Modell mit 70 Milliarden Parametern generiert auf einer
A100-GPU etwa 10 Tokens pro Sekunde. Ein Draft Model mit 7
Milliarden Parametern generiert etwa 100 Tokens pro Sekunde. Mit
K=4 und einer Akzeptanzrate von 70 Prozent generiert das
Speculative-Decoding-System etwa 25 Tokens pro Sekunde. Ein Speedup
von 2.5x ohne Qualitätsverlust.

Für eine Chat-Anwendung, bei der Nutzer Antworten in weniger als
einer Sekunde erwarten, ist das der Unterschied zwischen machbar
und frustrierend. Für eine Batch-Processing-Pipeline, die täglich
Millionen von Tokens generiert, ist das der Unterschied zwischen
einer GPU und drei.

## Das Zusammenspiel mit dem KV-Cache

Speculative Decoding funktioniert mit KV-Caches. Der Cache des
Target Model wird über die Verifikationsschritte hinweg gemeinsam
genutzt. Auch das Draft Model unterhält seinen eigenen Cache. Nach
der Akzeptanz wird der Cache des Target Model aktualisiert, sodass
er die akzeptierten Tokens enthält. Nach einer Ablehnung wird der
Cache des Target Model auf den Stand vor den abgelehnten Tokens
zurückgesetzt.

Das Cache-Management ist der heikelste Teil der Implementierung.
Fehlerhaftes Cache-Handling führt zu verfälschten Generierungen,
bei denen das Modell auf Tokens achtet, die es nicht sehen sollte,
oder Tokens nicht beachtet, die es sehen sollte. Die meisten
produktiven Speculative-Decoding-Implementierungen wenden mehr Code
für das Cache-Management auf als für die Akzeptanzlogik.

## Was man sich merken sollte

Speculative Decoding nutzt ein kleines, schnelles Draft Model, um
Tokens vorzuschlagen, und ein großes, langsames Target Model, um sie
zu verifizieren. Akzeptierte Tokens werden übernommen. Abgelehnte
Tokens werden neu generiert. Die Ausgabe ist mathematisch identisch
zu der des Target Model allein. Der Speedup entsteht dadurch, dass
teure Forward Passes des Target Model durch günstige Forward Passes
des Draft Model ersetzt werden.

Die Akzeptanzrate hängt davon ab, wie gut das Draft Model
vorhersagt, was das Target Model produziert hätte. Ein Draft Model
mit 10 Prozent der Größe des Target Model kann eine Akzeptanz von
70 bis 80 Prozent erreichen. Mit vier Kandidaten-Tokens pro Schritt
ergibt das einen Speedup von 2x bis 3x.

Speculative Decoding ist eine reine Inferenz-Optimierung. Kein
erneutes Training. Keine Änderungen an den Gewichten. Kein
Qualitäts-Tradeoff. Es ist kostenlose Geschwindigkeit für jedes
Model-Serving-System.
