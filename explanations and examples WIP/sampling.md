# Temperature, Top-K und Top-P: Textgenerierung steuern

## Was sie sind

Temperature, Top-K und Top-P sind drei Stellschrauben, mit denen
sich steuern lässt, wie ein Sprachmodell Text erzeugt. Sie
beeinflussen, wie das Modell das nächste Wort auswählt. Ohne sie
würde das Modell immer das einzelne wahrscheinlichste Wort wählen.
Die Ausgabe wäre langweilig und würde sich ständig wiederholen. Mit
diesen Stellschrauben lässt sich die Ausgabe fokussierter oder
kreativer gestalten. Man kann sie vorsichtig oder gewagt machen.

Man kann sich das Modell wie einen Koch vorstellen, der Zutaten
auswählt. Ohne jegliche Kontrolle greift der Koch für jedes Gericht
immer zur gängigsten Zutat. Jedes Hauptgericht ist Hähnchen. Jedes
Dessert ist Vanille. Temperature sagt dem Koch, hin und wieder auch
weniger gängige Zutaten auszuprobieren. Top-k beschränkt die
Vorratskammer auf nur die sinnvollsten Optionen. Top-p lässt den
Koch so lange Zutaten greifen, bis genug Vielfalt vorhanden ist,
und stoppt dann.

## Wo werden sie eingesetzt

Diese Stellschrauben werden direkt angewendet, nachdem das Modell
seine Rohwerte (Scores) erzeugt hat, und unmittelbar bevor es das
nächste Token auswählt.

```
Modell-Ausgabe (Logits für 50257 Tokens)
  → Durch Temperature teilen
    → Nur Top-k-Tokens behalten
      → Tokens behalten, bis die kumulierte Wahrscheinlichkeit Top-p überschreitet
        → Softmax zu Wahrscheinlichkeiten
          → Ein Token zufällig auswählen
```

Sie kommen ausschließlich während der Textgenerierung zum Einsatz.
Nicht während des Trainings. Während des Trainings sieht das Modell
immer die richtige Antwort. Während der Generierung gibt es keine
richtige Antwort. Das Modell muss den Raum möglicher nächster
Wörter erkunden. Diese Stellschrauben steuern, wie es das tut.

## Warum wir sie brauchen

Ohne jegliche Kontrolle macht das Modell nur eines: Es wählt das
Token mit der höchsten Wahrscheinlichkeit. Immer. Jedes Mal.

```
Prompt: "Die Katze saß auf der"
Modellvorhersage immer: "Matte"
Generiert: "Die Katze saß auf der Matte. Die Katze saß auf der Matte. Die Katze saß auf der Matte..."
```

Die Ausgabe dreht sich im Kreis. Sie bleibt in einer Schleife
hängen. Das passiert, weil der Pfad mit der höchsten
Wahrscheinlichkeit durch die Sprache oft eine Schleife ist. Sobald
das Modell *Die Katze saß auf der Matte* gesagt hat, ist die
wahrscheinlichste Fortsetzung wieder *Die Katze saß auf der Matte*.
Die Wahrscheinlichkeiten bilden eine Falle.

Die Stellschrauben durchbrechen diese Falle, indem sie
kontrollierten Zufall einführen. Statt immer das oberste Token zu
wählen, entscheidet sich das Modell manchmal für das zweit- oder
drittbeste. Die Ausgabe bleibt sinnvoll, vermeidet dabei aber
Wiederholungen.

## Temperature: Wie experimentierfreudig soll das Modell sein

Temperature ist eine einfache Division. Man nimmt die Rohwerte
(Scores) des Modells und teilt jeden einzelnen durch die
Temperature.

```
Niedrige Temperature (0.3):
  Die Scores werden verstärkt. Das oberste Token erhält noch mehr Wahrscheinlichkeit.
  Das Modell ist sicher und vorhersehbar.

  Prompt: "Die Hauptstadt von Frankreich ist"
  Ausgabe: "Paris, das in der Region Île-de-France liegt."

Hohe Temperature (1.5):
  Die Scores werden abgeflacht. Alle Tokens erhalten eher ähnliche Wahrscheinlichkeiten.
  Das Modell ist kreativ und unvorhersehbar.

  Prompt: "Die Hauptstadt von Frankreich ist"
  Ausgabe: "Paris, wo Baguettes davon träumen, Croissants zu werden."
```

Die Mathematik dahinter ist einfach. Hier ein winziges Beispiel mit
vier Kandidaten-Tokens.

```python
logits = [4.0, 2.0, 1.0, 0.5]

# Temperature 0.5 (fokussiert)
scaled = [4.0/0.5, 2.0/0.5, 1.0/0.5, 0.5/0.5]
       = [8.0, 4.0, 2.0, 1.0]
probs  = softmax([8.0, 4.0, 2.0, 1.0])
       = [0.97, 0.02, 0.01, 0.00]
# Token 0 hat eine Wahrscheinlichkeit von 97%. Sehr sicher.

# Temperature 2.0 (kreativ)
scaled = [4.0/2.0, 2.0/2.0, 1.0/2.0, 0.5/2.0]
       = [2.0, 1.0, 0.5, 0.25]
probs  = softmax([2.0, 1.0, 0.5, 0.25])
       = [0.48, 0.18, 0.18, 0.16]
# Token 0 hat nur eine Wahrscheinlichkeit von 48%. Viel breiter gestreut.
```

Eine Temperature von 0 bedeutet, immer das wahrscheinlichste Token
zu wählen. Das nennt man Greedy Decoding. Eine Temperature von 1
bedeutet, die natürlichen Wahrscheinlichkeiten ohne Veränderung zu
verwenden. Eine Temperature über 1 macht das Modell zufälliger.
Eine Temperature unter 1 macht das Modell fokussierter.

## Top-K: Nur die besten Optionen berücksichtigen

Temperature streut die Wahrscheinlichkeiten, aber selbst eine
winzige Wahrscheinlichkeit bleibt eine Chance für kompletten
Unsinn. Top-k setzt eine harte Grenze. Nur die k wahrscheinlichsten
Tokens werden berücksichtigt. Alles andere erhält die
Wahrscheinlichkeit null.

```python
# Nach der Temperature haben alle 50257 Tokens eine gewisse Wahrscheinlichkeit
# Mit top-k=50 behalten wir nur die 50 wahrscheinlichsten

v, _ = torch.topk(logits, 50)
logits[logits < v[:, -1:]] = float('-inf')
# Jetzt haben nur noch 50 Tokens eine Wahrscheinlichkeit ungleich null
```

Die magische Zahl ist oft 50. Das eliminiert wirklich unsinnige
Vervollständigungen und behält gleichzeitig genug Vielfalt für eine
interessante Ausgabe. Ein kleineres k wie 10 macht die Ausgabe
fokussierter. Ein größeres k wie 200 macht sie vielfältiger.

## Top-P: Dynamischer Grenzwert basierend auf Konfidenz

Top-k behält immer genau k Tokens. Doch die Konfidenz des Modells
variiert von Wort zu Wort. Manchmal ist sich das Modell sehr
sicher, und nur wenige Tokens sind sinnvoll. Manchmal ist sich das
Modell unsicher, und viele Tokens sind plausibel. Top-p passt sich
der jeweiligen Situation an.

Top-p, auch Nucleus Sampling genannt, behält die kleinste Menge an
Tokens, deren kumulierte Wahrscheinlichkeit p überschreitet.

```
Tokens nach Wahrscheinlichkeit sortiert:
[0.45, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

Top-p = 0.9:
  Kumuliert: 0.45 > behalten
  Kumuliert: 0.45 + 0.22 = 0.67 > behalten
  Kumuliert: 0.45 + 0.22 + 0.13 = 0.80 > behalten
  Kumuliert: 0.45 + 0.22 + 0.13 + 0.08 = 0.88 > behalten
  Kumuliert: 0.45 + 0.22 + 0.13 + 0.08 + 0.05 = 0.93 > stopp!
  Die ersten 5 Tokens behalten. Rest verwerfen.

Top-p = 0.5:
  Kumuliert: 0.45 > behalten
  Kumuliert: 0.45 + 0.22 = 0.67 > stopp!
  Die ersten 2 Tokens behalten.
```

Wenn sich das Modell sehr sicher ist, haben die obersten paar
Tokens vielleicht schon eine Gesamtwahrscheinlichkeit von 0.9.
Top-p behält dann nur diese wenigen. Wenn sich das Modell unsicher
ist, braucht es viel mehr Tokens, um 0.9 zu erreichen. Top-p behält
dann mehr Optionen. Dieses adaptive Verhalten ist der Grund, warum
Top-p oft gegenüber Top-k bevorzugt wird.

## Die empfohlene Kombination

Die meisten Produktivsysteme verwenden alle drei zusammen.

```python
logits = logits / temperature          # Schritt 1: Zufälligkeit steuern
logits = filter_top_k(logits, k=50)   # Schritt 2: Unsinn eliminieren
logits = filter_top_p(logits, p=0.9)  # Schritt 3: an Konfidenz anpassen
probs = softmax(logits)                # Schritt 4: in Wahrscheinlichkeiten umwandeln
next_token = sample(probs)            # Schritt 5: eines auswählen
```

Eine gängige Standardeinstellung, die für allgemeine Konversation
gut funktioniert, ist Temperature 0.7 mit Top-p 0.9 und Top-k 50.
Für sachliche Antworten senkt man die Temperature. Für kreatives
Schreiben erhöht man sie.

## Ein kleines Codebeispiel

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    # Temperature anwenden
    logits = logits / temperature

    # Top-k-Filterung
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = float('-inf')

    # Top-p-Filterung
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[mask] = float('-inf')

    # Sampling durchführen
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Test
logits = torch.tensor([[4.0, 2.0, 1.5, 0.8, 0.3, 0.1, 0.05, 0.02]])

print("Same prompt different temperatures:")
for temp in [0.3, 0.7, 1.5]:
    sampled = []
    for _ in range(5):
        t = sample_next_token(logits, temperature=temp, top_k=5)
        sampled.append(t.item())
    print(f"  T={temp}: samples={sampled}")
```

## Was man sich merken sollte

Temperature, Top-k und Top-p steuern, wie das Modell während der
Textgenerierung das nächste Token auswählt. Temperature passt die
Zufälligkeit der gesamten Verteilung an. Top-k behält nur die
besten k Optionen. Top-p passt die Anzahl der Optionen an die
Konfidenz des Modells an.

Ohne diese Kontrollen wäre die Textgenerierung deterministisch und
würde sich ständig wiederholen. Das Modell würde für immer in
denselben Phrasen kreisen. Mit diesen Kontrollen wird die
Generierung vielfältig und natürlich. Unterschiedliche
Temperature-Werte erzeugen unterschiedliche Schreibstile aus
demselben Modell. Deshalb kann dasselbe Sprachmodell sowohl
technische Dokumentation als auch Lyrik schreiben. Das Modell ist
dasselbe. Die Stellschrauben sind unterschiedlich.
