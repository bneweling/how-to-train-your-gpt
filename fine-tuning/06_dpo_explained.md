# DPO: Direct Preference Optimization

## Die kurze Antwort

DPO bringt einem Modell bei, gute Antworten gegenüber schlechten zu
bevorzugen. Statt wie RLHF ein separates Reward-Modell zu trainieren,
arbeitet DPO direkt mit Paaren von Antworten. Jedes Trainingsbeispiel
zeigt dem Modell für denselben Prompt eine bevorzugte (chosen) und eine
abgelehnte (rejected) Antwort. Das Modell lernt, die Wahrscheinlichkeit
der bevorzugten Antwort zu erhöhen und die Wahrscheinlichkeit der
abgelehnten Antwort zu senken. Kein Reward-Modell nötig. Kein
Reinforcement Learning nötig. Nur ein Dataset mit Präferenzen und eine
kleine Anpassung der Trainings-Loss.

## Wo es einzuordnen ist

DPO ist eine Fine-Tuning-Methode. Sie nimmt ein Basis-Chat-Modell, das
bereits weiß, wie man Anweisungen befolgt, und verbessert die Qualität
seiner Antworten. RLHF kam zuerst. DPO kam später und ist einfacher.
Beide verfolgen dasselbe Ziel: die Ausgaben des Modells besser mit dem
in Einklang zu bringen, was Menschen wollen.

```
Basismodell (kennt Sprache)
  → Instruction Tuning (kann Anweisungen befolgen)
    → DPO oder RLHF (weiß, welche Antworten besser sind)
      → Aligniertes Modell (hilfreich, harmlos, ehrlich)
```

## Warum DPO RLHF für die meisten Teams abgelöst hat

RLHF besteht aus drei Schritten. Ein Reward-Modell auf menschlichen
Präferenzdaten trainieren. Das Reward-Modell nutzen, um die Ausgaben des
Sprachmodells zu bewerten. Das Sprachmodell mittels Reinforcement
Learning aktualisieren, um den Reward zu maximieren. Jeder Schritt ist
komplex und fehleranfällig. Das Reward-Modell lässt sich austricksen.
Das RL-Training kann instabil werden. Der gesamte Prozess erfordert
ständige Überwachung.

DPO hat einen einzigen Schritt. Das Sprachmodell wird direkt auf den
Präferenzdaten mit einer modifizierten Loss-Funktion trainiert. Die Loss
ermutigt das Modell, die Wahrscheinlichkeit bevorzugter Antworten
relativ zu abgelehnten Antworten zu erhöhen. Kein separates
Reward-Modell. Kein RL-Algorithmus. Keine Instabilität. Die Trainings-
Loop ist dieselbe wie beim Standard-Fine-Tuning. Nur die Loss-Funktion
ändert sich.

## Wie DPO funktioniert

DPO vergleicht zwei Antworten auf denselben Prompt. Eine wurde von einem
Menschen bevorzugt. Eine wurde abgelehnt. Das Modell sieht beide und
passt seine Weights so an, dass die bevorzugte Antwort wahrscheinlicher
und die abgelehnte Antwort unwahrscheinlicher wird.

Die DPO-Loss ist eine modifizierte Cross-Entropy-Loss.

```
Gegeben:
  prompt: "Erkläre einem Kind die Schwerkraft"
  chosen: "Schwerkraft ist die Kraft, die Dinge zueinander zieht."
  rejected: "Schwerkraft ist eine fundamentale Wechselwirkung."

Die DPO-Loss:
  log_ratio = log(P(chosen | prompt)) - log(P(rejected | prompt))
  loss = -log(sigmoid(beta * log_ratio))
```

Der Beta-Parameter steuert, wie stark das Modell in Richtung der
bevorzugten Antworten gedrängt wird. Höheres Beta bedeutet ein stärkeres
Präferenzsignal. Niedrigeres Beta ist konservativer. Typische
Beta-Werte liegen zwischen 0.1 und 0.5.

Das Referenzmodell ist das Modell vor dem DPO-Training. DPO vergleicht
die Wahrscheinlichkeiten des aktuellen Modells mit den
Wahrscheinlichkeiten des Referenzmodells. Das verhindert, dass sich das
Modell zu weit von seinem ursprünglichen Verhalten entfernt. Ohne das
Referenzmodell könnte das Modell lernen, die bevorzugten Antworten
wortwörtlich zu wiederholen. Mit dem Referenzmodell lernt es das
allgemeine Muster dessen, was eine gute Antwort ausmacht.

```
Vollständige DPO-Loss mit Referenzmodell:

log_ratio_current = log(P_current(chosen | prompt)) - log(P_current(rejected | prompt))
log_ratio_ref = log(P_ref(chosen | prompt)) - log(P_ref(rejected | prompt))
loss = -log(sigmoid(beta * (log_ratio_current - log_ratio_ref)))
```

Die Subtraktion des Log-Ratios des Referenzmodells ist das, was DPO
funktionieren lässt. Sie stellt die Frage: Bevorzugt das aktuelle Modell
die bevorzugte Antwort STÄRKER als das Referenzmodell es tat. Wenn ja,
ist die Loss klein. Wenn nein, ist die Loss groß und das Modell wird
aktualisiert.

## Das Datenformat

Die DPO-Trainingsdaten sind einfach. Jedes Beispiel besteht aus einem
Prompt, einer bevorzugten Antwort und einer abgelehnten Antwort.

```json
{
  "prompt": "Explain gravity to a child.",
  "chosen": "Gravity is what keeps your feet on the ground. It pulls everything toward the Earth.",
  "rejected": "Gravity is a fundamental interaction. Newton described it mathematically."
}
```

Die bevorzugte Antwort ist besser. Vielleicht verwendet sie einfachere
Sprache. Vielleicht ist sie vollständiger. Vielleicht vermeidet sie
sachliche Fehler. Die abgelehnte Antwort ist schlechter. Vielleicht
verwendet sie Fachjargon. Vielleicht ist sie zu knapp. Vielleicht
enthält sie einen Fehler. Das Modell lernt aus dem Unterschied.

Die Qualität des DPO-Trainings hängt vollständig von der Qualität der
Präferenzpaare ab. Wenn die abgelehnte Antwort tatsächlich besser ist
als die bevorzugte, lernt das Modell die falsche Lektion. Jedes
Präferenzpaar muss sorgfältig kuratiert werden. Qualitativ hochwertige
Daten sind teuer in der Erstellung, aber für gute Ergebnisse
unerlässlich.

## Ein kleines Codebeispiel

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_model, reference_model, prompt_ids,
             chosen_ids, rejected_ids, beta=0.1):
    """
    Berechnet die DPO-Loss für ein einzelnes Präferenzpaar.
    policy_model ist das Modell, das trainiert wird.
    reference_model ist eingefroren und repräsentiert das Verhalten vor DPO.
    """
    with torch.no_grad():
        ref_chosen_logp = reference_model(chosen_ids)[0].log_softmax(-1).sum()
        ref_rejected_logp = reference_model(rejected_ids)[0].log_softmax(-1).sum()
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp

    policy_chosen_logp = policy_model(chosen_ids)[0].log_softmax(-1).sum()
    policy_rejected_logp = policy_model(rejected_ids)[0].log_softmax(-1).sum()
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp

    logits = policy_log_ratio - ref_log_ratio
    loss = -F.logsigmoid(beta * logits)

    return loss


# Beispielhafte Verwendung (konzeptionell)
preference_pairs = [
    {
        "prompt": "Explain gravity to a child.",
        "chosen": "Gravity is what keeps your feet on the ground.",
        "rejected": "Gravity is a fundamental physical interaction.",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris. It is known for the Eiffel Tower.",
        "rejected": "Paris.",
    },
]

policy_model = ...  # Dein Chat-Modell
reference_model = ...  # Eingefrorene Kopie desselben Modells

for pair in preference_pairs:
    prompt_ids = tokenize(pair["prompt"])
    chosen_ids = tokenize(pair["chosen"])
    rejected_ids = tokenize(pair["rejected"])

    loss = dpo_loss(policy_model, reference_model, prompt_ids,
                    chosen_ids, rejected_ids, beta=0.1)
    loss.backward()
    optimizer.step()
```

## DPO versus RLHF

| Aspekt | RLHF | DPO |
|---|---|---|
| Schritte | 3 (Reward-Modell + PPO + KL-Penalty) | 1 (direkte Loss) |
| Stabilität | Kann instabil sein. Erfordert sorgfältiges Tuning | Stabil. Standard-Supervised-Learning |
| Rechenaufwand | Höher. Reward-Modell muss mitlaufen | Niedriger. Nur das Policy-Modell |
| Code-Komplexität | Hoch. PPO-Implementierung ist tricky | Niedrig. Modifizierte Loss-Funktion |
| Daten | Dieselben Präferenzpaare | Dieselben Präferenzpaare |
| Qualität | Etabliert. Wird von ChatGPT verwendet | In vielen Benchmarks vergleichbar |

DPO hat RLHF in der Open-Source-Community weitgehend abgelöst. Es ist
einfacher zu implementieren und erzielt in den meisten Benchmarks
vergleichbare Ergebnisse. Der Hauptvorteil von RLHF ist, dass es aus
Online-Feedback lernen kann, bei dem das Modell Antworten generiert und
Menschen sie in Echtzeit bewerten. DPO benötigt vorab gesammelte
Präferenzpaare. Für Offline-Datasets schneiden beide Methoden ähnlich
ab.

## Wann DPO eingesetzt werden sollte

Verwende DPO, wenn du Zugang zu menschlichen Präferenzdaten hast und die
Qualität der Antworten eines Modells über das hinaus verbessern
möchtest, was Instruction Tuning allein erreichen kann. Das Modell muss
bereits Anweisungen befolgen können. DPO kann einem Basismodell nicht
beibringen zu chatten. Es kann nur die Qualität eines Modells
verbessern, das bereits chattet.

Verwende DPO, wenn du schädliche Ausgaben reduzieren oder die
Faktentreue verbessern oder Antworten prägnanter gestalten oder an
bestimmte Style-Guidelines anpassen möchtest. Jede Qualitätsdimension,
die sich als Präferenz zwischen zwei Antworten ausdrücken lässt, kann
mit DPO optimiert werden.

## Was du dir merken solltest

DPO trainiert ein Modell direkt auf Präferenzpaaren, ohne ein separates
Reward-Modell. Die Loss-Funktion ermutigt das Modell, bevorzugte
Antworten gegenüber abgelehnten zu bevorzugen. Ein Referenzmodell
verhindert, dass sich das Modell zu weit von seinem ursprünglichen
Verhalten entfernt. DPO ist einfacher und stabiler als RLHF und erzielt
vergleichbare Ergebnisse. Es ist die Standardmethode zur
Präferenzoptimierung in der Open-Source-Community.
