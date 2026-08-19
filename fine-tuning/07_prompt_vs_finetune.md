# Prompt Engineering versus Fine-Tuning

## Die kurze Antwort

Sowohl Prompt Engineering als auch Fine-Tuning verändern, wie sich ein
Modell verhält. Der Unterschied liegt darin, wo diese Veränderung
stattfindet. Prompt Engineering verändert den Input. Fine-Tuning
verändert das Modell selbst. Wähle Prompt Engineering, wenn du schnelle
Ergebnisse für wenige Anwendungsfälle brauchst. Wähle Fine-Tuning, wenn
du konsistentes Verhalten in großem Maßstab brauchst.

## Wo sie ansetzen

```
Roher Nutzer-Input
  → Prompt Engineering fügt Kontext und Anweisungen hinzu
    → Basis- oder fine-getuntes Modell
      → Antwort
```

Prompt Engineering verpackt die Anfrage des Nutzers in zusätzlichen
Text, der das Modell lenkt. Die Modell-Weights ändern sich dabei nie.
Jede Anfrage erhält dieselbe Prompt-Vorlage. Das Modell interpretiert
die Vorlage und antwortet entsprechend.

Fine-Tuning bäckt die Anleitung direkt in die Weights des Modells ein.
Der Prompt kann einfach bleiben, weil das Modell bereits weiß, wie es
sich verhalten soll. Das Verhalten ist konsistent, weil es in den
Weights kodiert ist und nicht im Prompt-Text.

## Wann man Prompt Engineering einsetzt

Prompt Engineering ist das Erste, was man ausprobieren sollte. Es ist
kostenlos. Es dauert Minuten. Es funktioniert mit jedem Modell über
jede API.

Setze Prompt Engineering ein, wenn:
- Du wenige gut definierte Anwendungsfälle hast
- Sich das gewünschte Verhalten in wenigen Sätzen beschreiben lässt
- Du keine perfekte Konsistenz über Hunderte von Variationen hinweg brauchst
- Du prototypisch arbeitest und schnell iterierst
- Du eine gehostete API nutzt und das Modell nicht verändern kannst

Prompt Engineering kann bemerkenswerte Ergebnisse erzielen. Ein gut
geschriebener System-Prompt kann ein Basis-Chat-Modell dazu bringen,
sich wie ein Fachexperte für medizinische Fragen, ein Coach für
kreatives Schreiben oder ein Code-Reviewer zu verhalten. Die Technik
wird besser, je besser die Basismodelle werden. Jede Modellgeneration
braucht weniger Prompting, um dieselben Ergebnisse zu erzielen.

```python
# Beispiel für Prompt Engineering
system_prompt = """
You are a helpful medical assistant. Always:
- Use simple language a patient can understand
- Never diagnose. Suggest seeing a doctor instead
- Cite reliable sources when possible
- Ask clarifying questions if symptoms are vague
"""

user_question = "My head hurts and I feel dizzy."

full_prompt = f"{system_prompt}\n\nPatient: {user_question}\nAssistant:"
response = model.generate(full_prompt)
```

Der Prompt erledigt die ganze Arbeit. Das Modell bleibt unverändert.
Für unterschiedliche Anwendungsfälle können unterschiedliche Prompts
verwendet werden. Ein medizinischer Prompt für Gesundheitsfragen. Ein
juristischer Prompt für die Vertragsprüfung. Ein kreativer Prompt für
die Geschichten-Generierung. Alles läuft auf demselben Basismodell.

## Wann man fine-tunt

Prompt Engineering stößt an Grenzen. Lange Prompts verbrauchen Platz im
Context Window. Komplexes Verhalten lässt sich schwer in Worten
beschreiben. Manche Muster lassen sich leichter vorführen als erklären.
Prompt-Injection-Angriffe können System-Prompts überschreiben.
Konsistenz über Tausende von Anfragevariationen hinweg ist schwer zu
garantieren.

Fine-Tuning begegnet diesen Grenzen. Das Modell lernt das gewünschte
Verhalten anhand von Beispielen. Das Context Window bleibt frei für den
eigentlichen Nutzer-Input. Das Verhalten ist in den Weights kodiert und
kann nicht durch eine geschickt formulierte Nutzernachricht überschrieben
werden. Die Konsistenz entsteht durch Tausende von Trainingsbeispielen,
die jeden Grenzfall abdecken.

Setze Fine-Tuning ein, wenn:
- Du Hunderte oder Tausende Beispiele des gewünschten Verhaltens hast
- Das Verhalten zu komplex ist, um es in einem Prompt zu beschreiben
- Du das Context Window für Nutzer-Input statt für Anweisungen brauchst
- Du konsistentes Verhalten brauchst, das nicht per Prompt Injection unterlaufen werden kann
- Du das Modell in großem Maßstab produktiv betreibst

```python
# Beispiel für Fine-Tuning mit LoRA
training_data = [
    {"symptom": "headache and dizziness", "response": "These symptoms can have many causes..."},
    {"symptom": "chest pain", "response": "Chest pain should be evaluated by a doctor immediately..."},
    {"symptom": "sore throat", "response": "A sore throat is often caused by a viral infection..."},
    # ... Hunderte weitere Beispiele
]

model = load_base_model()
model = add_lora(model, rank=16)

for example in training_data:
    prompt = f"Patient: {example['symptom']}\nAssistant:"
    response = example['response']
    loss = compute_loss(model, prompt, response)
    loss.backward()

# Jetzt antwortet das Modell medizinisch, ganz ohne Prompt-Anweisungen
response = model.generate("Patient: My head hurts and I feel dizzy.\nAssistant:")
```

Nach dem Fine-Tuning erzeugt das Modell medizinische Antworten, ohne
einen System-Prompt zu benötigen. Das Verhalten steckt in den Weights.
Das Context Window ist frei für die detaillierte Beschreibung der
Symptome durch den Patienten.

## Der Kostenvergleich

| Aspekt | Prompt Engineering | Fine-Tuning |
|---|---|---|
| Umsetzungszeit | Minuten | Stunden bis Tage |
| Kosten | Kostenlos (API-Kosten pro Anfrage) | GPU-Zeit + Datensammlung |
| Benötigte Expertise | Schreibfähigkeiten | ML-Engineering-Kenntnisse |
| Konsistenz | Variiert je nach Prompt-Formulierung | Konsistent über Variationen hinweg |
| Genutztes Context Window | 200 bis 2000 Token für den Prompt | 0 Token für Anweisungen |
| Anfälligkeit für Injection | Hoch | Niedrig |
| Iterationsgeschwindigkeit | Sofort | Stunden pro Experiment |
| Modellabhängigkeit | Muss für jedes Modell neu gemacht werden | Adapter lässt sich zwischen Modellen übertragen |

## Der hybride Ansatz

Die meisten Produktivsysteme nutzen beides. Fine-Tuning bringt dem
Modell das Kernverhalten bei. Prompt Engineering übernimmt die
situationsspezifischen Details, die sich mit jeder Anfrage ändern.

```
Fine-Tuning: bringt dem Modell bei, ein medizinischer Assistent zu sein
Prompt: fügt patientenspezifischen Kontext, aktuelle Laborwerte, Medikamentenliste hinzu

Fine-Tuning: bringt dem Modell bei, Python-Code zu reviewen
Prompt: fügt den konkreten zu prüfenden Code, Coding-Standards, Kontext hinzu

Fine-Tuning: bringt dem Modell bei, in einer Markenstimme zu schreiben
Prompt: fügt das konkrete Thema, die Zielgruppe, die gewünschte Länge hinzu
```

Das fine-getunte Modell liefert das Fundament. Der Prompt liefert die
Details. Zusammen sind sie wirkungsvoller als jedes für sich allein.

## Der Entscheidungsablauf

```
Hast du wenige Anwendungsfälle und brauchst noch heute Ergebnisse?
  → Prompt Engineering

Brauchst du konsistentes Verhalten über Tausende von Nutzereingaben hinweg?
  → Fine-Tuning

Lässt sich das gewünschte Verhalten klar in Worten beschreiben?
  → Prompt Engineering

Ist das Verhalten schwer zu beschreiben, aber leicht mit Beispielen zu demonstrieren?
  → Fine-Tuning

Nutzt du eine gehostete API und kannst das Modell nicht verändern?
  → Prompt Engineering

Hast du einen Datensatz guter Antworten für deinen Anwendungsfall?
  → Fine-Tuning

Ist jede Anfrage einzigartig mit spezifischem Kontext, den das Modell braucht?
  → Prompt Engineering

Musst du verhindern, dass Nutzer Anweisungen überschreiben?
  → Fine-Tuning
```

## Was du dir merken solltest

Prompt Engineering und Fine-Tuning ergänzen sich. Prompt Engineering
ist schnell und flexibel und ändert sich mit jeder Anfrage. Fine-Tuning
ist langsam und konsistent und steckt in den Weights des Modells.
Beginne mit Prompt Engineering. Wechsle zu Fine-Tuning, wenn du
Konsistenz, Skalierung oder Widerstandsfähigkeit gegen Injection-
Angriffe brauchst. Die meisten Produktivsysteme kombinieren beides:
fine-getunte Modelle mit aufgabenspezifischen Prompts.
