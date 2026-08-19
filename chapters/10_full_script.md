# Kapitel 10 — Das vollständige Skript (main.py)

Das vollständige, lauffähige Trainingsskript liegt im Wurzelverzeichnis des Repos als `main.py`.

Es enthält alles inline — Tokenizer, Modell, Trainingsschleife und Inferenz —, sodass du das Repo klonen und mit einem einzigen Befehl ausführen kannst.

```bash
python main.py
```

Standardmäßig wird ein winziges Modell (d_model=256, 4 Layer, 4 Heads) verwendet, das auf 5.000 Wikipedia-Artikeln für 500 Schritte trainiert wird. Dies dauert etwa 2–5 Minuten auf der CPU oder wenige Sekunden auf der GPU. Das Skript enthält außerdem eine auskommentierte GPT-2-Small-Konfiguration (768 Dimensionen, 12 Layer, 12 Heads) für den Fall, dass du eine GPU zur Verfügung hast.

Nach dem Training speichert das Skript einen Checkpoint unter `checkpoints/model.pt`, plottet eine Loss-Kurve und generiert Beispieltext zu einigen Prompts.

---

**Zurück:** [Kapitel 9 — Inferenz](09_inference.md)
**Weiter:** [Kapitel 11 — Glossar](11_glossary.md)
