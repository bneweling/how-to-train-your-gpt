# Was ist Fine-Tuning

## Die kurze Antwort

Fine-Tuning nimmt ein Modell, das bereits Sprache beherrscht, und bringt
ihm eine bestimmte Fähigkeit bei. Das Modell verfügt bereits über
Grammatik, Fakten und Gesprächsmuster aus dem Pretraining. Fine-Tuning
fügt eine zusätzliche Trainingsebene auf einem fokussierten Datensatz
hinzu. Das Ergebnis ist ein Modell, das Anweisungen befolgt, Stimmungen
klassifiziert, Sprachen übersetzt oder jede andere Aufgabe erfüllt, die
sich mit Beispielen beschreiben lässt.

## Wo es in der Pipeline steht

```
Rohtext aus dem Internet
  → Pretraining (teuer, allgemein)
    → Basismodell (kennt Sprache, aber keinen Chat)
      → Fine-Tuning (günstiger, fokussiert)
        → Chat-Modell (hilfreich, befolgt Anweisungen)
```

Pretraining dauert Wochen auf Tausenden von GPUs. Fine-Tuning dauert
Stunden auf einer einzigen GPU. Das Basismodell leistet die eigentliche
Schwerarbeit. Fine-Tuning lenkt es lediglich in die richtige Richtung.

## Warum wir es brauchen

Ein Basismodell kann Text vervollständigen. Gib ihm *Die Hauptstadt von
Frankreich ist* ein und es vervollständigt mit *Paris.* Gib ihm *Fasse
diesen Artikel zusammen* ein und es vervollständigt mit *in ein paar
Sätzen* statt tatsächlich zusammenzufassen. Das Basismodell weiß nicht,
dass es Anweisungen befolgen soll. Es wurde darauf trainiert, das
nächste Wort in Internettext vorherzusagen. Internettext enthält
Beispiele von Menschen, die um Zusammenfassungen bitten, aber auch
Beispiele von Menschen, die Vervollständigungen für *Fasse diesen
Artikel zusammen* schreiben. Das Modell hat gelernt zu vervollständigen.
Nicht zu befolgen.

Fine-Tuning bringt dem Modell bei, zwischen beidem zu unterscheiden.
Durch das Training mit Beispielen von Anweisungen, gepaart mit den
gewünschten Antworten, lernt das Modell zu erkennen, wann es
aufgefordert wird, etwas zu tun, und wie es die korrekte Ausgabe
erzeugt.

## Die drei Ansätze

| Ansatz | Was passiert | Benötigte Hardware | Wie lange |
|---|---|---|---|
| Full Fine-Tuning | Aktualisiert jedes Gewicht im Modell | 4-8 A100-GPUs | Stunden bis Tage |
| LoRA | Trainiert kleine Adaptermatrizen. Ursprüngliche Gewichte eingefroren | Einzelne 3090/4090 | Minuten bis Stunden |
| QLoRA | Quantisiert das Basismodell auf 4-Bit. Trainiert LoRA-Adapter | Einzelne Laptop-GPU | Stunden |

Full Fine-Tuning aktualisiert alle 152 Millionen Gewichte. Dies ist der
leistungsfähigste Ansatz, erfordert aber Hardware, die die meisten
Menschen nicht besitzen. Das Modell wird vollständig an die neue
Aufgabe angepasst, aber das ursprüngliche Modell geht dabei verloren.
Du musst für jede Aufgabe, für die du fine-tunst, eine separate
vollständige Kopie des Modells speichern.

LoRA verändert die ursprünglichen Gewichte nicht. Es fügt kleine
trainierbare Matrizen neben ihnen hinzu. Am Ende hast du das
ursprüngliche Modell plus winzige Adapterdateien, die jeweils nur
wenige Megabyte groß sind. Du kannst Adapter austauschen wie das
Wechseln eines Objektivs an einer Kamera. Ein Basismodell. Viele
Adapter. Jeder Adapter macht das Modell in etwas anderem gut.

QLoRA ist LoRA mit einem zusätzlichen Kniff. Es komprimiert das
Basismodell auf vier Bit pro Gewicht, bevor LoRA angewendet wird. Die
Komprimierung reduziert den Speicherbedarf zum Laden des Modells um
das Vier- bis Achtfache. Ein Modell mit sieben Milliarden Parametern,
das normalerweise vierzehn Gigabyte benötigt, läuft mit QLoRA in unter
vier Gigabyte. Das macht Fine-Tuning auf Consumer-Hardware zugänglich.

## Wann welchen Ansatz verwenden

Wenn du Zugriff auf einen Cluster von GPUs hast und maximale
Performance für eine geschäftskritische Aufgabe brauchst, verwende
Full Fine-Tuning.

Wenn du eine einzelne gute GPU hast und dem Modell eine neue Fähigkeit
beibringen willst, verwende LoRA. Das deckt fast alle ab, die
Anwendungen bauen. Chatbots, Kundensupport-Agenten, Code-Assistenten
und Content-Generatoren werden routinemäßig mit LoRA gebaut.

Wenn du einen Laptop oder eine ältere GPU hast und trotzdem fine-tunen
willst, verwende QLoRA. Die Qualität ist etwas geringer als bei LoRA,
aber das Modell lernt die Aufgabe trotzdem erfolgreich. Der Unterschied
schrumpft, je besser die Basismodelle werden.

## Das Datenformat

Fine-Tuning-Daten sind einfach. Du stellst Prompt-Antwort-Paare bereit.

```
{
  "instruction": "Übersetze ins Französische: Hallo, wie geht es dir?",
  "response": "Bonjour, comment allez-vous?"
}
```

Jedes Beispiel bringt dem Modell eine Sache bei. Bei dieser Eingabe
diese Ausgabe erzeugen. Nachdem es Tausende von Beispielen gesehen hat,
generalisiert das Modell und kann korrekte Ausgaben für Eingaben
erzeugen, die es noch nie gesehen hat.

Das Format variiert je nach Aufgabe, aber das Prinzip bleibt gleich.
Zeige dem Modell, was zu tun ist. Lass es das Muster selbst
herausfinden. Erkläre ihm die Regeln nicht explizit. Lass es aus
Beispielen lernen, so wie es Sprache aus Sätzen gelernt hat.

## Was Fine-Tuning nicht kann

Fine-Tuning kann dem Modell kein neues Wissen beibringen, das nicht in
seinen Pretraining-Daten enthalten war. Wenn das Modell nie mit
medizinischen Unterlagen trainiert wurde, kann es nicht plötzlich zum
Arzt werden. Fine-Tuning kann nur Wissen, das das Modell bereits
besitzt, neu anordnen und anwenden.

Fine-Tuning kann grundlegende architektonische Einschränkungen nicht
beheben. Ein Modell mit einem kleinen Kontextfenster bleibt klein. Ein
Modell, das halluziniert, wird weiterhin halluzinieren. Fine-Tuning
kann bestimmte Fehlermodi reduzieren, aber nicht eliminieren.

Fine-Tuning kann aus einem schlechten Modell kein gutes machen. Das
Basismodell muss bereits kompetent sein. Fine-Tuning auf einem winzigen,
inkompetenten Modell erzeugt ein winziges, kompetentes Modell. Der
Abstand zwischen Basismodellen unterschiedlicher Größe bleibt auch nach
dem Fine-Tuning bestehen. Ein fine-getuntes 7B-Modell holt ein
fine-getuntes 70B-Modell nie ein.

## Was du dir merken solltest

Fine-Tuning passt ein vortrainiertes Modell an eine bestimmte Aufgabe
an. Es ist günstiger und schneller als Pretraining, weil es von einem
Modell ausgeht, das bereits Sprache beherrscht. LoRA ist der
Standardansatz, weil es schnell und effizient ist und winzige
Adapterdateien erzeugt. QLoRA erweitert LoRA, sodass es auf
Consumer-Hardware läuft. Das Datenformat besteht aus einfachen
Instruktion-Antwort-Paaren. Fine-Tuning fügt kein neues Wissen hinzu.
Es ordnet vorhandenes Wissen so an, dass es Mustern folgt.
