# Datenaufbereitung für das Fine-Tuning

## Die kurze Antwort

Fine-Tuning-Daten sind eine Sammlung von Beispielen. Jedes Beispiel
zeigt dem Modell, was zu tun ist. Man gibt ihm eine Eingabe und die
gewünschte Ausgabe. Nachdem das Modell tausende Beispiele gesehen hat,
lernt es das Muster und kann für neue Eingaben korrekte Ausgaben
erzeugen. Die Qualität der Daten ist wichtiger als die Menge. Das
Bereinigen der Daten und ihre konsistente Formatierung machen den
Großteil der Arbeit aus.

## Das Grundformat

Jedes Beispiel besteht mindestens aus einer Anweisung und einer
Antwort. Manche Beispiele enthalten außerdem einen System-Prompt oder
Kontext.

```json
{
  "instruction": "Fasse diesen Absatz in einem Satz zusammen.",
  "input": "Die Katze saß drei Stunden lang auf der Matte...",
  "response": "Eine Katze blieb längere Zeit auf einer Matte."
}
```

Das Feld `input` ist optional. Viele Aufgaben benötigen nur eine
Anweisung. Das Modell lernt aus dem Muster Anweisung gefolgt von
Antwort.

## Chat-Templates

Die meisten Modelle, die per Fine-Tuning angepasst wurden, verwenden
ein Chat-Format. Anweisung und Antwort werden in spezielle Token
eingebettet, die dem Modell mitteilen, wo die Nutzernachricht endet und
die Antwort des Assistenten beginnt.

```
<|user|>
Was ist die Hauptstadt von Frankreich?
<|assistant|>
Die Hauptstadt von Frankreich ist Paris.
<|endoftext|>
```

Die genauen Token hängen vom Basismodell ab. LLaMA verwendet `[INST]`
und `[/INST]`. Mistral verwendet `<s>[INST]` und `[/INST]`.
Open-Source-Chat-Modelle nutzen häufig die Marker `<|user|>` und
`<|assistant|>`. Werden die falschen Token verwendet, erkennt das
Modell das Format nicht und erzeugt sinnlosen Text.

## Wie viele Daten man braucht

Das Minimum liegt bei etwa einhundert Beispielen. Darunter kann das
Modell nicht generalisieren. Es merkt sich die Beispiele auswendig und
versagt bei neuen Eingaben. Tausend Beispiele sind für die meisten
Aufgaben solide. Zehntausend sind gut für komplexe Aufgaben wie
mehrstufige Konversationen oder Codegenerierung. Über zehntausend
hinaus nimmt der Nutzen schnell ab.

```
100 Beispiele:     Absolutes Minimum. Modell generalisiert womöglich nicht.
1.000 Beispiele:   Gut für die meisten Klassifikations- und einfachen QA-Aufgaben.
5.000 Beispiele:   Gut für Instruction Following und Zusammenfassung.
10.000+ Beispiele: Komplexe Aufgaben. Danach nimmt der Nutzen ab.
```

Die Qualität jedes einzelnen Beispiels ist wichtiger als die
Gesamtzahl. Tausend sorgfältig geschriebene Beispiele schlagen
zehntausend schlampige. Jeder Fehler in den Trainingsdaten bringt dem
Modell bei, genau diesen Fehler zu machen. Das Modell lernt, Muster zu
replizieren. Nicht, sie zu bewerten.

## Die Daten bereinigen

Geh deine Daten vor dem Training durch und entferne alles, was dem
Modell schlechte Angewohnheiten beibringen würde.

Beispiele für das, was entfernt werden sollte: Antworten, die
abgeschnitten oder unvollständig sind. Antworten, die der Anweisung
widersprechen. Anweisungen, die mehrdeutig oder unmöglich zu befolgen
sind. Doppelte Beispiele. Beispiele, bei denen die Antwort in der
falschen Sprache oder im falschen Format vorliegt. Beispiele mit
anstößigen oder schädlichen Inhalten.

```
Gutes Beispiel:
  Anweisung: "Was ist 2 + 2?"
  Antwort: "2 + 2 ergibt 4."

Schlechtes Beispiel (vage):
  Anweisung: "Erzähl mir etwas über Mathematik."
  Antwort: "Ist cool."

Schlechtes Beispiel (widersprüchlich):
  Anweisung: "Gib eine kurze Antwort."
  Antwort: "Lass mich das über mehrere Absätze hinweg ganz genau erklären..."
```

## Die Daten ausbalancieren

Fine-Tuning kann das Modell aus dem Gleichgewicht bringen. Trainierst
du nur auf eine Aufgabe, wird das Modell bei allem anderen schlechter.
Das nennt man katastrophales Vergessen (catastrophic forgetting). Das
Modell verlernt, ein normales Gespräch zu führen, weil jedes
Trainingsbeispiel eine bestimmte Aufgabe ist.

Mische einige allgemeine Konversationsbeispiele unter die Daten. Etwa
zehn Prozent deiner Daten sollten normaler Chat sein. Das bewahrt das
Modell davor, seine allgemeinen Fähigkeiten zu verlieren, während es
die neue Aufgabe lernt.

## Der Prompt ist entscheidend

Der Wortlaut der Anweisung beeinflusst die Ergebnisse. Unterschiedliche
Formulierungen führen zu unterschiedlichem Verhalten. Trainierst du mit
Anweisungen wie *Fasse das zusammen*, lernt das Modell, bei diesen
Worten zusammenzufassen. Fragt jemand stattdessen *Gib mir eine
Kurzfassung*, erkennt das Modell die Anfrage womöglich nicht als
Zusammenfassungsauftrag.

Formuliere Anweisungen auf mehrere Arten. Gib für jede Aufgabe drei
oder vier unterschiedliche Formulierungen derselben Anweisung an. Das
bringt dem Modell die Absicht hinter den Worten bei, statt nur den
genauen Wortlaut.

```
"Fasse diesen Absatz zusammen."
"Kannst du das zusammenfassen?"
"Gib mir eine Zusammenfassung des folgenden Texts."
"Fasse kurz zusammen, was hier steht."
```

Alle vier bedeuten dasselbe. Das Training mit allen vier Varianten
macht das Modell robust gegenüber unterschiedlichen Formulierungen.

## Die Daten aufteilen

Teile deine Daten in Trainings- und Validierungsmengen auf. Trainiere
auf neunzig Prozent. Evaluiere auf zehn Prozent. Die Validierungsmenge
zeigt dir, ob das Modell die Aufgabe lernt oder die Beispiele nur
auswendig lernt.

Sinkt der Trainings-Loss, während der Validierungs-Loss steigt, lernt
das Modell auswendig. Beende das Training. Reduziere die Anzahl der
Schritte oder erhöhe den Dropout. Sinken beide, bist du auf dem
richtigen Weg.

## Ein reales Beispiel

Hier ist ein kleines Instruction-Tuning-Dataset im Format, das Alpaca
verwendet.

```json
[
  {
    "instruction": "Nenne drei Tipps, um gesund zu bleiben.",
    "input": "",
    "output": "1. Iss eine ausgewogene Ernährung mit reichlich Gemüse..."
  },
  {
    "instruction": "Was sind die drei Primärfarben?",
    "input": "",
    "output": "Die drei Primärfarben sind Rot, Blau und Gelb."
  },
  {
    "instruction": "Beschreibe die folgende Stadt in drei Sätzen.",
    "input": "Tokio",
    "output": "Tokio ist die Hauptstadt Japans und eine der..."
  }
]
```

Jedes Beispiel ist eine in sich geschlossene Lektion. Das Modell sieht
die Anweisung und lernt, die Ausgabe zu erzeugen. Nach tausenden
Beispielen kann es Variationen von Anweisungen bewältigen, die es noch
nie zuvor gesehen hat.

## Was du dir merken solltest

Fine-Tuning-Daten sind Anweisung-Antwort-Paare. Das Format nutzt
spezielle Token, die vom jeweiligen Basismodell abhängen. Tausend gute
Beispiele schlagen zehntausend schlechte. Bereinige die Daten
sorgfältig. Das Modell lernt, Muster zu replizieren, nicht ihre
Qualität zu beurteilen. Mische allgemeine Konversation unter, um
katastrophales Vergessen zu verhindern. Variiere die Formulierung von
Anweisungen, um Robustheit aufzubauen. Teile die Daten in Trainings-
und Validierungsmengen auf, um auf Auswendiglernen zu prüfen.
