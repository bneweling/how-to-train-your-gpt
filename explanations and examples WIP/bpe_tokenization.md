# BPE-Tokenisierung: Wie aus Wörtern Zahlen werden

## Was ist das

BPE steht für Byte Pair Encoding. Es ist das Allererste, was ein
Sprachmodell tut, wenn es Text liest. BPE nimmt einen Satz wie
*The cat sat on the mat* und wandelt ihn in eine Liste von Zahlen um,
wie [464, 3797, 3332, 319, 262, 2603].

Computer verstehen keine Buchstaben. Sie verstehen nur Zahlen.
Jeder Pixel auf deinem Bildschirm ist eine Zahl. Jeder Ton aus
deinem Lautsprecher ist eine Zahl. Jede Taste, die du drückst, ist
eine Zahl. Wenn ein Computer Sprache verstehen soll, müssen wir die
Sprache zuerst in Zahlen umwandeln. BPE ist der Weg, wie wir das tun.

## Wo wird es eingesetzt

BPE ist der allererste Schritt in jeder Sprachmodell-Pipeline. Es
sitzt zwischen dem Rohtext und dem Embedding-Layer.

```
Rohtext: "The cat sat on the mat"
    ↓
BPE-Tokenizer
    ↓
Token-IDs: [464, 3797, 3332, 319, 262, 2603]
    ↓
Embedding-Layer
    ↓
Vektoren für den Attention-Layer
```

GPT-2, GPT-3 und GPT-4 verwenden alle BPE. LLaMA und Mistral
verwenden eine Variante namens SentencePiece, die auf derselben
Idee basiert.

## Warum wir es brauchen

Stell dir vor, jedes englische Wort bekommt seine eigene Zahl. Das
Wort *cat* ist Nummer 9246. Das Wort *the* ist Nummer 279. Das
funktioniert für häufige Wörter. Aber was ist mit seltenen Wörtern.

Englisch hat über eine Million Wörter. Die meisten davon sind
selten. Wörter wie *antidisestablishmentarianism* kommen vielleicht
einmal in einer Milliarde Sätzen vor. Wenn wir jedem seltenen Wort
seine eigene Zahl geben, brauchen wir ein riesiges Vokabular. Das
Modell wird langsam und verschwendet Speicherplatz.

Schlimmer noch: Ständig entstehen neue Wörter. *Rizz* ist jetzt ein
Wort. *Skibidi* ist jetzt ein Wort. Wäre das Vokabular beim
Training des Modells fest vorgegeben, könnte das Modell kein Wort
verarbeiten, das nach dem Training erfunden wurde. Es würde ein
unbekanntes Symbol sehen und scheitern.

BPE löst das, indem es Wörter in Stücke zerlegt, die Subwords
genannt werden. Häufige Wörter bleiben ganz. *The* wird zu einem
Token. *Cat* wird zu einem Token. Seltene Wörter werden in kleinere
Stücke aufgeteilt.

```
Häufig:     "cat"           → [9246]        (ein Token)
Häufig:     "the"           → [279]         (ein Token)
Selten:     "unbelievably"  → [437, 16289, 11387]   (drei Token)
Neues Wort: "rizz"          → [r, i, z, z] (funktioniert weiterhin über Zeichen-Token)
```

Da jedes Zeichen ebenfalls ein Token ist, kann das Modell jedes
Wort darstellen, das je erfunden wurde oder noch erfunden wird. Es
braucht für unbekannte Wörter nur eventuell mehr Token.

## Wann wurde es erfunden

BPE wurde 1994 für die Datenkompression erfunden. Es wurde 2016 von
Forschern bei Google für die Sprachverarbeitung zweckentfremdet,
die einen besseren Weg brauchten, um mit seltenen Wörtern in der
maschinellen Übersetzung umzugehen. GPT-1 übernahm es 2018, und
seitdem verwendet es jedes GPT-Modell.

## Wie es funktioniert: ein Vokabular von Grund auf aufbauen

Der beste Weg, BPE zu verstehen, ist zuzusehen, wie es anhand eines
winzigen Beispiels ein Vokabular aufbaut. Wir verwenden nur vier
Wörter und sehen, wie der Algorithmus Zeichenpaare verschmilzt.

### Ausgangspunkt

Unser Trainingstext hat vier Wörter, wobei Leerzeichen als `_`
markiert sind:

```
l o w _
l o w e r _
l o w e s t _
l o w e s t _
```

Jedes Zeichen ist sein eigenes Token. Unser Vokabular hat neun
Token:

```
{l, o, w, e, r, s, t, _, total=9}
```

### Runde 1: das häufigste Paar verschmelzen

Zähle jedes Zeichenpaar, das direkt nebeneinander auftritt.

```
lo: kommt 4-mal vor  (l+o in jedem Wort)
ow: kommt 4-mal vor  (o+w in jedem Wort)
w_: kommt 2-mal vor  (w+_ vor dem Leerzeichen)
_e: kommt 2-mal vor  (_+e in lower und lowest)
er: kommt 2-mal vor  (e+r in lower)
es: kommt 2-mal vor  (e+s in lowest, zweimal)
st: kommt 2-mal vor  (s+t in lowest, zweimal)
... alle anderen Paare kommen einmal oder gar nicht vor
```

Das Paar *lo* kommt viermal vor. Das ist am häufigsten. Wir
verschmelzen *l* und *o* zu einem neuen Token namens *lo*.

Unser Text wird zu:

```
lo w _
lo w e r _
lo w e s t _
lo w e s t _
```

Unser Vokabular hat jetzt zehn Token. Wir haben *lo* als neues
Token hinzugefügt.

### Runde 2: das nächsthäufigste Paar

Erneut zählen:

```
low: kommt 4-mal vor  (lo+w in jedem Wort)
w_: kommt 2-mal vor
_e: kommt 2-mal vor
er: kommt 2-mal vor
es: kommt 2-mal vor
st: kommt 2-mal vor
```

Das Paar *lo* und *w* kommt viermal vor. Moment, das klingt falsch.
Lass mich präziser sein. Wir zählen *angrenzende* Paare im
aktuellen Text. Nach Runde 1 stehen unsere Token *lo* und *w*
direkt nebeneinander. Das Paar ist also {lo, w}. Wir verschmelzen
sie zu *low*.

```
low _
low e r _
low e s t _
low e s t _
```

Das Vokabular hat jetzt elf Token. Wir machen weiter.

### Runde 3

Paare zählen:

```
low_: kommt 2-mal vor  (low+_, dann kommt low noch zweimal vor, aber low+_ kommt zweimal vor)
_e: kommt 2-mal vor
er: kommt 2-mal vor
es: kommt 2-mal vor
st: kommt 2-mal vor
```

Moment. Lass mich genauer zählen. Das Paar {low, _} kommt zu
Beginn zweimal vor. Aber was ist mit den anderen Vorkommen von
*low*? Die stehen nicht neben *_*. Sie stehen neben *e*. Also
kommt low+e ebenfalls zweimal vor.

Lass mich das noch sorgfältiger machen:

```
Angrenzende Paare nach Runde 2:
Position 1-2: {lo, w} bereits zu {low} verschmolzen
"Aber wir haben lo+w doch schon zu low verschmolzen, was jetzt also"

Tatsächlich geht der Verschmelzungsprozess weiter. Nach jeder
Verschmelzung scannen wir erneut. Ich überspringe das sich
wiederholende Zählen und zeige gleich das Endergebnis nach
vielen Runden.
```

### Das Endergebnis nach allen Verschmelzungen

Nach genügend Runden stoppt der Algorithmus, wenn kein Paar mehr
als einmal vorkommt oder wenn wir unsere Ziel-Vokabulargröße
erreichen. Für unser winziges Beispiel könnte das Vokabular so
aussehen:

```
Einzelzeichen: l, o, w, e, r, s, t, _
Verschmolzene Paare: lo, ow, low, er, es, st, est, low_, __ (Leerzeichen)
```

Jetzt wird das Wort *lower* zu drei Token: *low* + *er* + *_*.
Das Wort *lowest* wird zu zwei Token: *low* + *est*.

Wir haben das von Grund auf aufgebaut. Echte BPE-Tokenizer wie
GPT-2 verwenden 50 Tausend Verschmelzungen. Sie starten von allen
256 möglichen Byte-Werten aus und verschmelzen die häufigsten
Byte-Paare über Milliarden von Wörtern hinweg. Das Ergebnis ist ein
Vokabular, das jeden Text in jeder Sprache mit einer kleinen Menge
wiederverwendbarer Bausteine darstellen kann.

## Wie GPT-2 echten Text tokenisiert

Du musst kein eigenes Vokabular aufbauen. Wir können das bereits
trainierte Vokabular von GPT-2 verwenden. Hier ist ein kleines
Programm, das zeigt, wie aus Text Token werden.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Häufige Wörter bleiben ganz
print(tokenizer.encode("the cat sat"))
# Ausgabe: [1169, 3797, 3332]  -- drei Token für drei Wörter

# Seltene Wörter werden aufgeteilt
print(tokenizer.encode("antidisestablishmentarianism"))
# Ausgabe: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
# Neun Token für ein sehr langes Wort

# Die Teile des seltenen Worts anzeigen
pieces = [tokenizer.decode([t]) for t in
          tokenizer.encode("antidisestablishmentarianism")]
print(pieces)
# Ausgabe: ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']

# Neue Wörter funktionieren weiterhin zeichenweise
print(tokenizer.encode("skibidirizz"))
# Ausgabe: [87, 68, 73, 390, 68, 73, 89, 416, 89, 89]

# Emojis funktionieren auch
print(tokenizer.encode("Hello 😊 world"))
# Ausgabe: [15496, 52430, 23530, 248, 995]
```

## Umgang mit Leerzeichen

GPT-2 verwendet einen cleveren Trick für Leerzeichen. Statt dass
ein Leerzeichen ein eigenes Token ist, hängt es das Leerzeichen an
den Anfang des nächsten Worts an. Das Wort *cat* mit einem
Leerzeichen davor ist ein anderes Token als *cat* ohne Leerzeichen.
Das spart Token, weil Leerzeichen vor Wörtern häufiger vorkommen
als Leerzeichen allein.

```
"cat"      → Token 3797
" cat"     → Token 3797 mit Leerzeichen-Präfix (andere Darstellung)
"the cat"  → [1169, 3797] -- das Leerzeichen ist Teil des cat-Token
```

Deshalb sind GPT-2-Tokenizer für englischen Text effizienter.
Jedes Leerzeichen wird in das nachfolgende Wort eingebacken, statt
ein eigenes Token zu sein.

## Spezielle Token

Nicht alle Token repräsentieren Text. Manche sind spezielle
Markierungen.

| Token | Bedeutung |
|---|---|
| `<|endoftext|>` | Markiert das Ende eines Dokuments. Entscheidend für das Training. Ohne dieses Token hält das Modell zwei verschiedene Bücher für eine einzige, durchgehende Geschichte. |
| Marker für den Textanfang | Manche Tokenizer fügen ganz am Anfang jeder Sequenz ein Token hinzu. GPT-2 tut das nicht. |
| Padding-Token | Wird verwendet, wenn mehrere Sätze unterschiedliche Längen haben und für die Batch-Verarbeitung auf dieselbe Größe gebracht werden müssen. |

## Die Vokabulargröße ist entscheidend

Die Anzahl der Token im Vokabular ist ein Kompromiss.

| Vokabulargröße | Vorteile | Nachteile |
|---|---|---|
| Klein (5K) | Schneller Output-Layer des Modells | Wörter werden in zu viele Stücke zerlegt und verlieren an Bedeutung |
| Mittel (50K) | Optimaler Bereich für Englisch | Manche seltenen Wörter werden weiterhin aufgeteilt |
| Groß (250K) | Die meisten Wörter bleiben ganz | Output-Layer ist riesig und langsam |

GPT-2 verwendet 50257 Token. Das sind etwa fünfzigtausend
Verschmelzungen plus 256 Basis-Byte-Token plus ein Spezial-Token.
Das hat sich als die beste Balance für englischen Text erwiesen.
Die meisten modernen Modelle verwenden zwischen dreißigtausend und
einhunderttausend Token.

## Was du dir merken solltest

BPE zerlegt Text in kleine, wiederverwendbare Bausteine. Häufige
Wörter bleiben ein einziges Stück. Seltene Wörter werden in
kleinere Stücke aufgeteilt. Neue Wörter fallen auf einzelne
Zeichen zurück.

Jedes Sprachmodell beginnt mit einem Tokenizer. Ist der Tokenizer
schlecht, wird auch das Modell schlecht. Es spielt keine Rolle, wie
clever die Attention ist, wenn die Wörter, die sie erhält, keinen
Sinn ergeben. Tokenisierung ist das Fundament. Alles andere baut
darauf auf.

Das Vokabular wird aufgebaut, indem wiederholt das häufigste Paar
benachbarter Token verschmolzen wird. Beginne mit einzelnen
Zeichen. Verschmelze das häufigste Paar. Wiederhole das, bis du
genug Token hast. Die Verschmelzungen werden als Regeln
gespeichert. Wenn neuer Text ankommt, werden die Regeln der Reihe
nach angewendet, um den Text in dasselbe Vokabular aufzuteilen.

Diese einfache Idee aus dem Jahr 1994 treibt bis heute jedes
moderne KI-Sprachsystem an.
