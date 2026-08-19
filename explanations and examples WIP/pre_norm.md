# Pre-Norm vs. Post-Norm: Wo normalisiert wird

## Was ist das

Pre-Norm und Post-Norm sind zwei Möglichkeiten, die Normalisierung
innerhalb eines Transformer-Blocks zu platzieren. Pre-Norm
normalisiert vor jedem Sublayer. Post-Norm normalisiert nach jedem
Sublayer. Der Unterschied besteht aus einer einzigen Codezeile, aber
die Auswirkung auf das Training ist enorm.

Pre-Norm (modern):
```
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

Post-Norm (Original):
```
x = Norm(x + Attention(x))
x = Norm(x + FFN(x))
```

Man kann es sich wie das Bearbeiten eines Dokuments vorstellen.
Pre-Norm bereinigt den unordentlichen Entwurf, bevor Änderungen
vorgenommen werden. Man startet mit einer klaren Basis. Post-Norm
nimmt Änderungen an einem unordentlichen Entwurf vor und bereinigt
anschließend das Ergebnis. Die Änderungen könnten auf Rauschen
basieren.

## Wo wird es verwendet

Die Wahl zwischen Pre-Norm und Post-Norm betrifft jeden
Transformer-Block im Modell. Bei einem Modell mit zwölf Blöcken wird
diese Wahl vierundzwanzig Mal pro Forward Pass getroffen.

## Warum sich Pre-Norm durchgesetzt hat

Das ursprüngliche Transformer-Paper verwendete
Post-Normalisierung. Die Autoren normalisierten den Output jedes
Sublayers. Das funktionierte bei Modellen mit sechs Layern. Als
Forscher versuchten, auf mehr Layer zu skalieren, wurde das Training
instabil. Das Modell konvergierte jenseits von etwa zwölf Layern
nicht mehr.

Das Problem war die Residual Connection. Bei Post-Norm verläuft der
Residual-Pfad durch die Normalisierung. Das bedeutet, dass auch der
Gradient, der durch die Residual Connection zurückfließt,
normalisiert wird. Normalisierung staucht die Gradientenmagnitude.
Nach vielen Layern wird der Gradient zu klein, um die frühen Layer
zu trainieren.

Bei Pre-Norm umgeht der Residual-Pfad die Normalisierung
vollständig. Der Gradient, der durch die Residual Connection
zurückfließt, wird nie normalisiert. Er erreicht die frühen Layer
mit voller Stärke. Deshalb ermöglicht Pre-Norm das Training von
Modellen mit sechsundneunzig Layern oder mehr.

```
Post-Norm-Gradientenpfad:
  Loss → Norm → Sublayer → Norm → Sublayer → ... → Input
  Jede Norm komprimiert den Gradienten.
  Nach N Layern ist der Gradient um 0.5^N kleiner.

Pre-Norm-Gradientenpfad:
  Loss → + → ... → + → Input (über Residual Connections)
  Keine Normalisierung auf dem Residual-Pfad.
  Der Gradient kommt unabhängig von der Tiefe in voller Stärke an.
```

## Wann wurde es entdeckt

Pre-Norm wurde 2019 von Forschern bei Google vorgeschlagen, die
untersuchten, warum tiefe Transformer schwer zu trainieren waren.
Sie fanden heraus, dass das Vertauschen der Normalisierungsposition
das Training bei jeder Tiefe stabil machte. GPT-3 übernahm Pre-Norm
im Jahr 2020. Seitdem verwendet jedes Modell Pre-Norm. Post-Norm
findet sich heute nur noch in Legacy-Code und historischen
Vergleichen.

## Wie sie sich Schritt für Schritt unterscheiden

Verfolgen wir ein einzelnes Token durch einen Block mit beiden
Ansätzen.

### Pre-Norm (was wir verwenden)

```
x = ein Vektor [0.5, -0.3, 0.8, -0.1]

Schritt 1: norm(x) = [0.6, -0.4, 1.0, -0.1]  (neu skaliert, um sauberer zu sein)
Schritt 2: attention(norm(x)) = [0.1, 0.0, -0.2, 0.3]
Schritt 3: x + attention = [0.6, -0.3, 0.6, 0.2]
Schritt 4: norm(step3_result) = [0.8, -0.4, 0.8, 0.3]
Schritt 5: ffn(norm(step4_result)) = [-0.1, 0.2, 0.0, 0.1]
Schritt 6: step3_result + ffn = [0.5, -0.1, 0.6, 0.3]
```

Der Output [0.5, -0.1, 0.6, 0.3] ähnelt dem Input [0.5, -0.3, 0.8,
-0.1]. Das Modell hat kleine Anpassungen an einer sauberen Basis
vorgenommen.

### Post-Norm (Originalpaper)

```
x = ein Vektor [0.5, -0.3, 0.8, -0.1]

Schritt 1: attention(x) = [2.5, -0.1, -1.8, 0.9]  (große Outputs)
Schritt 2: x + attention = [3.0, -0.4, -1.0, 0.8]
Schritt 3: norm(step2_result) = [1.5, -0.2, -0.5, 0.4]  (gestaucht)
Schritt 4: ffn(step3_result) = [-0.8, 1.2, -0.3, 0.7]
Schritt 5: step3_result + ffn = [0.7, 1.0, -0.8, 1.1]
Schritt 6: norm(step5_result) = [0.4, 0.6, -0.5, 0.7]
```

Der Output unterscheidet sich stärker vom Input, da die
Normalisierung bei jedem Schritt alles verändert. Das klingt in der
Theorie gut. Mehr Transformation. Aber in der Praxis stört die
ständige Normalisierung den Gradientenfluss und macht tiefe
Netzwerke untrainierbar.

## Das Gradientenargument

Stellen wir uns ein Netzwerk mit sechsundneunzig Layern vor. Jeder
Layer hat zwei Normalisierungsoperationen. Bei Post-Norm fließt der
Gradient auf seinem Weg zurück zum ersten Layer durch
einhundertzweiundneunzig Normalisierungsoperationen. Jede
Normalisierung komprimiert den Gradienten ein wenig. Nach
einhundertzweiundneunzig Kompressionen ist der Gradient bei Layer
eins praktisch null.

Bei Pre-Norm umgehen die Residual Connections die Normalisierung.
Der Gradient fließt direkt über den Residual-Highway. Auf dem
Shortcut-Pfad durchläuft er nie eine Normalisierung. Nur der Pfad
durch die Sublayer durchläuft die Normalisierung. Der Shortcut-Pfad
liefert jedem Layer ein starkes Gradientensignal.

Deshalb ist Pre-Norm eine harte Anforderung für tiefe Transformer.
Es ist keine Präferenz. Es ist eine notwendige Bedingung dafür, dass
das Training jenseits einer bestimmten Tiefe überhaupt funktioniert.

## Was man sich merken muss

Pre-Norm normalisiert den Input vor jedem Sublayer. Post-Norm
normalisiert den Output nach jedem Sublayer. Pre-Norm ermöglicht das
Training tiefer Netzwerke, weil die Residual Connections die
Normalisierung umgehen und das Gradientensignal erhalten. Post-Norm
beschränkt das Training auf flache Netzwerke, weil die
Normalisierung die Gradienten staucht.

Jedes moderne Sprachmodell verwendet Pre-Norm. Wenn man in einer
Codebasis auf Post-Norm stößt, handelt es sich entweder um einen Bug
oder um ein historisches Artefakt. Das ursprüngliche
Transformer-Paper lag in diesem einen Detail falsch. Die Lösung
wurde ein Jahr später entdeckt und ist seitdem Standard.
