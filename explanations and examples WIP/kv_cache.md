# KV-Cache: Textgenerierung schnell machen

## Was ist das

Der KV-Cache speichert die Key- und Value-Vektoren aller vorherigen
Tokens während der Textgenerierung. Wenn das Modell das nächste
Token generiert, verwendet es diese gespeicherten Vektoren wieder,
anstatt sie von Grund auf neu zu berechnen. Das macht die
Generierung hunderte Male schneller.

Man kann es sich wie das Schreiben einer langen E-Mail vorstellen.
Ohne KV-Cache würde man die gesamte E-Mail jedes Mal von vorne
lesen, sobald man ein neues Wort tippt. Mit einem KV-Cache
erinnert man sich an alles, was man bereits geschrieben hat, und
muss nur noch über das neue Wort nachdenken. Der Unterschied im
Aufwand ist enorm.

## Wo wird er verwendet

Der KV-Cache befindet sich innerhalb des Attention-Layers. Jeder
Attention Head in jedem Transformer-Block hat seinen eigenen
Cache. Bei einem Modell mit zwölf Blöcken und zwölf Heads gibt es
einhundertvierundvierzig separate Caches. Jeder speichert die
Keys und Values für jedes bisher generierte Token.

```
Textgenerierung ohne Cache:
  Für jedes neue Token:
    Die GESAMTE Sequenz durch alle Layer laufen lassen
    Dies berechnet K und V für jedes vergangene Token neu
    Die Zeit wächst quadratisch mit der Sequenzlänge

Textgenerierung mit Cache:
  Für jedes neue Token:
    Nur K und V für das neue Token berechnen
    An den Cache anhängen
    Gecachte K und V für alle vergangenen Token wiederverwenden
    Die Zeit wächst linear mit der Sequenzlänge
```

## Warum wir ihn brauchen

Die Attention-Formel lautet Q mal K transponiert. Die Q-Matrix hat
eine Zeile pro Token. Die K-Matrix hat eine Zeile pro Token. Wenn
wir fünfhundert Tokens haben, multiplizieren wir eine
Fünfhundert-mal-vierundsechzig-Matrix mit einer
Vierundsechzig-mal-fünfhundert-Matrix. Das ist bereits einiges an
Arbeit.

Ohne Cache würden wir diese Multiplikation für jedes neue Token
von Grund auf neu durchführen. Wenn die Sequenz ein Token lang
ist, machen wir einen Vergleich. Wenn sie zwei Tokens lang ist,
machen wir zwei Vergleiche für Token null und zwei für Token eins.
Wenn sie fünfhundert Tokens lang ist, machen wir fünfhundert
Vergleiche für jedes der fünfhundert Tokens. Der Gesamtaufwand
wächst mit dem Quadrat der Sequenzlänge. Das ist quälend langsam.

Mit einem Cache berechnen wir nur die Vergleiche, an denen das
neue Token beteiligt ist. Token fünfhundert vergleicht sich mit
allen fünfhundert vorherigen Tokens. Das sind fünfhundert neue
Vergleiche. Nicht fünfhundert zum Quadrat. Die gecachten Keys
ersparen uns die Neuberechnung der alten Vergleiche, die sich
nicht verändert haben.

### Die Geschwindigkeitszahlen

```
Sequenzlänge 100:
  Ohne Cache: 100² = 10,000 Vergleiche pro Generierungsschritt
  Mit Cache:  100 Vergleiche pro Generierungsschritt
  Speedup: 100×

Sequenzlänge 1000:
  Ohne Cache: 1,000² = 1,000,000 Vergleiche
  Mit Cache:  1,000 Vergleiche
  Speedup: 1,000×
```

Bei einer langen Konversation mit tausenden Tokens macht der
KV-Cache den Unterschied zwischen Sekunden und Minuten Wartezeit
für jedes neue Wort.

## Wann wurde er erfunden

Der KV-Cache wurde bereits im ursprünglichen Transformer-Paper aus
dem Jahr 2017 beschrieben. Er war keine später hinzugefügte
Optimierung, sondern von Anfang an Teil des Designs. Die Autoren
wussten, dass autoregressive Generierung ohne ihn quälend langsam
wäre. Jede Transformer-Implementierung seit 2017 verwendet eine
Form von KV-Cache.

## Wie er Schritt für Schritt funktioniert

### Schritt 1: das erste Token

Wir haben einen Prompt, der fünf Tokens lang ist. Das Modell
verarbeitet alle fünf Tokens parallel während eines
Prefill-Schritts.

```
Prefill:
  Tokens: [The, cat, sat, on, the]
  K-Cache für Head 0: speichert K für alle 5 Tokens  (5 × 64 Matrix)
  V-Cache für Head 0: speichert V für alle 5 Tokens  (5 × 64 Matrix)
  ... wiederholen für alle 12 Heads ...
```

Dieser erste Schritt ist aufwendig, aber er passiert nur einmal.

### Schritt 2: das nächste Token generieren

Das Modell sagt voraus, dass das nächste Token *mat* ist. Wir
hängen *mat* an die Sequenz an. Jetzt haben wir sechs Tokens.

```
Generierungsschritt 1:
  Neues Token: [mat]
  Berechne K nur für "mat": (1 × 64 Matrix)
  Berechne V nur für "mat": (1 × 64 Matrix)
  Vollständiger K-Cache: 5 alte Zeilen + 1 neue Zeile = 6 × 64 Matrix
  Vollständiger V-Cache: 5 alte Zeilen + 1 neue Zeile = 6 × 64 Matrix
  Berechne Q nur für "mat": (1 × 64 Matrix)
  Attention-Scores: Q_mat × K_full^T = 1 × 6 Vergleiche
  Nur das Q des neuen Tokens wurde berechnet. Alte Q-Werte werden nicht benötigt.
```

Wir haben nur einen neuen Key, einen neuen Value und eine neue
Query berechnet. Die Attention-Scores für *mat* werden gegen alle
sechs Tokens berechnet, weil *mat* auf alles achten muss, was
zuvor kam. Aber die Attention-Scores für *The*, *cat* und *sat*
werden nicht neu berechnet. Sie ändern sich nicht. Warum sollten
sie auch. Ihr Kontext hat sich nicht verändert. Nur das neue Token
hat neuen Kontext zu verarbeiten.

### Schritt 3: Speicherwachstum

Für jedes neue Token fügen wir jedem K-Cache und jedem V-Cache
eine Zeile hinzu. Der Cache wächst linear mit der Sequenzlänge.

```
Speicherbedarf für ein GPT-2-Small-Modell bei der Generierung von 1000 Tokens:

K-Cache: 12 Layer × 12 Heads × 1000 Tokens × 64 Dimensionen × 2 Bytes (bfloat16)
       = 18,432,000 Bytes
       = 17.6 MB

V-Cache: gleich wie K-Cache = 17.6 MB

Gesamter KV-Cache: 35.2 MB
```

Fünfunddreißig Megabyte sind nichts für eine moderne GPU. Aber
bedenke, das ist GPT-2 Small mit nur zwölf Layern und 768
Dimensionen.

```
Speicherbedarf für ein GPT-3-Large-Modell bei der Generierung von 1000 Tokens:

K-Cache: 96 Layer × 96 Heads × 1000 Tokens × 128 Dimensionen × 2 Bytes
       = 2,359,296,000 Bytes
       = 2.2 GB

V-Cache: gleich wie K-Cache = 2.2 GB

Gesamter KV-Cache: 4.4 GB
```

Jetzt macht der Cache einen erheblichen Teil des GPU-Speichers
aus. Bei langen Konversationen über tausende Tokens hinweg kann
der KV-Cache zum dominanten Speicherverbraucher werden. Das ist
einer der Gründe, warum der Betrieb großer Modelle bei langen
Kontextlängen enorme Mengen an VRAM erfordert.

## Eine vereinfachte Code-Skizze

Das ist kein Produktionscode, aber es zeigt die Idee.

```python
class AttentionWithCache:
    def __init__(self):
        self.k_cache = None  # Speichert die akkumulierten K-Werte
        self.v_cache = None  # Speichert die akkumulierten V-Werte

    def forward(self, x, use_cache=False):
        q, k, v = self.qkv_proj(x)

        if use_cache and self.k_cache is not None:
            # Neue K- und V-Werte an den Cache anhängen
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        # Cache für den nächsten Schritt aktualisieren
        self.k_cache = k
        self.v_cache = v

        # Normale Attention-Berechnung
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        return weights @ v

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
```

Die Kernidee steht in den Zeilen elf und zwölf. Anstatt die alten
K- und V-Werte zu verwerfen, hängen wir die neuen an. Der Cache
wächst mit der Zeit. Zu Beginn einer neuen Konversation setzen wir
den Cache auf leer zurück.

## Der Kompromiss

Der KV-Cache tauscht Speicher gegen Geschwindigkeit. Wir nehmen in
Kauf, dass die Generierung mehr GPU-Speicher verbraucht, um jeden
Schritt deutlich schneller zu machen. Für die meisten realen
Anwendungen ist das ein guter Tausch. Speicher ist vergleichsweise
billig, verglichen mit der Geduld des Nutzers, der auf jedes neue
Wort wartet.

Bei sehr langen Sequenzen wird der Speicherbedarf jedoch
unerschwinglich. Forscher haben Techniken wie Grouped Query
Attention und Multi Query Attention entwickelt, die die Anzahl der
K- und V-Heads reduzieren. Weniger Heads bedeuten einen kleineren
Cache. Diese Techniken sind in modernen Modellen wie LLaMA 2 und
Mistral Standard.

## Was man sich merken sollte

Der KV-Cache speichert zuvor berechnete Keys und Values, sodass
sie nicht für jedes neue Token neu berechnet werden müssen. Das
ändert die Komplexität der Textgenerierung von quadratisch zu
linear in der Sequenzlänge. Bei einer Sequenz von tausend Tokens
ist das eine tausendfache Beschleunigung.

Der Preis dafür ist Speicher. Der Cache wächst mit der
Sequenzlänge und der Modellgröße. Bei kleinen Modellen ist der
Speicherbedarf vernachlässigbar. Bei großen Modellen mit langen
Kontextlängen kann der Cache zig Gigabyte verbrauchen.

Ohne KV-Cache wäre die Textgenerierung so langsam, dass Chatbots
unbenutzbar wären. Jede Antwort würde Minuten statt Sekunden
dauern. Der Cache ist eine notwendige Optimierung für jeden
produktiven Einsatz.
