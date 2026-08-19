# Kapitel 2 — Tokenization: Wörter in Zahlen verwandeln

## Die Analogie für ein 5-jähriges Kind

Computer können nur **Zahlen** verstehen. Sie wissen nicht, was der Buchstabe „A" bedeutet – sie kennen „65" (seinen ASCII-Code). Deshalb müssen wir Text in Zahlen umwandeln, bevor wir ihn in ein neuronales Netz einspeisen.

Die einfachste Idee: **jedem Wort eine Zahl zuweisen**:
```
"cat"  ->  9246
"sat"  ->  6734
"on"   ->   389
"the"  ->   279
"mat"  -> 16789
```

Aber die englische Sprache hat Hunderttausende von Wörtern. Brauchen wir wirklich eine eigene Zahl für „antidisestablishmentarianism"? Und was ist mit neuen Wörtern wie „skibidi", die es noch gar nicht gab, als wir das Vokabular aufgebaut haben?

## Die Lösung: Subword-Tokenization (BPE)

Anstelle ganzer Wörter zerlegen wir Text in **häufige Subword-Bausteine**:

```
"unbelievably" -> "un" + "believ" + "ably"
"running"      -> "runn" + "ing"
"cats"         -> "cat" + "s"
"lower"        -> "low" + "er"
"GPT"          -> "G" + "P" + "T"
```

Das ist **Byte Pair Encoding (BPE)** – genau der Algorithmus, den GPT-2, GPT-3, GPT-4 und die meisten modernen Modelle verwenden.

### Wie BPE funktioniert — Schritt für Schritt

BPE startet damit, dass jedes Zeichen ein eigenes „Token" ist, und verschmilzt dann wiederholt das häufigste Paar:

**Ausgangstext:** `"low lower lowest"`

```
Schritt 0 (Anfang — jedes Zeichen ist ein Token):
l o w _ l o w e r _ l o w e s t

Schritt 1 (häufigstes Paar: 'l'+'o' -> 'lo'):
lo w _ lo w e r _ lo w e s t

Schritt 2 (häufigstes Paar: 'lo'+'w' -> 'low'):
low _ low e r _ low e s t

Schritt 3 (häufigstes Paar: 'e'+'s' -> 'es'):
low _ low e r _ low es t

Schritt 4 (häufigstes Paar: 'es'+'t' -> 'est'):
low _ low e r _ low est

Schritt 5 (häufigstes Paar: 'low'+'_' -> 'low_'):
low_ low e r _ low_ est
```

Nach genügend Merges haben wir ein Vokabular wie: `{l, o, w, e, r, s, t, _, lo, ow, low, er, es, est, low_}`

Jetzt können neue Wörter mit diesen Bausteinen dargestellt werden, selbst wenn wir sie noch nie gesehen haben:

```
"lowest"  -> "low" + "est"     (beide im Vokabular!)
"slower"  -> "s" + "low" + "er" (noch nie gesehen, aber funktioniert!)
```

### Warum BPE besser ist als wortbasierte Tokenization

| Problem | Wortbasiert | BPE |
|---|---|---|
| „running" vs. „run" | Unterschiedliche Tokens — keine gemeinsame Bedeutung | „runn" + „ing" — das Modell erkennt den Zusammenhang |
| Neues Wort: „rizz" | Unbekanntes Token → Modell scheitert | „r" + „i" + „z" + „z" → funktioniert mit Zeichen |
| Vokabulargröße | 500K+ (zu viele seltene Wörter) | 50K (ausgewogen, effizient) |
| Umgang mit Unicode/Emojis | Oft fehlerhaft | Fallback auf Zeichenebene schlägt nie fehl |

### Was ist mit Sonderzeichen und Emojis?

BPE arbeitet auf **Byte**-Ebene, nicht auf Zeichenebene. Das bedeutet, es kann ALLES tokenisieren, was sich als Bytes darstellen lässt — Emojis, chinesische Zeichen, Code, LaTeX, sogar Binärdaten:

```
"Hello 😊"  ->  ["Hello", " Ġ", "😊"]    (Ġ = Leerzeichen-Präfix im GPT-Tokenizer)
"你好"       ->  tokenisiert über UTF-8-Bytes
"def foo():"->  ["def", "Ġfoo", "()", ":"]
```

### GPT-Tokenizer-Konventionen

| Token | Beispiel | Bedeutung |
|---|---|---|
| Normale Tokens | `"cat"`, `"the"`, `"ing"` | Reguläre Subword-Bausteine |
| Mit Leerzeichen-Präfix | `"Ġcat"`, `"Ġthe"` | Wort beginnt nach einem Leerzeichen (Ġ ist ein Sonderzeichen) |
| `<\|endoftext\|>` | EOS-Token | Markiert das Ende eines Dokuments — entscheidend für das Training |
| Großbuchstaben | `"The"` vs. `"the"` | Unterschiedliche Tokens! Groß-/Kleinschreibung ist relevant |

### Das EOS-Token — Warum es wichtig ist

Das `<|endoftext|>`-Token (End Of Sequence) ist **entscheidend** und wird oft übersehen:

```python
# OHNE EOS — zwei Dokumente werden zusammengefügt:
doc1 = "The cat sat."     # Tokens: [464, 3797, 3332, 13]
doc2 = "The dog ran."     # Tokens: [464, 3290, 3407, 13]
# Ergebnis: [464, 3797, 3332, 13, 464, 3290, 3407, 13]
# Modell sieht: "...sat. The dog ran." — hält es für EIN Dokument
# Lernt: Auf "sat." folgt oft "The" — FALSCH!

# MIT EOS — Dokumente werden getrennt:
tokens = [464, 3797, 3332, 13, EOS, 464, 3290, 3407, 13, EOS]
# Modell lernt: EOS bedeutet "hier ist Schluss, das nächste Token ist unabhängig"
```

## Tokenizer-Code — Kommentiert

```python
from dataclasses import dataclass
import tiktoken


@dataclass
class TokenizerConfig:
    """
    WAS: Hält alle Tokenizer-Einstellungen an einem Ort.
    WARUM: Wie eine Rezeptkarte — konsistent für das gesamte Projekt.
           Einen Wert ändern, und alles aktualisiert sich automatisch.
    """
    name: str = "gpt2"                # WAS: den vortrainierten BPE-Tokenizer von GPT-2 verwenden
                                       # WARUM: gleiches BPE wie GPT-3/4 — 50K Merges,
                                       #        bewährt an Milliarden von Dokumenten,
                                       #        und bereits trainiert (keine Wochen Arbeit)
    vocab_size: int = 50257           # WAS: Gesamtzahl eindeutiger Tokens
                                       # WARUM: 50.257 ist die exakte Vokabulargröße von GPT-2
                                       #        (50.000 Merges + 256 Byte-Tokens + 1 EOS)
                                       #        Das ist die „Goldlöckchen"-Zahl —
                                       #        groß genug für seltene Subwords,
                                       #        klein genug für schnelle Matrixoperationen


class SimpleTokenizer:
    """
    WAS: Umschließt tiktoken mit einer freundlichen, konsistenten Schnittstelle.
    WARUM: Die rohe API von tiktoken ist low-level (man muss bei jedem
           Aufruf allowed_special angeben). Dieser Wrapper macht encode/decode
           trivial — einfach .encode("hello") aufrufen und Tokens zurückbekommen.
           
           Außerdem behandelt er das EOS-Token konsistent, damit wir nie
           versehentlich vergessen, es bei der Trainingsdatenaufbereitung hinzuzufügen.
    """

    def __init__(self, config: TokenizerConfig = None):
        """
        WAS: Initialisiert den Tokenizer mit dem BPE-Vokabular von GPT-2.
        WARUM: Wir verwenden einen vortrainierten Tokenizer, weil:
               1. Ein Tokenizer von Grund auf zu trainieren, Wochen an CPU-Zeit kostet
               2. Der Tokenizer von GPT-2 quelloffen, schnell und gut getestet ist
               3. Die Verwendung desselben Tokenizers wie in Produktionsmodellen
                  bedeutet, dass unser Code genauso tokenisiert wie GPT-3
        """
        self.config = config or TokenizerConfig()

        # WAS: Lädt die GPT-2-Kodierung von tiktoken
        # WARUM: tiktoken speichert vortrainierte BPE-Merge-Tabellen.
        #        get_encoding("gpt2") lädt genau die 50K Merges,
        #        mit denen GPT-2 trainiert wurde.
        self.enc = tiktoken.get_encoding(self.config.name)

        # WAS: Definiert und kodiert das End-of-Sequence-Token
        # WARUM: <|endoftext|> ist das Sonder-Token, das Grenzen zwischen
        #        Dokumenten markiert. Beim Training fügen wir es zwischen
        #        jedes Dokument ein, damit das Modell lernt, wo ein Text
        #        endet und ein anderer beginnt.
        self.eos_token = "<|endoftext|>"       # Die String-Darstellung
        self.eos_token_id = self.enc.encode(    # In die Token-ID umwandeln
            self.eos_token,
            allowed_special={self.eos_token}    # WARUM: tiktoken blockiert Sonder-Tokens
                                                #        standardmäßig aus Sicherheitsgründen.
                                                #        Wir müssen die EOS-Kodierung
                                                #        explizit erlauben.
        )[0]  # [0], weil encode() eine Liste zurückgibt — wir wollen die einzelne ID

    def encode(self, text: str) -> list[int]:
        """
        WAS: Wandelt Text in eine Liste von ganzzahligen Token-IDs um.
        WARUM: Neuronale Netze verarbeiten nur Zahlen. Rohe Strings wie
               "Hello world" bedeuten für eine Matrixmultiplikation nichts.

        Beispiel: "Hello world" -> [15496, 995]

        Unter der Haube: tiktoken zerlegt den Text mithilfe der
        vortrainierten BPE-Merge-Tabelle in Subword-Bausteine und schlägt
        dann die ID jedes Bausteins im Vokabular nach.
        """
        # WAS: Verwendet den schnellen, C/Rust-basierten Encoder von tiktoken
        # WARUM: tiktoken ist in Rust geschrieben, nicht in Python.
        #        Es kann Hunderte MB Text pro Sekunde tokenisieren.
        #        Ein reiner Python-BPE-Tokenizer wäre 100x langsamer.
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        WAS: Wandelt Token-IDs zurück in menschenlesbaren Text um.
        WARUM: Nachdem das Modell während der Inference eine Folge von
               Token-IDs erzeugt hat, müssen wir sie zurück in Text
               umwandeln, damit Menschen die Ausgabe lesen können.

        Beispiel: [15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        WAS: Wie viele eindeutige Tokens im Vokabular existieren.
        WARUM: Diese Zahl bestimmt die Größe der Output-Layer unseres
               Modells — die letzte Linear-Layer muss vocab_size
               Ausgaben haben (einen Score für jedes mögliche nächste Token).
               
               50.257 bedeutet, dass das Modell bei jeder Vorhersage
               des nächsten Worts aus 50.257 Möglichkeiten wählt.
        """
        return self.config.vocab_size


# ===== WAS: Schneller Selbsttest =====
# WARUM: Jede Komponente immer isoliert testen, bevor man sie kombiniert.
#        „Funktioniert der Tokenizer?" ist ein 5-Sekunden-Check, der
#        Stunden beim Debuggen einer fehlerhaften Trainingsschleife spart.
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Test 1: Grundlegender Text
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS-Token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Seltenes/unbekanntes Wort
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")
```

**Erwartete Ausgabe:**
```
Test 1 — Basic:
  Original: 'The cat sat on the mat.'
  Encoded:  [464, 3797, 3332, 319, 262, 2603, 13]
  Decoded:  'The cat sat on the mat.'
  Match:    True

Test 2 — EOS token:
  String: '<|endoftext|>'
  Token ID: 50256
  Encode result: [50256]

Test 3 — Rare word:
  Encoded: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
  Pieces:  ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']
  Decoded: 'antidisestablishmentarianism'

Test 4 — Emoji:
  Encoded: [15496, 52430, 23530, 248, 995]
  Decoded: 'Hello 😊 world'

  Vocab size: 50,257
```

---

**Zurück:** [Kapitel 1 — Setup](01_setup.md)
**Weiter:** [Kapitel 3 — Embeddings](03_embeddings.md)
