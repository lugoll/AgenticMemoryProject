# RAG-Benchmark: Ergebnisse & Analyse

**Datum:** 2026-05-25  
**Kontext:** MKBA SS26 — RAG-Benchmark auf HotpotQA (BM25 / Vector / Graph)  
**Datensatz:** HotpotQA distractor split, N=100 (50 bridge + 50 comparison), seed=42

---

## 1. Wie die Evaluation funktioniert

### 1.1 Metriken im Überblick

Das Evaluation-Script (`04_evaluate.py`) berechnet drei Metriken pro Antwort:

**Exact Match (EM)**  
Normalisierter String-Vergleich: Kleinbuchstaben, Satzzeichen entfernt, Artikel (*a/an/the*) entfernt.  
`"Unbreakable."` und `"unbreakable"` zählen als gleich. EM ist binär (0 oder 1).

**F1-Score (Token-Overlap)**  
Misst den Anteil gemeinsamer Tokens zwischen Modell-Antwort und Referenzantwort.  
Precision = gemeinsame Tokens / Tokens in Modell-Antwort  
Recall = gemeinsame Tokens / Tokens in Referenzantwort  
F1 = harmonisches Mittel. Robuster als EM bei paraphrasierten Antworten.

**LLM-as-Judge (Verdict)**  
Wird nur aufgerufen wenn: Fragetyp = `bridge` **und** EM = false.  
Der Judge (qwen2.5:14b) erhält Frage, Referenzantwort und Modell-Antwort und gibt eines zurück:
`CORRECT` / `PARTIAL` / `INCORRECT`

Comparison-Fragen (Ja/Nein) werden nicht dem Judge vorgelegt — bei EM=false ist die Antwort
im Regelfall einfach falsch.

### 1.2 Judge-Prompt (Grading-Regeln)

```
1. INCORRECT — Modell sagt "I don't know", gibt falsche Antwort, widerspricht.
2. CORRECT   — Modell-Antwort enthält die korrekte Antwort, auch mit Zusatzworten.
               Extra-Kontext degradiert NICHT zu PARTIAL.
               Synonyme und Umformulierungen zählen als CORRECT.
3. PARTIAL   — Genuinely incomplete: z.B. eine von zwei nötigen Personen genannt.
```

### 1.3 Aggregierte Accuracy-Formel

```
correct_count = EM-Treffer
              + Judge=CORRECT   (bridge, EM=false)
              + 0.5 × Judge=PARTIAL (bridge, EM=false)

accuracy_judge = correct_count / N
```

**Wichtig:** EM=true-Fälle haben `judge_verdict = null` — das ist by design.
Der Judge wird nicht aufgerufen wenn EM schon true ist (spart Tokens).
In der Aggregation werden EM=true-Fälle über `em_count` gezählt,
der Judge-Loop überspringt sie (`if not exact_match`).
Es gibt keine Doppelzählung und kein fehlendes Ergebnis.

---

## 2. Ergebnisse (N=100, Stand: 2026-05-25)

### 2.1 Haupttabelle

| Variante | EM gesamt | F1 | IDK-Rate | Query-Tokens∅/q | Ingest-Tokens gesamt | Ingest-Zeit |
|---|---|---|---|---|---|---|
| **Vector** | **51 %** | **0.613** | 12 % | 641 | 0 (kein LLM) | ~72 s |
| **BM25**   | 36 % | 0.481 | 27 % | 630 | 0 (kein LLM) | < 5 s |
| **Graph**  | 19 % | 0.212 | 70 % | **195** | **~1.000.000** | ~100 Min |

Query-Tokens = Prompt + Completion, gemessen über N=100 Fragen (990 Dokumente ingested).  
Graph-Ingest-Tokens: 990 Dokumente × ~1.000 Tokens (Prompt + Completion, max_tokens=1024).  
Ingest-Telemetrie-Datei (setup.json) wurde im Rahmen der Evaluation-Bereinigung gelöscht;
Schätzung basiert auf Chunk-Größe × Dokumentanzahl und ist konsistent mit der Laufzeit.

**Token-Ökonomie:** Graph spart 446 Tokens/Frage gegenüber Vector (195 vs. 641).
Der Ingest-Vorsprung von ~1.000.000 Tokens wäre erst nach ~2.240 Fragen amortisiert —
und das bei deutlich schlechterer Antwortqualität (19 % vs. 51 % EM).

### 2.2 Aufschlüsselung nach Fragetyp

| Variante | EM bridge (N=50) | EM comparison (N=50) |
|---|---|---|
| **Vector** | **58 %** | 44 % |
| **BM25**   | 30 % | **42 %** |
| **Graph**  | 18 % | 20 % |

### 2.3 Graph: Fehleranalyse (70 IDK-Antworten)

| Kategorie | Anzahl | Anteil | Ursache |
|---|---|---|---|
| Antwort **nicht im Kontext** | **49** | **70 %** | Retrieval-Failure — Graph hat die Information gar nicht geliefert |
| Antwort **im Kontext**, IDK trotzdem | 21 | 30 % | LLM-Reasoning-Failure — Triple-Format zu abstrakt |

Beide Kategorien liegen auf demselben Kontext-Niveau: mittlere Snippet-Anzahl = 10,0 bei
IDK-Fällen, 10,0 bei beantworteten Fragen — der Graph liefert immer exakt `top_k=10` Triples.
Das Problem ist nicht *wie viel* Context, sondern *welcher* Content darin steht.

---

## 3. Interpretation & wissenschaftliche Einordnung

### 3.1 Vector gewinnt — und das ist nicht überraschend

Das Ergebnis ist kein Fehler. Es ist die zu erwartende Konsequenz der Datensatz-Wahl.

Yang et al. (2018) beschreiben im HotpotQA-Paper explizit den Design-Gegensatz zu
wissensbasenbasierten Datensätzen:

> *"datasets that target multi-hop reasoning are constructed using existing knowledge bases (KBs).
> As a result, these datasets are constrained by the schema of the KBs they use, and therefore
> the diversity of questions and answers is inherently limited."*

HotpotQA wurde als **Text-Span-Task** konstruiert: Die Antwort ist ein direktes Zitat
aus dem Wikipedia-Paragraphen. Wer den Originaltext in Triples komprimiert, verliert
exakt die Information, die als Antwort erwartet wird — vor allem spezifische Attributwerte
wie Geburtsdaten, Jahreszahlen und Namen von Nebenfiguren.

**Vector RAG gibt den Originaltext unverändert weiter.** Das ist der entscheidende Vorteil:
kein Informationsverlust durch Kompression.

### 3.2 Die Token-Effizienz von Graph ist kein Vorteil für HotpotQA

Graph verbraucht im Schnitt nur **189 Prompt-Tokens** pro Anfrage — gegenüber ~630 bei
BM25 und Vector. Das klingt effizient. Die Kontext-Zeichen erzählen die Wahrheit:

| Variante | Prompt-Tokens∅ | Kontext-Zeichen∅ | Snippets∅ |
|---|---|---|---|
| BM25 / Vector | ~625 | ~2 300 | 5 Textpassagen |
| Graph | 189 | 427 | 10 Triples |

Graph liefert **5× weniger Zeicheninhalt** trotz doppelt so vieler Snippets — weil Triples
strukturell kurz sind (`"Larry Fedora born_on September 10, 1962"`). BM25 und Vector liefern
vollständige Textpassagen mit Satz-Kontext. Für HotpotQA ist Satz-Kontext der Inhalt der
Antwort — Triples sind eine Zusammenfassung auf Kosten der Antwortpräzision.

Die Token-Einsparung bei Graph ist deshalb kein Effizienzgewinn, sondern ein
Qualitätsverlust. Das stimmt mit dem Befund von Couto & Ebecken (2025) überein:
Auch auf Finanzdaten (10-K Reports) — einer für GraphRAG strukturell günstigeren Domäne
mit klaren Entitäten und numerischen Relationen — gewinnt Vector RAG knapp:

| Benchmark | Vector | Graph | Domäne |
|---|---|---|---|
| Couto 2025 | 91.67 % | 90 % | Finanzdaten, multiple choice |
| Dieser Run | **51 %** | 19 % | HotpotQA, open-end faktoid |

Der größere Abstand bei HotpotQA ist direkt auf die Text-Span-Natur des Benchmarks
zurückzuführen.

### 3.3 Warum BM25 gut abschneidet — und was das über den Ansatz sagt

BM25 erzielt 36 % EM ohne jeden Ingest-Aufwand und ohne LLM bei der Indexierung.
Bei Comparison-Fragen (42 %) ist es fast gleichauf mit Vector (44 %).

Das ist konsistent mit dem State-of-the-Art auf HotpotQA: Trivedi et al. (2022) zeigen,
dass **IRCoT** (Interleaved Retrieval with Chain-of-Thought) mit BM25 als Retrieval-Basis
auf HotpotQA bis zu 15 F1-Punkte über Standard-RAG erreicht — ohne Graph, ohne spezielle
Einbettungen. Das Prinzip:

```
1. BM25-Retrieval mit Original-Query
2. LLM generiert ersten Reasoning-Step: "Lost Gravity was built by Mack Rides."
3. Dieser Satz wird neue BM25-Query
4. Weiterer Retrieval-Schritt → Mack-Rides-Artikel
5. Iteration bis Antwort gefunden
```

IRCoT löst das Multi-Hop-Problem nicht durch bessere Indexstruktur, sondern durch
**iteratives Retrieval mit expliziter Bridge-Entität**. Das ist der Ansatz, den unser
Graph mit BFS versucht zu modellieren — aber IRCoT bleibt im Originaltext statt
den Text zu komprimieren.

### 3.4 Die 70 % IDK-Rate des Graphen: zwei strukturell verschiedene Probleme

Von 70 IDK-Antworten des Graph-Modells haben 49 (70 %) die Antwort **nicht im Kontext** —
das ist Retrieval-Failure, nicht LLM-Failure. Die restlichen 21 (30 %) haben die Antwort
als Triple im Kontext, können sie aber nicht extrahieren.

Das zweite Problem (21 Fälle) deutet darauf hin, dass Triples wie
`"North Carolina Tar Heels football team led_by Larry Fedora"` für das LLM schwer
zu interpretieren sind, wenn die Frage *"In welchem Monat wurde der Trainer geboren?"* lautet —
der Kontext enthält die Person, aber nicht das gesuchte Attribut.

Das erste Problem (49 Fälle) ist das strukturelle Bottleneck: Der Graph-BFS findet den
Pfad zur Antwort-Entität nicht innerhalb von `max_hops=3`, weil viele HotpotQA-Antworten
atomare Attributwerte sind, die keine eigenständigen Knoten im Graph bilden.

### 3.5 Microsoft GraphRAG (Edge 2024) löst ein anderes Problem

Das Microsoft-Paper beschreibt GraphRAG für **Query-Focused Summarization** —
Fragen wie *"What are the main themes across this corpus?"*. Das ist das strukturelle
Gegenteil von HotpotQA: Community-Summaries und Themen-Traversal helfen bei
Überblicks-Fragen, nicht bei präzisen Faktoid-Fragen mit exaktem Antwort-String.

Die Architektur-Übertragung (Graph für Multi-Hop-QA) ist wissenschaftlich motiviert
und experimentell testbar — das Ergebnis zeigt jedoch, dass der Einsatzbereich von
GraphRAG klar begrenzt ist.

---

## 4. Kernaussage (für die Präsentation)

> *GraphRAG verschiebt Token-Aufwand vom Retrieval in den Ingest — zahlt sich aus,
> wenn Fragen die Corpus-Struktur benötigen (Edge 2024: Summarization, Sensemaking).
> Für faktoid-basiertes Multi-Hop-QA auf Text-Span-Ebene ist Vector RAG überlegen,
> weil es den Originaltext verlustfrei übergibt. HotpotQA wurde explizit als Text-Span-Task
> designed, um die Beschränkungen KB-basierter Datensätze zu vermeiden (Yang 2018).*

### Limitationen (müssen benannt werden)

- Benchmark-Wahl begünstigt Vector RAG strukturell
- Auf Summarization-Benchmarks (QASPER, NarrativeQA) wäre das Ergebnis möglicherweise
  umgekehrt
- Lokale Modelle (llama3.1:8b, qwen2.5:14b) — kein direkter Vergleich mit kommerziellen
  Benchmark-Ergebnissen
- N=100 reicht für Aussagen auf Variantenebene; für Signifikanztests wäre N=500 nötig
  (±3.5 PP Konfidenzintervall bei α=0.05)

---

## 5. Quellen

| Kürzel | Vollreferenz | Relevanz |
|---|---|---|
| Yang 2018 | Yang et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP 2018. | Benchmark-Design, Text-Span-Rationale |
| Edge 2024 | Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130, 2024. | GraphRAG Originalansatz (Summarization, nicht Faktoid-QA) |
| Trivedi 2022 | Trivedi et al. *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.* ACL 2023. | State-of-the-Art HotpotQA, IRCoT |
| Couto 2025 | Couto & Ebecken. *Graph RAG vs. Vector RAG: A Performance Comparison in Response Generation.* CILAMCE 2025. | Empirischer Vergleich GraphRAG vs. VectorRAG, Finanzdomäne |
| Bordes 2013 | Bordes et al. *Translating Embeddings for Modeling Multi-Relational Data (TransE).* NeurIPS 2013. | Begründung MultiDiGraph-Architektur |
