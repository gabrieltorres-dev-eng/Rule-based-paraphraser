# Rule-Based Humanizing Paraphraser

A paraphraser that rewrites text **without any generative model writing the output**. It changes the sentence structure, paragraph order and word choice of an input, then keeps the candidate that is most structurally different from the original while still reading like human writing.

> **GitHub description:** Rule-based paraphraser that rewrites text with human-sourced templates, dictionaries and corpora, never a generative model. Judges structure with tree-edit distance and burstiness.

---

## Contents

- [The one hard rule](#the-one-hard-rule)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Commands and flags](#commands-and-flags)
- [Reading the report](#reading-the-report)
- [Data and resources](#data-and-resources)
- [Building the data](#building-the-data)
- [Repeat passes](#repeat-passes)
- [Testing](#testing)
- [Limits](#limits)
- [Project layout](#project-layout)

---

## The one hard rule

**Provenance.** Every word in the output must come from one of three places:

1. the input text,
2. a human-written template bank (`data/templates/bank.jsonl`),
3. a human-sourced dictionary or corpus (`data/dictionaries/`, mined from human paraphrase pairs and checked against WordNet).

Closed-class glue words that a rewrite must insert (`and`, `which`, `who`, `the`, `this means`, …) are allowed only if the human corpora attest them (`data/dictionaries/glue.tsv`). If that file is missing, nothing is inserted.

Models are used **only as judges**, never as writers:

| Model | Role |
|---|---|
| GloVe vectors | Check that a swapped word fits its context and is close in meaning |
| spaCy `en_core_web_sm` | Parse sentences into dependency trees |
| KenLM | Score fluency (perplexity), scoring only |
| WordNet, Moby thesaurus | Propose synonym *candidates*; filters decide |

No model proposes text.

---

## Quick start

Requirements: Python 3.10+ and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/build_all.py          # first time only: builds data/ (see below)
```

Paraphrase a file:

```bash
python -m paraphraser run in.txt -o out.txt
```

A more human-like result that changes less:

```bash
python -m paraphraser run in.txt -o out.txt --structure human
```

The strongest setting:

```bash
python -m paraphraser run in.txt -o out.txt --lexical-rate 0.9 --swap-every 4 --candidates 12 -n 2 --save-passes
```

---

## How it works

Every input goes through the same eight stages. Stages 1–7 run once per **candidate** (a version with its own random seed and recipe). Stage 8 chooses which candidate to keep.

```
input ─► 1 Ingest ─► 2 Macro ─► 3 Routing ─► 4 Deflate ─► 5 Lexical ─► 6 Register tells ─► 7 Heal ─► 8 Select ─► output
           ▲                                                                                              │
           └────────────── none passes: stronger transforms, new seeds ◄──────────────────────────────────┘
```

### Stage 1 — Ingest

- **Locking protected spans.** Names, numbers, dates, quotations, citations, URLs and reference sections are replaced by placeholders (`QZ12X`). Rewrites treat a placeholder as an opaque noun, so nothing can split "World War I" or damage a citation. Placeholders are restored at the end, and a candidate is rejected if one leaks into the output.
- The lock is computed once on the original; every candidate starts from a clone of it.
- **Idiom swaps.** Known phrases are replaced with a plainer human equivalent from `idioms.tsv`, about one per `--swap-every` words. Headings are skipped, a phrase is swapped only a few times per document, and the pairs were checked for grammatical fit when mined.

### Stage 2 — Macro (paragraphs and sentence order)

- **Discourse tagging.** Each paragraph gets a role (intro, body, conclusion) from simple cues.
- **Breaking subject chains.** Three or more consecutive sentences with the same subject: the third is rewritten (passive first, clause reorder second).
- **Body swapping.** Independent body paragraphs can be reordered. A paragraph that opens with a causal, sequential or pronoun link is never separated from the paragraph before it.
- **Conclusion relocation.** Thesis and concluding sentences move toward human-typical positions.
- **Sentence swaps.** Neighbouring sentences trade places when neither depends on the other. The first and last sentence of a paragraph stay put.
- **Paragraph resizing.** Paragraphs are split or merged toward the human paragraph-length mix. A paragraph never opens with an additive connective such as "Also,".

### Stage 3 — Routing (rewrites inside and between sentences)

Each rewrite checks a grammar guard on the sentence's dependency tree, then returns new text or nothing. An unsafe rewrite is skipped, never forced.

**Clause and phrase order**

- *Clause transposition:* "Because X, Y" ↔ "Y because X".
- *Prepositional phrase moves:* "… in August 1914" → "In August 1914, …". Connectives such as "On the other hand," never move.
- *Coordination swaps:* "A, and B" → "B, and A" where both halves are independent.
- *Front-to-back and appositives:* a fronted phrase moves behind the main clause; an appositive moves to the front.

**Inside a sentence**

- *Voice:* active ↔ passive, with correct auxiliaries and participles.
- *Possessive ↔ of-phrase:* "the community's use" ↔ "the use of the community".
- *Participle ↔ relative clause:* "microgrids serving towns" ↔ "microgrids that serve towns".
- *Optional "that"* after reporting verbs; reversing three-item lists; swapping two-item pairs.
- *Verb → noun:* "Researchers analyze the data" → "Researchers carry out the analysis of the data", using WordNet-derived nouns only.

**Splitting and merging**

- A sentence joined by "and", "but" or "so" is split, and the second half gets a connective from the human bank ("Also,", "However,", "As a result,").
- "A because B." → "B. That is why A."
- A length sculptor splits and fuses sentences toward the human sentence-length histogram.

**Cross-sentence layers** (run last, since they change the sentence count)

- *Merge neighbours:* two sentences with the same subject become one, joined with "and". Not when the second opens with a link word or uses a copula that would make "is A and is B".
- *Relative clauses:* "… 1917. This would widen…" → "…, which would widen…". Uses "who" for people.
- *Varied openings:* when nearby sentences open the same way, a time or place phrase moves to the front. Never strands "was" without its complement.

A rewrite that would recreate a sentence form from an earlier pass is rejected.

### Stage 4 — Deflate (deletion only)

Removes AI-style inflation. It deletes words and never adds any, so the provenance rule holds.

- Intensifiers: *fundamentally, immensely, drastically, truly* … (kept after a negation).
- Filler adjectives: *overall, mighty, crucial*, and *true/real* only in "a true test".
- Same-meaning pairs: "reshaped and transformed" → "reshaped". Counted as a pair when WordNet puts both in one synset or the embeddings are very close.

### Stage 5 — Lexical (synonym swapping)

Replaces content words with synonyms, filtered:

- **Sense choice.** The WordNet sense is picked by comparing the sentence context with each sense's gloss, so "bank" in a river sentence is not swapped for a money word.
- **Tiers** (loose mode, the default): (1) dominant-sense synonyms, (2) second-sense synonyms, (3) similar-to, see-also and parent senses, (4) Moby thesaurus entries that WordNet confirms for the same part of speech.
- **Filters.** Context fit, closeness of meaning, frequency band (plainer words preferred), and no compounds, hyphenated terms, phrasal verbs or names.
- **Repeat control.** A replacement cannot repeat within a sentence or paragraph and appears at most twice in a document. Words an earlier pass removed are never restored.

### Stage 6 — Register tells

Adds back human irregularities at the rate measured for the chosen register, using rules and template-bank text only:

- *Contractions:* expanded or contracted to match the register's rate.
- *Punctuation mix:* some asides become dash pairs and some semicolons become sentence breaks. Citations and numbers are never touched.
- *Fragments:* a short copular statement can become a fragment ("This is not always true." → "Not always true."), mid-paragraph only.
- *Hedges:* an absolute claim ("always", "proves") can get a hedge frame from the bank.

### Stage 7 — Heal

A rule-based repair pass for leftovers: *a/an* agreement, doubled words and spaces, capitals after sentence ends, stray punctuation, broken ellipses and URLs. Then every placeholder is restored.

### Stage 8 — Select (structural divergence gate)

In aggressive mode, candidates are chosen by **how different their structure is**, not how many words changed. A pure synonym swap scores zero.

**What is measured**

- **Tree edit distance (TED).** Each sentence is parsed to a dependency tree whose nodes carry only a relation and part of speech (e.g. `nsubj|NOUN`), never the word. Selkow's top-down tree edit distance counts the edits between input and output trees, normalised to 0–1. Sentences are first aligned by shared content stems, so splits, merges and reorders are compared against the right partner.
- **Order distance.** Kendall distance over the aligned sentence order.
- **Burstiness.** B = (σ − μ) / (σ + μ) of sentence lengths. Near −1 is very regular, near 0 random-like. Human prose sits in a negative band that depends on register; AI text tends to be too even.

**Targets are sampled, not fixed**

- Each document draws a target TED from the measured distribution of human paraphrase pairs (MRPC, QQP), so documents aim at different levels.
- The floor is the human 20th percentile (override with `--min-ted`).
- The burstiness floor is drawn between the register's 10th percentile and part-way to its median (override with `--min-burst`). Applies to texts with six or more sentences.

**Recipes make each candidate different.** Every candidate gets a random weight for each of 14 transform families (clause, voice, phrase, coord, list, inner, split, complex, para, sent_order, burst, merge, relcl, opening). Some families are switched off, and at least four stay on. One candidate may mostly reorder clauses; another may mostly split and merge.

**Hard gates and the choice**

1. A candidate must have no placeholder leak, TED at or above the floor, and (for 6+ sentences) burstiness at or above its floor.
2. If only burstiness fails, the best candidates get a targeted repair (split the longest sentence or fuse the shortest neighbours).
3. Passing candidates are **sampled** with a softmax over a utility score. It rewards TED up to the target, sentence reordering and total change, with only a small penalty for statistical misfit.
4. If nothing passes, the system **escalates**: transform probabilities are multiplied by 1.4, then 1.8, with fresh seeds. The gate is never loosened. If it still fails, the highest-TED candidate is returned and the report says `passed_structure_gate: false`.

### Statistical checks and hygiene

- **Statistical metrics:** sentence-length histogram distance, sentence-length variance, type-token ratio, stopword ratio, punctuation entropy and KenLM perplexity, each against a human 5th–95th percentile band. In aggressive mode these are tie-breakers, not gates. In `--structure human` mode they are the gates. Texts under 8 sentences are judged on perplexity only.
- **Change ratio:** share of words that differ from the reference, from a word-level diff.
- **Salted seeds:** without `--seed`, each document's seed comes from its random ID, a persistent salt, the clock and OS randomness.
- **N-gram audit:** 8-word spans from earlier outputs are stored; spans that appear in another document's output are counted as *leakage* and penalised.
- **Idiom rotation:** idiom usage is recorded so frequent swaps are used less over time.
- **Logging:** every transform is written to `data/.state/logs/documents.jsonl` with its seed.

---

## Commands and flags

Run from the project folder as `python -m paraphraser <command>`.

| Command | What it does |
|---|---|
| `run in.txt -o out.txt` | Paraphrase one file and print the JSON report |
| `batch indir outdir` | Paraphrase every `.txt` in a folder; models load once, much faster than separate runs |
| `calibrate` | Re-measure human style profiles from `data/corpus` (`--perplexity` also fits the KenLM band) |
| `score in.txt` | Print the statistical metrics of a file without changing it |

| Flag | Effect |
|---|---|
| `--register` | `essay` (default), `blog`, `forum`, `technical` or `academic`: selects the human profile |
| `--structure aggressive` / `human` | Aggressive (default) rearranges as much as it safely can and uses the structural gate. Human matches only the measured base rates |
| `--candidates N` | Candidates per pass (default 4). More gives the gate more to choose from |
| `-n N` / `--repeat N` | Paraphrase the output again, N times in total |
| `--save-passes` | With `-n`, also write each pass as `<output>.passN.txt` |
| `--seed S` | Fix the random draws (default: salted, different each run) |
| `--lexical loose` / `strict` | Loose (default) allows more synonym tiers |
| `--lexical-rate R` | Fraction of eligible content words to try swapping |
| `--swap-every N` | One idiom swap per N words (default 8) |
| `--min-ted X` | Override the structural distance floor (0–1) |
| `--min-burst X` | Override the burstiness floor (−1 to 1) |
| `--no-hygiene` | Skip the n-gram store, usage rotation and logging |
| `--quiet` | Hide the loading report and progress bar |

**Suggested settings**

- *Natural, changes less:* `--structure human --candidates 8 --lexical-rate 0.5 --swap-every 6`
- *Strongest:* `--lexical-rate 0.9 --swap-every 4 --candidates 12 -n 2 --save-passes`

Raising `--min-ted` above about 0.25 will often report a failed gate, because current rewrites top out around there.

---

## Reading the report

The run prints a JSON report. The useful fields:

| Field | Meaning |
|---|---|
| `change_ratio` | Share of words changed from the input (per pass: `change_vs_original`) |
| `structure.ted`, `order`, `burstiness` | Measured structural change |
| `structure.target`, `floor`, `burst_floor` | The targets this document was given |
| `structure.passed_structure_gate` | Whether the hard gate was met |
| `structure.escalations` | How many stronger rounds were needed |
| `structure.recipe` | The transform weights this candidate used |
| `transforms` | How many times each rewrite fired |
| `leakage` | Spans repeated from other documents' outputs |
| `passed_all_gates`, `violations` | The statistical bands; informational in aggressive mode |
| `passes[]` | Per-pass details in repeat mode, including `discarded` |

---

## Data and resources

Everything the system uses lives under `data/`. Only grammar rules and classification lists are hard-coded.

| Resource | What it is | Used for |
|---|---|---|
| `dictionaries/idioms.tsv` | Phrase pairs mined from human paraphrase text, checked against WordNet senses | Idiom swaps (Stage 1) |
| `dictionaries/phrase_pairs.tsv` | Raw mined pairs from human-vs-human sets | Source for the idiom builder |
| `dictionaries/glue.tsv` | Closed-class words with human corpus counts | Gate for inserted glue words |
| `templates/bank.jsonl` | Human-written connectives and templates, tagged by function and register | "Also," / "However," / hedges |
| `profiles/<register>.json` | Measured sentence-length mix, paragraph sizes and transform rates | What "human-like" means per register |
| `profiles/structure.json` | Quantiles of human paraphrase TED, plus burstiness bands per register | Per-document structure targets |
| `corpus/<register>/` | Human text per register | Building profiles, glue counts, burstiness bands |
| `pairs/` | Human paraphrase sets (MRPC, QQP, COCO) | Idiom mining and the structure distribution |
| WordNet, Moby thesaurus | Synonym graphs | Lexical candidates |
| GloVe vectors | Word embeddings | Context-fit and closeness filters |
| KenLM model | N-gram language model | Perplexity scoring |
| `en_core_web_sm` | spaCy parser and tagger | Trees, sentence boundaries, parts of speech |

Heavy models load once per run, and the start-up report shows how long each takes.

---

## Building the data

These scripts run once on your machine. They are the only place outside models are used, and only to classify or look things up.

| Script | Produces |
|---|---|
| `download_data.py` | Corpora, WordNet, GloVe, Moby thesaurus, MRPC/QQP/COCO pairs, KenLM files (`--only <step>` to run part) |
| `build_phrase_pairs.py` | `phrase_pairs.tsv` from human-vs-human pair sets |
| `build_idioms.py` | `idioms.tsv`, keeping only pairs whose WordNet senses and embeddings agree |
| `build_templates.py` | `bank.jsonl`, the human connective and template bank |
| `build_glue.py` | `glue.tsv`, closed-class words seen at least 5 times in human text |
| `build_thesaurus.py` | The cleaned Moby thesaurus for tier-4 candidates |
| `build_fragments.py`, `label_fragments_deepseek.py` | Fragment sample used to calibrate the fragment rate |
| `gen_ai_pairs_deepseek.py` | AI text on the same topics, used only as a lookup key when mining pairs. Never a source of output words |
| `build_structure_profile.py` | `structure.json`: human TED quantiles and burstiness bands |
| `install_kenlm.py` | KenLM for perplexity scoring |
| `build_all.py` | Runs the steps above in order and skips those already done (`--redo <steps>` to rerun some) |

---

## Repeat passes

`-n N` paraphrases the output N times, each with its own seed. Later passes could undo earlier ones, because many rewrites are two-way, so four guards apply:

- **Seen forms:** every sentence form from earlier passes is recorded and cannot be recreated.
- **Anchor to the original:** from pass 2 on, structure and change are measured against the original input.
- **Retired words:** words earlier passes removed cannot return through the lexical stage.
- **Discard on regression:** a pass that leaves the text closer to the original than the previous pass is discarded.

Rewrites are local, so after about two passes the gains are small. More sentence-level rewrite types raise the ceiling more than extra passes do.

---

## Testing

```bash
python -m pytest tests -q
```

The suite covers the structure measures, recipes, the lexical filters, the cross-sentence layers, deflation, repeat passes and an end-to-end run. The KenLM warning in the output is expected when the KenLM model is not installed.

---

## Limits

- **Structural ceiling.** On a 360-word essay, output tree distance reaches about 0.20–0.25. The human median for paraphrase pairs is about 0.45, so outputs are still closer to their source than a typical human paraphrase.
- **Modest change.** An aggressive run changes about 15–25% of words. Repeat passes add little after the second.
- **Source style carries through.** If the input is AI-written, its rhythm and stock phrases largely survive, because the system reorders, splits, merges, swaps and deletes but does not generate new prose.
- **Synonym quality.** Filters block most bad swaps, not all. Domain-heavy text offers few candidates, and the Moby thesaurus may load few entries, so WordNet supplies most of the pool.
- **Parser errors.** A wrong parse can make a rewrite look safe. Guards catch most cases, but read the output.

---

## Project layout

```
paraphraser/          the package
  cli.py              command line
  pipeline.py         runs the stages, candidates, gates, repeat passes
  structure.py        tree edit distance, burstiness, targets, recipes
  intensity.py        transform-probability recipes and seen/retired forms
  ingest/             protected-span locking, idiom swaps
  macro/              paragraph and sentence-order operations
  routing/            sentence-level rewrites and cross-sentence layers
  register/           deflation, contractions, punctuation, fragments, hedges
  lexical/            synonym candidates and filters
  healing/            repair pass
  scoring/            statistical metrics and bands
  hygiene/            seeds, n-gram store, logging
  glue.py             provenance gate for inserted glue words
scripts/              one-time data builders
data/                 dictionaries, templates, profiles, corpora (see above)
tests/                pytest suite
```

---

## Licence and credits

Synonym and parsing resources are used under their own licences (WordNet, Moby, GloVe, spaCy, KenLM, and the MRPC, QQP and COCO datasets). Check each before redistributing data.
