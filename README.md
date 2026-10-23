How the Rule-Based Humanizing Paraphraser Works
Oct 22, 2026 · @Gabe
What it is and the one hard rule
The paraphraser rewrites a text by rule: no generative model ever writes a word of the output. Every word that appears in the result either came from your input, from a human-written template bank, or from a human-sourced dictionary or corpus. This is the provenance rule, and the whole design bends around it.
Models are still used, but only as judges. GloVe word vectors check that a swapped word still fits its sentence. MiniLM and NLI models compare meaning. KenLM scores how fluent the text is. None of them propose text.
The goal is a rewrite that is structurally different from the input and does not read like AI output. Two ideas follow from that:
• Structure matters more than vocabulary. Swapping synonyms barely changes how a sentence is built. The system therefore reorders clauses, splits and merges sentences, moves paragraphs, and then measures how different the sentence structure is, and refuses outputs that are not different enough.
• Humans are the reference. Targets (sentence-length mix, how bursty the sentences are, how different a paraphrase should be) are measured from real human writing and real human paraphrase pairs, not guessed.
The pipeline at a glance
Every input goes through the same stages, in this order. The whole run is repeated for several candidates (different random seeds), and only the best candidate that passes the gates is kept.
1. Ingest: protect names, numbers, citations and references; apply idiom swaps from the human dictionary.
2. Macro: reorder at paragraph level and swap neighbouring sentences.
3. Routing: rewrite sentence structure (clause order, voice, phrase moves, splits, merges, relative clauses, varied openings).
4. Deflate: delete AI-style intensifiers, filler adjectives and same-meaning word pairs.
5. Lexical: swap content words for filtered synonyms.
6. Register tells: contractions, punctuation mix, fragments and hedges drawn from human patterns.
7. Heal: repair spacing, capitals and punctuation damage; restore protected spans.
8. Select: measure each candidate against the input and keep one that passes the structural gate.
The selection step is where candidates compete. A candidate must reach a minimum dependency-tree edit distance from the input and a minimum burstiness of sentence lengths. If none passes, the system raises the strength of the transforms and tries again, rather than lowering the bar.
The dashed arrow is the retry loop: when no candidate clears the gate, the pipeline strengthens the transforms and draws fresh candidates.
Resources and data
Everything the system knows lives in files under data/. Nothing is hard-coded except grammar rules and lists that only classify text (for example, which words count as intensifiers). Anything that gets inserted into output must trace back to a human source.
Resource
What it is
What it is used for
dictionaries/idioms.tsv
689 phrase pairs mined from human paraphrase text and checked against WordNet senses
Phrase-level swaps in Ingest
dictionaries/phrase_pairs.tsv
1,427 raw mined pairs (human-vs-human sets such as MRPC, QQP, COCO)
Source material the idiom builder filters
dictionaries/glue.tsv
25 closed-class words and phrases (and, which, who, this means, …) with human corpus counts
Gate for any glue word a rewrite inserts
templates/bank.jsonl
302 human-written connectives and templates, tagged by function and register
Transitions such as "Also," or "However,"; joiners; hedges
profiles/<register>.json
Measured sentence-length mix, paragraph sizes and transform rates for essay, blog, forum, technical, academic
What "human-like" means for each register
profiles/structure.json
101 quantiles of structural distance between human paraphrase pairs, plus per-register burstiness bands
Per-document structure targets and floors
corpus/<register>/
Human text per register
Building profiles, glue counts and burstiness bands
pairs/
Human paraphrase sets (MRPC, QQP, COCO)
Idiom mining and the structure distribution
WordNet, Moby thesaurus
Synonym graphs
Candidate words for the lexical stage
GloVe vectors
Word embeddings
Judge whether a swap fits its context
KenLM
N-gram language model
Perplexity (fluency) scoring only
spaCy en_core_web_sm
Parser and tagger
Dependency trees, sentence boundaries, parts of speech
All heavy models load once per run and the report at start-up shows the time each one takes. They only score and filter; they never write.
Stage 1: Ingest
Locking protected spans. Before anything is rewritten, names, numbers, dates, quotations, citations, URLs and whole reference sections are replaced by placeholder tokens of the form QZ12X. Rewrites treat a placeholder as an opaque noun, so nothing can split "World War I" or damage a citation. At the very end every placeholder is swapped back, and a check refuses any candidate where a placeholder leaked into the output.
The lock is computed once on the original text. Each candidate starts from a clone of that locked base, which is a large speed saving.
Idiom swaps. The idiom dictionary maps a phrase to a plainer or differently built human equivalent (for example a long preposition phrase to a shorter one). The swapper scans the text for known phrases and replaces roughly one per N words (--swap-every, default 8). Safeguards:
• Headings are skipped.
• A single phrase can be swapped only a few times per document, so the same replacement does not repeat.
• The pair must be grammatically compatible with its slot, which the dictionary builder checked when it was mined.
Stage 2: Macro (paragraphs and sentence order)
The macro stage changes how the document is organised, before any single sentence is touched.
• Discourse tagging. Each paragraph gets a role (intro, body, conclusion) from simple cues. A detector flags the typical AI shape of background, then elaboration, then solution.
• Breaking subject chains. When three or more sentences in a row share a subject, the third is rewritten (passive first, clause reorder second) so the repetition breaks.
• Body swapping. Body paragraphs that do not depend on each other are reordered. A paragraph that opens with a causal, sequential or pronoun link ("Because of this…", "It…") is never separated from the paragraph before it.
• Conclusion relocation. Thesis and concluding sentences are moved toward a position typical of human writing instead of always sitting last.
• Sentence swaps. Neighbouring sentences inside a paragraph can trade places when neither starts with a link word or pronoun. The first and last sentence of every paragraph stay put so each paragraph still opens and closes properly.
• Paragraph resizing. Paragraphs are split or merged toward the paragraph-length mix measured from human text. A paragraph may not start with an additive connective such as "Also,"; such a sentence is pulled back into the previous paragraph.
In the default aggressive mode these steps run at high probability. In human mode (--structure human) they run only at the base rates measured from human writing.
Stage 3: Routing (rewrites inside and between sentences)
Each rewrite takes a sentence, checks a grammar guard on its dependency tree, and returns new text or nothing. A rewrite that cannot be done safely is skipped rather than forced. Any word a rewrite adds must come from the sentence itself, the template bank, or the glue list.
Clause and phrase order
• Clause transposition: "Because X, Y" becomes "Y because X" and back. It refuses sentences with nested subordination or pronouns that would lose their referent.
• Prepositional phrase moves: a trailing time or place phrase moves to the front ("… in August 1914" becomes "In August 1914, …"), and fronted ones move back. Connective phrases such as "On the other hand," never move.
• Coordination swaps: "A, and B" clauses trade places.
• Front-to-back and appositives: a fronted phrase after the last comma moves behind the main clause; an appositive moves to the front.
Inside a sentence
• Voice: active to passive with correct auxiliaries and participles.
• Possessive and of-phrase: "the community's use" and "the use of the community".
• Participle and relative clause: "microgrids serving towns" and "microgrids that serve towns".
• The optional "that" after reporting verbs; reversing three-item lists; swapping two-item pairs.
• Verb to noun: "Researchers analyze the data" becomes "Researchers carry out the analysis of the data", using only WordNet-derived nouns and the glue words "carry out", "the" and "of".
Splitting
• A sentence joined by "and", "but" or "so" is split in two and the second half gets a connective taken from the human bank ("Also,", "However,", "As a result,").
• "A because B" becomes "B. That is why A."
• "A. This means B." becomes "A, which means B."
• A length sculptor splits and fuses sentences until the sentence-length mix matches the human histogram for the register.
Sentence forms that already appeared in an earlier pass are tracked, and a rewrite that would only recreate one is rejected.
Cross-sentence layers
Three rewrites work across neighbouring sentences. They run last in the routing stage, because they change how many sentences there are. Each is a separate setting in the random recipe, so documents get different mixes.
• Merge neighbours. Two sentences with the same subject become one: "Canada would field 600,000 men. Canada would send a force to Europe." becomes "Canada would field 600,000 men and would send a force to Europe." It will not merge when the second sentence opens with a link word, or when the second sentence uses a copula ("is", "was") that would give a chain of "is A and is B".
• Relative clauses. A sentence that starts with "This…" or repeats the previous sentence's last noun is folded into it. "… the Conscription Crisis of 1917. This would widen the divide…" becomes "… the Crisis of 1917, which would widen the divide…". It uses "who" for people. Both "which" and "who" must be in the glue list.
• Varied openings. If nearby sentences open the same way, or a sentence starts with "It", "This" or "There", a time or place phrase is moved to the front. "Canadian soldiers would gain a reputation … on the Western Front" gets "On the Western Front," in front. It never strands a verb such as "was" without its complement.
Why they matter: these change the shape of the tree for several sentences at once, which is what raises the structural distance score that selection checks.
Stage 4: Lexical (synonym swapping)
The lexical stage replaces content words (nouns, verbs, adjectives) with synonyms. WordNet and the Moby thesaurus propose candidates; filters decide whether any is accepted. Nothing is generated.
Choosing the sense. Most words have several meanings. For each word the system compares the sentence context (average GloVe vector of the other content words) with the dictionary gloss of each WordNet sense and keeps the best-fitting sense. This is what stops "bank" in a river sentence from being swapped for a money word.
Candidate tiers. In loose mode (the default) candidates come from four tiers, tried with stricter closeness limits as they get looser:
1. The dominant sense's synonyms.
2. The second sense's synonyms.
3. Related senses: similar-to, see-also and the parent category.
4. Moby thesaurus entries that list each other, and that WordNet confirms for the same part of speech.
Filters. A candidate is dropped unless:
• it fits the sentence context about as well as the original (embedding similarity);
• it is close in meaning to the original (embedding closeness above a threshold);
• it is not much rarer than the original, and plainer words are preferred (word frequency from the wordfreq list);
• it is not the same word in another form, not part of a compound or hyphenated term, not a phrasal verb, and not a name.
Rare or formal words are always attempted, because those are the typical AI-sounding ones. Words that are frequent in the document are treated as domain vocabulary and left alone.
Repeat control. A replacement word cannot already be in the same sentence, cannot repeat inside a paragraph, and can appear at most twice in the document. In repeat passes, words an earlier pass removed from the text can never be brought back. Each replacement is locked as a placeholder so later stages cannot change it again.
Register tells and AI-ism deflation
Deflation (deletion only). AI text tends to over-inflate. The deflate step removes words and never adds any, so the provenance rule cannot be broken. It deletes:
• intensifier adverbs such as fundamentally, immensely, drastically, truly (kept after a negation, so "did not fundamentally change" is untouched);
• filler adjectives such as overall, mighty, crucial, and true or real only in the pattern "a true test";
• same-meaning pairs: "reshaped and transformed" becomes "reshaped". Two words count as a pair when WordNet puts them in one synset or their embeddings are very close.
Register tells. Human writing has small irregularities that AI text smooths away. This step adds them back at the rate measured from the chosen register, using rules and human bank templates only:
• Contractions: expand or contract ("do not" and "don't") to match the register's contraction rate.
• Punctuation mix: some parenthetical asides become dash pairs and some semicolons become sentence breaks. Citations and numbers are never touched.
• Fragments: a short copular statement can become a fragment ("This is not always true." becomes "Not always true."), mid-paragraph only.
• Hedges: an absolute claim ("always", "never", "proves") can get a hedge frame taken from the human template bank.
Healing and the glue gate
Healing. After all rewrites, a rule-based pass repairs leftovers: a and an before the next word's sound, agreement after swaps, doubled spaces and words, capitals after sentence ends, stray punctuation, broken ellipses and URLs. It runs only on text outside protected spans. The last act is restoring every placeholder to its original text.
Glue gate. Some rewrites must insert small function words such as and, which, who, this means or the. These are not content, but they are still words the system adds, so they are held to the provenance rule. A script (build_glue.py) counts each phrase in the human corpora and writes the ones seen at least 5 times to glue.tsv. At run time a rewrite may insert a glue word only if it appears in that file. If the file is missing, the gate fails closed and nothing is inserted.
Selection: the structural divergence gate
In aggressive mode a candidate is judged first by how different its structure is from the input, not by how many words changed. A pure synonym swap scores zero.
What is measured
• Tree edit distance (TED). Each sentence is parsed into a dependency tree whose nodes are labelled only with a grammatical relation and a part of speech (for example nsubj|NOUN), never with the word itself. Selkow's top-down tree edit distance counts the inserts, deletes and relabels needed to turn the input's trees into the output's. Moving a clause costs a delete plus an insert, so reordering shows up. Input and output sentences are first aligned by shared content-word stems, so a split, a merge or a reorder is compared against the right partner. The total is divided by the combined size of both trees, giving a number from 0 to 1.
• Order distance. How many aligned sentence pairs came out in a different order (Kendall).
• Burstiness. For the list of sentence lengths in words, B = (σ − μ) / (σ + μ). Near −1 is very regular; near 0 is random-like; human prose sits in a negative band that differs by register. AI text tends to be too even.
Targets are sampled, not fixed
• Each document draws a target TED from the measured distribution of structural distance between real human paraphrase pairs (MRPC, QQP), skewed toward the more-different half. Different documents aim at different levels.
• The floor is the human 20th percentile of that distribution (override: --min-ted).
• The burstiness floor is drawn between the register's 10th percentile and part-way to its median (override: --min-burst). It applies only to texts with six or more sentences.
Recipes make each candidate different
Every candidate gets a random recipe: a weight for each of 14 transform families (clause, voice, phrase, coord, list, inner, split, complex, para, sent_order, burst, merge, relcl, opening). Weights come from a log-normal draw, each family is switched off with probability 0.22, and at least four stay on. The weight multiplies the base probability of that family, so one candidate rewrites mostly clause order, another mostly splits and merges. The same transformation is not applied every time.
Hard gates and the choice
1. A candidate must have no placeholder leak, TED at or above the floor, and (for 6+ sentences) burstiness at or above its floor.
2. If only burstiness fails, the best candidates get a targeted repair: split the longest sentence or fuse the shortest neighbours until it passes.
3. Among passing candidates the winner is sampled with a softmax over a utility score that rewards TED up to the target, sentence reordering and total change from the original, and gives only a small penalty for statistical misfit (perplexity, TTR, and similar).
4. If nobody passes, the system escalates: it multiplies all transform probabilities by 1.4, then 1.8, with fresh seeds. The gate is never loosened. If it still fails, the candidate with the highest TED is returned and the report says passed_structure_gate: false.
Statistical scoring, seeds and hygiene
Statistical metrics. Each candidate is also measured on sentence-length histogram distance from the human mix, sentence-length variance, type-token ratio (vocabulary variety), stopword ratio, punctuation entropy and KenLM perplexity. Each has a human band (5th to 95th percentile) for the register. A value outside its band is a violation whose size is how far it is out.
In aggressive mode these are tie-breakers inside the utility score, not gates, so passed_all_gates: false in the report does not mean the output failed. In human mode they are the gates, and the system relaxes the strictest band by 50% up to two times, then draws one fresh batch of candidates. Texts with fewer than 8 sentences are judged on perplexity only, since per-document statistics are meaningless on tiny texts.
Change ratio. The share of words that differ from a reference text, from a word-level diff. In repeat passes it is measured against the original input, so a pass cannot count as progress by undoing earlier work.
Hygiene (on unless --no-hygiene).
• Salted seeds: when no --seed is given, each document's seed comes from a hash of its random ID, a persistent salt file, the clock and operating-system randomness, so two runs never share a seed.
• N-gram audit: a rolling store of 8-word spans from earlier outputs counts how many spans in a new candidate are new compared with the input but already appeared in another document's output. This is leakage; it catches template bank and dictionary phrases repeating across documents, and is penalised in selection.
• Idiom rotation: usage counts are saved so frequently used idiom swaps are used less often over time.
• Logging: every transform is written to a JSON log with the seed, so any output can be explained.
Repeat passes (-n)
-n N takes the output and paraphrases it again, N times in total. Each pass has its own seed. Left alone, later passes would undo earlier ones, because many rewrites are two-way (clause order, synonym swaps). Four guards prevent that:
• Seen forms. Every sentence form from earlier passes is recorded; a rewrite that would recreate one is rejected.
• Anchor to the original. From pass 2 on, TED and change are measured against the original input, not the previous pass, so a candidate that walks changes back scores lower.
• Retired words. Content words an earlier pass removed are barred from coming back through the lexical stage.
• Discard on regression. If a pass leaves the text closer to the original than the pass before, it is discarded, marked "discarded": true in the report, and the better text is kept.
The report lists per-pass change against the previous pass and against the original, and merges the transform counts of all passes.
Limit. Rewrites are local, so once each eligible sentence has been reshaped there is little new to do. After about two passes, extra passes usually change only a few percent more. More sentence-level rewrite types raise this ceiling more than more repeats do.
Commands, flags and the report
Run everything as python -m paraphraser <command> from the project folder.
Command
What it does
run in.txt -o out.txt
Paraphrase one file and print a JSON report
batch indir outdir
Paraphrase every .txt in a folder; models load once, so it is much faster than separate runs
calibrate
Re-measure the human style profiles from data/corpus (--perplexity also fits the KenLM band)
score in.txt
Print the statistical metrics of a file without changing it
Flag
Effect
--register
essay (default), blog, forum, technical or academic: picks the human profile
--candidates N
Versions generated per pass (default 4); more gives the gate more to choose from
-n N
Re-paraphrase the output N times; --save-passes also writes each pass as <output>.passN.txt
--seed S
Fixes the random draws; without it each run is salted and different
--structure aggressive / human
Aggressive (default) rearranges everything it safely can and uses the structural gate; human matches only measured human base rates
--lexical loose / strict
Loose (default) allows more synonym tiers and swaps
--lexical-rate R
Fraction of eligible content words to try swapping
--swap-every N
One idiom swap per N words (default 8)
--min-ted X, --min-burst X
Override the structural gate floors
--no-hygiene, --quiet
Skip the n-gram store and logging; hide the progress bar
Strongest setting: --lexical-rate 0.9 --swap-every 4 --candidates 12 -n 2 --save-passes. Raising --min-ted beyond about 0.25 will often report a failed gate, because current rewrites top out around there.
Reading the report. The most useful fields are:
• change_ratio: share of words that differ from the input (change_vs_original per pass in repeat mode).
• structure.ted, order, burstiness: the measured structural change, against target, floor and burst_floor.
• structure.passed_structure_gate: whether the hard gate was met; escalations shows how many stronger rounds were needed.
• structure.recipe: the weights this candidate used.
• transforms: a count of each rewrite that fired.
• leakage: spans repeated from other documents' outputs.
• passed_all_gates and violations: the older statistical bands; informational in aggressive mode.
Speed, progress and the build scripts
Speed. Parsing is the main cost, so the system avoids repeating it:
• Each text is parsed once and cached; a batch warm-up parses many sentences together.
• Protected spans are locked once on the original, and every candidate clones that base.
• Word-frequency lookups and the KenLM model load lazily and are cached.
• Short documents (under 8 sentences) skip the per-document statistics and are scored on perplexity only.
• The batch command loads the models once for a whole folder.
Progress display. A start-up report lists each resource with its load time. A single bar then shows percent, which pass out of N, the current stage, elapsed time and an estimate of time left. The bar only moves forward, even when extra candidates are added.
Build and calibration scripts. These run once, on your machine, and produce the files in data/. They are the only place any outside model is used, and only to classify or look up, never to write output text.
Script
Produces
download_data.py
Corpora, WordNet, GloVe, Moby thesaurus, MRPC, QQP and COCO pairs, KenLM files
build_phrase_pairs.py
phrase_pairs.tsv from human-versus-human pair sets
build_idioms.py
idioms.tsv: keeps only pairs whose WordNet senses and embeddings agree
build_templates.py
bank.jsonl: the human connective and template bank
build_glue.py
glue.tsv: closed-class words seen at least 5 times in human text
build_thesaurus.py
The cleaned Moby thesaurus used for tier-4 candidates
build_fragments.py, label_fragments_deepseek.py
The fragment sample used to calibrate fragment rate; labels only classify existing human sentences
gen_ai_pairs_deepseek.py
AI text for the same topics, used only as a lookup key when mining pairs, never as a source of output words
build_structure_profile.py
structure.json: human paraphrase TED quantiles and per-register burstiness bands
install_kenlm.py
KenLM for perplexity scoring
build_all.py
Runs the steps above in order and skips the ones already done
Known limits and next steps
Limits seen in testing
• Structural ceiling. On a 360-word essay, the output's tree distance reaches about 0.2 to 0.25. The human median for paraphrase pairs is about 0.45, so the output is still closer to its source than a typical human paraphrase. The default floor (human 20th percentile, near 0.2) is set to what current rewrites can reach.
• Low overall change. About 15 to 25% of words differ from the input in an aggressive run. Rewrites are local, so a second pass adds little.
• Source style carries through. If the input is itself AI-written, its rhythm (many "X was a Y" sentences) and stock phrases stay, because the system only reorders, splits, merges, swaps and deletes.
• Synonym quality. Filters block most bad swaps but not all (a word can still shift a nuance). Few candidates survive the filters in domain-heavy text, and the Moby thesaurus loaded only 3 roots in the last run, so the pool is mostly WordNet.
• Parser errors. A wrong parse can make a rewrite look safe when it is not. The guards catch most cases, but a read-through is still needed.
Most valuable next steps
1. More kinds of sentence-level rewrite (for example joining different-subject sentences with a human-bank connective such as while or whereas, or turning "It would also…" into a relative clause), because rewrites, not repeat passes, set the ceiling.
2. Check why the Moby thesaurus produced so few roots and rebuild it.
3. Re-run build_structure_profile.py as more human pairs are added, and raise the floor as the rewrites improve.
4. Add a human-readable diff view of what changed, per sentence, from the transform log.#   R u l e - b a s e d - p a r a p h r a s e r  
 