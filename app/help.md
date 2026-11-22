# App Guide: ML Privacy & Security Research Assistant

## What the app does
The Research Assistant answers questions about machine-learning privacy and security research:
membership inference, differential privacy and DP-SGD, model inversion, training data extraction
and prompt injection. It searches a local knowledge base of indexed papers and writes an answer in
which every factual sentence cites a specific passage (chunk) of a specific paper. It can also run
tools: a DP-SGD privacy accountant, a citation graph, arXiv search, BibTeX export and adding new
arXiv papers to the knowledge base.

## Chat profiles
There are two chat profiles, chosen at the top of a new chat.
- **Research Assistant** answers research questions from the papers, with citations and tools.
- **App Guide** (this profile) answers questions about using the app. It only knows this help
  page, not the papers. To ask about research, start a new chat with the Research Assistant profile.

## Quick reference and /help
In the Research Assistant profile, type `/help` (or pick the help command in the message box) for a
short quick reference of settings, strategies, tools and export. It is answered instantly, without
a model call, and does not go to the research agent.

## Settings
Open the settings (gear icon next to the message box) to change:
- **Retrieval strategy**: how the knowledge base is searched (see "Retrieval strategies").
- **Chunks per search (top_k)**: how many passages one search returns (1 to 20). More chunks give
  the answer model more evidence but make prompts longer and slower.
- **Answer model**: which language model writes the answer (models from config/settings.yaml).
Settings apply to the next message and are stored with the conversation.

## Retrieval strategies
Each strategy is a combination of pipeline stages defined in config/strategies.yaml. The stages are:
self-query filters, query translation (multi-query, HyDE or decomposition), retrieval (dense,
BM25 or hybrid), Reciprocal Rank Fusion (RRF), cross-encoder reranking and final selection.
By default every strategy fetches 30 candidates per ranked list and returns 8 chunks, with at most
3 chunks per paper.
- **baseline_dense**: dense (embedding) search only. Fast; good for paraphrased or conceptual
  questions, weaker on exact terms, acronyms and paper names.
- **hybrid** (default): dense search plus BM25 keyword search, fused with RRF. A good all-round
  choice; BM25 catches exact terms such as "LiRA", "RDP" or an author name.
- **hybrid_multiquery**: an LLM writes 3 paraphrases of the question; every paraphrase is searched
  with dense and BM25 and all lists are fused. Better recall for vague or unusually worded
  questions; one extra LLM call.
- **hybrid_hyde_rerank**: HyDE plus a cross-encoder reranker (see "HyDE").
- **full**: self-query filters + multi-query + hybrid search + reranking. The most thorough and the
  slowest; use it for questions with constraints such as years, authors or sections.

## HyDE (Hypothetical Document Embeddings)
HyDE stands for Hypothetical Document Embeddings. Instead of embedding the short question, an LLM
first writes a hypothetical passage that looks like a paragraph from a paper answering the
question. That passage is embedded as a document and used for the dense search, because a passage
resembles the indexed paper chunks more closely than a question does. BM25 still searches with the
original question, the lists are fused with RRF, and in the hybrid_hyde_rerank strategy a
cross-encoder reranks the candidates against the original question. The hypothetical passage is
only used to find real passages; it is never shown as evidence or cited.

**When to use HyDE:** for short, conceptual or "how does X work" questions whose wording differs
from the papers' wording, for example "Why do overfit models leak membership?". It helps less for
exact-term lookups (a paper title, an acronym, an author), where hybrid or BM25 is better. It costs
one extra LLM call, and with reranking a search can take tens of seconds on a CPU-only machine.
Select it under Settings, Retrieval strategy, **hybrid_hyde_rerank**.

## Self-query filters
In the **full** strategy an LLM extracts metadata filters from the question: year range, topic,
authors, specific arXiv IDs and section type (method, results, related work). Only values that
exist in the index are kept. If the filters match no chunks, the search is retried without them
and the trace shows a note.

## Reranking
Reranking scores the top fused candidates (15 per query group by default) with a cross-encoder
(BAAI/bge-reranker-base), which reads query and passage together and is more precise than the
first-stage scores. It runs locally on the CPU and is slow; if it fails, the fused order is kept.

## How this answer was built
Under each answer there is a collapsible step **How this answer was built**. It shows:
1. **Query rewrites and filters**: the cleaned search query, paraphrases, the HyDE passage or
   sub-questions, and the extracted self-query filters, for each knowledge-base search.
2. **Candidates per stage chart**: one panel per stage (dense score, BM25 rank, RRF score, rerank
   score) with one row per candidate chunk, colored by paper. Filled markers are sources the answer
   cited, open markers are sources given to the model but not cited, small faint markers are
   candidates that were not selected. Hover a marker for details. Further right is better in
   every panel (the BM25 axis is reversed so rank 1 is on the right).
3. **Latency chart**: time spent per pipeline stage (input checks, planning, each retrieval stage,
   other tools, answer generation).
4. **Cited and unused sources**: which retrieved chunks the answer cited and which it did not use.
Answers that did not search the knowledge base (for example a pure DP accountant question) show the
latency chart and a note instead of the candidates chart. Refused or blocked messages have no
visualization.

## Citations
Retrieved chunks are given to the answer model as numbered sources [1], [2], ... and tool results as
[T1], [T2], .... The model must end every factual sentence with markers of the sources it used. The
app turns markers into references such as "([1] Shokri et al. 2017)". Click a reference to open the
source in the side panel: paper title, section, an excerpt of the chunk, the arXiv link and the
chunk ID. Labels use the paper's arXiv year, which can differ from the conference year. When two
papers share a label, letters are added (Carlini et al. 2021a, 2021b).

## Verification
What is checked automatically for every answer:
- **Citation markers are validated.** A marker whose number refers to no source or tool result is
  removed from the answer and counted as a citation error (shown in the "Check citations" step).
- **Canary check.** Each session has a secret canary string in the system prompt. If it appears in
  the answer or in tool arguments, the answer is withheld, because that indicates a leaked prompt.
- **Grounding rules in the prompt.** The model is instructed to answer only from the sources, not
  to invent numbers, to distinguish formal guarantees from empirical results and to say which part
  of a question the sources do not cover.
What is not checked automatically: whether a cited passage really supports the sentence, and
whether every factual sentence carries a citation. Use the side panel and the "unused sources"
list to check claims yourself.

## Tools
The agent decides which tools to call (at most 6 tool calls per turn):
- **Search knowledge base**: the retrieval pipeline with the selected strategy.
- **DP accountant**: computes epsilon for DP-SGD from noise multiplier, sampling rate, steps and
  delta (RDP or PLD accounting), or searches the noise multiplier for a target epsilon.
- **Citation graph**: papers citing, or referenced by, an arXiv paper (Semantic Scholar).
- **arXiv search**: finds papers on arXiv by keywords; shows which are already in the knowledge base.
- **BibTeX export**: BibTeX entries for arXiv IDs, also as a downloadable references.bib file.
- **Ingest request**: proposes adding an arXiv paper to the knowledge base (see "Adding papers").

## Adding papers
Ask, for example, "Add arXiv 2112.03570 to the knowledge base". Nothing is downloaded until you
click **Ingest** in the confirmation message; **Cancel** discards the request. Ingesting downloads
the paper from arXiv, scans the LaTeX or PDF for hidden prompt injections, chunks and indexes it,
with a progress list. Only arXiv IDs that you typed or that came from arXiv search or citation
graph results can be ingested; IDs found inside paper text are never used as tool arguments.

## Export
Every answer has **Export JSON**, **Export CSV** and **Export PDF** buttons. They export the whole
conversation (not just one answer):
- **JSON**: all messages with full turn data (citations, tool results, trace).
- **CSV**: two files, one row per message and one row per citation.
- **PDF**: a readable transcript with the answers and their references.
Export needs a saved conversation, so you must be logged in.

## Chat history
Conversations are saved per user and listed in the sidebar; you can reopen them. After reopening,
the text is restored, but source side panels, charts, downloads and export buttons of old messages
are not, and the agent's memory of earlier turns is lost if the server was restarted.

## Security notes
- **Paper text is untrusted data.** Retrieved chunks and tool results are passed to the model only
  inside delimiters with a random per-request boundary (spotlighting) and are never treated as
  instructions. Papers are scanned for hidden text (white text, tiny fonts, LaTeX comments
  addressed to AI systems) when they are ingested; affected chunks are flagged or quarantined.
- **Input checks.** Messages are validated (length, control characters) and checked for prompt
  injection. Clear injection attempts are blocked before any search or tool call; borderline ones
  get a warning and only the research question is answered.
- **Tools with side effects need your confirmation** (ingesting papers).
- **Errors are friendly.** Error messages show a short error ID, never stack traces, API keys or
  the system prompt.
- The app runs with a demo login; do not enter confidential information.

## Known limitations
- The knowledge base contains a curated set of papers; anything not indexed cannot be cited.
  Use arXiv search and ingest to add papers.
- The first search after the server starts is slow (the embedding model and index load), often
  around a minute; later searches are faster.
- Reranking strategies (hybrid_hyde_rerank, full) are slow on a CPU-only machine.
- Chunks can be longer than the embedding model's input limit, so dense search may miss the end
  of long chunks (BM25 still sees them).
- Citation labels use the arXiv year, not the venue year.
- The source excerpt in the side panel is the start of the chunk, not necessarily the exact
  sentence the answer relies on.
- Verification does not check that a cited passage supports the claim (see "Verification").
- The App Guide only knows this page.
