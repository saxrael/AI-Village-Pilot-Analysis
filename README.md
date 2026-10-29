# ai-village-pilot

Pilot analysis of the [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village) (AI Digest), run before applying to the AI Village Data Exploration Grant as a Data Scout.

## Goal

Show that the dataset can be worked with end to end: read the large gated tables, link agent claims in chat to the same agent's computer-use logs, and report what the links show, with each check validated before it is trusted.

## Deliverables

| Deliverable | Where |
|---|---|
| Analysis code, one module per check | `pilot/` |
| Unit and end-to-end tests on synthetic data | `tests/` |
| Colab notebooks that run each check | `notebooks/` |
| Aggregate result tables and figure | `results/` |
| Results summary with validation status | this README |

The Data Scout deliverable (a one-page report and code) builds on these modules.

## Results

Data: 45 agents, April 2025 to October 2026. Counts are from the full chat and computer-use tables.

| Check | Module | Result | Validation | Status |
|---|---|---|---|---|
| Identifier audit | `identifier_census`, `identifier_provenance` | 33,559 quoted identifiers; 98.2% of 21,055 first mentions appear in the quoting agent's own tool output before the quote (20,676 own, 184 another agent's, 33 only after, 162 not found) | Unit-tested; fixture run end to end | Complete |
| Push check | `push_check` | 1,903 of 2,184 "pushed" claims (87.1%) match a successful push of that commit in the agent's own logs; 162 show another successful push, 18 only a failure, 101 no push output | Unit-tested; labels not yet hand-checked | Complete |
| Screenshot check | `screenshot_check` | Identifier found on screen for 5 of 24 sampled untraced quotes (4 exact, 1 fuzzy); 1 uncheckable | Sample size gives a 95% interval of 2.2% to 11.2% on the hit rate (Wilson) | Feasibility only |
| Origin check | `origin_check` | The term "cold verification" appears in no goal or human message before 11 September 2026, when one agent introduced it and invited two others | First twelve uses read against the chat | Complete |
| Term diffusion | `term_diffusion` | Adoption order, speed and model family of each term | Unit-tested; tables regenerate from the notebook | Re-run pending |
| Claim-to-log matching | `claim_evidence` | Labels each cold-verification report by whether the agent's own logs show the quoted check count, hash or script | Hand-check of 20 labels: 4 right | Failed validation; numbers withheld |
| Cross-agent verification census | none | Counted who verified whom from chat text alone | Hand-check of 20: 1 to 2 right | Failed validation; dropped |

## Checks that failed validation

Two checks did not pass the hand-check and were dropped from the results above.

| Check | Hand-check | What changed |
|---|---|---|
| Cross-agent verification census (chat only) | 1 to 2 of 20 labels right | Dropped. Verification is now read from the agent's own logs instead of chat wording |
| Claim-to-log matching, first labelling rule | 4 of 20 labels right | Numbers withheld. Module kept in `pilot/claim_evidence.py`; the rule is being rewritten to score only first-person reports and to require a script run in the same turn |

## Run

```
pip install -r requirements.txt
export HF_TOKEN=...          # needs access to aidigestorg/ai-village
python pilot/identifier_census.py --out-dir out
python pilot/identifier_provenance.py --claims out/identifier_claims.jsonl --out-dir out
python pilot/push_check.py --provenance out/identifier_provenance.jsonl --out-dir out
python pilot/screenshot_check.py --windows out/untraced_windows.jsonl --rematch out/rematch_untraced.csv --out-dir out
python pilot/term_diffusion.py --out-dir out
python pilot/origin_check.py --out-dir out
python pilot/claim_evidence.py --out-dir out
pytest
```

Each script takes `--local-dir` to read a downloaded copy of the tables instead of streaming from the Hub. `notebooks/` runs the same commands in Colab with `HF_TOKEN` stored as a secret.

## Repository

```
pilot/        analysis modules, shared helpers in common.py and stats.py
tests/        unit tests and end-to-end runs on a synthetic dataset
notebooks/    01_identifier_audit, 02_term_diffusion, 03_claim_evidence, 04_origin_check
results/      aggregate tables only
```

Message text, per-claim files and screenshots are not committed; the dataset is gated, and its terms restrict redistribution.

## Data and citation

AI Digest, AI Village dataset, `aidigestorg/ai-village`, used under its terms (research use only, no re-identification). Please cite AI Digest and the AI Village project when using results from this repository.
