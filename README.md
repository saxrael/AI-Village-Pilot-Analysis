# ai-village-pilot

A pilot analysis of the [AI Village dataset](https://huggingface.co/datasets/aidigestorg/ai-village) from AI Digest. I ran it before applying to the AI Village Data Exploration Grant as a Data Scout, to show that I can work with this dataset and to find the questions worth exploring.

## What this covers

AI agents in the Village post claims in chat, like "pushed commit f11a175" or "cold verification passed, 210/210 checks". The dataset also holds what the same agents did on their computers. The pilot links the two and asks whether what agents say matches what their own logs show. Every check was tested before I trusted its results.

## What's in the repo

- `pilot/` the analysis code, one script per check
- `tests/` tests that run every script on a small made-up dataset
- `notebooks/` Colab notebooks that run each check
- `results/` summary tables only

## Results

The data covers 45 agents from April 2025 to October 2026.

| Check | What I found | How I tested it | Status |
|---|---|---|---|
| Quoted identifiers | Agents quoted 33,559 commit hashes and similar identifiers. For 98.2% of first mentions (20,676 of 21,055), the agent's own tool output showed the identifier before it quoted it. | Code tests on made-up data | Done |
| Push claims | Of 2,184 "pushed" claims, 1,903 (87.1%) matched a real push of that commit in the agent's own logs. 162 showed a different successful push, 18 only a failed push, and 101 no push at all. | Code tests on made-up data; results not yet read by hand | Done |
| Screenshots | For quotes the logs could not explain, I searched screenshots on a sample of 24. I found the identifier on screen in 5 (4 exact, 1 close match) and could not check 1. | The sample is small, so the true rate could be anywhere from 2.2% to 11.2% | Feasibility test |
| Where "cold verification" came from | No goal or human message used the term first. One agent introduced it on 11 September 2026 and invited two others to use it. | Read the first twelve uses in the chat | Done |
| How terms spread | Which agents and model families picked up each term, and how fast. | Code tests on made-up data | Code ready, tables to be re-run |
| Verification claims against logs | Checks whether the agent's own logs show the check count, hash or script it quoted. | Read 20 results by hand: 4 were right | Dropped, see below |
| Who verifies whom | Counted agents verifying each other's work across model families, from chat text alone. | Read 20 results by hand: 1 to 2 were right | Dropped, see below |

## Directions I dropped

I started with two hypotheses:

1. **Do agents verify each other's work, and does it cross model families?** I counted this from chat text alone. When I read 20 of the results by hand, only 1 or 2 were right, because chat wording does not show who actually checked what. I dropped it.
2. **Do the verification claims agents post match what their own logs show?** My first matching rule got 4 of 20 right when I read them by hand, so I dropped those numbers.

Both changed the direction and the hypothesis to:

**When AI agents verify each other's work, the verification holds up:** the checks are real, results stay faithful when passed on, the receipts agents quote can be traced, and agents check their own model family's work the same way they check other families'.

## How to run

```
pip install -r requirements.txt
export HF_TOKEN=...          # token with access to aidigestorg/ai-village
python pilot/identifier_census.py --out-dir out
python pilot/identifier_provenance.py --claims out/identifier_claims.jsonl --out-dir out
python pilot/push_check.py --provenance out/identifier_provenance.jsonl --out-dir out
python pilot/screenshot_check.py --windows out/untraced_windows.jsonl --rematch out/rematch_untraced.csv --out-dir out
python pilot/term_diffusion.py --out-dir out
python pilot/origin_check.py --out-dir out
python pilot/claim_evidence.py --out-dir out
pytest
```

Add `--local-dir` to any script to read a downloaded copy of the tables instead of streaming them. The notebooks do the same in Colab, with `HF_TOKEN` saved as a secret.

## Data

AI Digest, AI Village dataset (`aidigestorg/ai-village`), used under its terms: research use only, no re-identification. The dataset is gated and cannot be redistributed, so this repo holds code and summary tables only, with no message text or screenshots. If you use results from here, please cite AI Digest and the AI Village project.

## Authorship

I wrote this code with Claude, and directed and ran the analysis myself. Israel Ayeni.
