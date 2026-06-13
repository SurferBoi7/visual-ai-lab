---
name: Interleaved continual learning
description: How the text worker avoids Catastrophic Forgetting when multiple datasets are added.
---

## The rule
`commitToEngine` must send **all active datasets** to the worker as both a combined corpus AND as the `corpusDatasets` array. Sending only the new corpus resets the model's knowledge of prior datasets.

## Worker behavior
- `InitOpts.corpusDatasets?: string[]` — optional per-dataset array
- Vocab is always built from the **combined** corpus (all datasets joined) so token ids are consistent across datasets
- `DatasetWindowSet` holds per-dataset windows + their own shuffle cursor
- When `datasetWindowSets.length > 1`, `trainEpoch` does round-robin: for each step, one example is drawn from each dataset in rotation → every gradient update sees all knowledge sources

**Why:** Sequential training on a new dataset causes the network to overwrite weights that encode prior knowledge. Round-robin interleaving keeps gradient signals from all datasets mixed in every epoch.

**How to apply:**
- When adding any new dataset, recompute `updatedDatasets`, filter `.active`, map to `.text`, join as `combinedCorpus`, and pass `corpusDatasets` alongside `corpus` in the worker reset message.
- Do NOT call `setLLMConfig(c => ({ ...c, corpus: ... }))` in `commitToEngine` — the `LLMArchitect` useEffect already rebuilds corpus from datasets; double-updating causes a race.
