
This project investigates whether large language models can count how many words in a list match a given semantic category, including three main steps:

- Dataset Creation
- Benchmarking
- Causal Mediation Analysis (In Progress)

---

## Dataset Creation
The dataset consists of 5,000 prompts that require a model to count how many items in a list match a specified semantic category.
- WordNet and ConceptNet were initially considered, but were excluded due to sparsity, noise, and overlapping category definitions.
- Final word lists were manually constructed to ensure clarity and dataset consistency.

### Categories Used 
- fruit
- animal
- sport
- vehicle
- color
- instrument
- occupation
- furniture
- plant

### Prompt Format Example

Type: fruit
List: [dog, apple, cherry, bus, cat, grape, bowl]
Answer: (3)

---

## Benchmarking

Three open-weight language models were tested on a sample of the first 100 prompts from the matching_count_dataset.json dataset in a zero-shot setting.

### Results

| Model                  | Accuracy | Notes                                      |
|-----------------------|----------|--------------------------------------------|
| LLaMA 3.1 8B Instruct | 18%      | Best performance overall                   |
| GPT-J                 | 0%       | Failed to extract or infer numeric answer  |
| GPT-2                 | 0%       | Failed due to model size and lack of tuning|

### Evaluation Notes
- Same prompt structure used for all models.
- No reasoning tokens or few-shot examples were added.
- Evaluation compared model outputs directly to the numeric ground truth.

---

## Causal Mediation Analysis (In Progress)

### Objective
To determine whether intermediate hidden states in transformer models encode a running count of matching category words as the list is processed sequentially.

### Current Status
Initial activation extraction began:
- Captured residual stream activations at every GPT-2 layer (resid_mid) using forward hooks during inference on structured counting prompts.
- Mapped activations to token positions corresponding to list words using approximate offsets, based on annotated word positions from the dataset.
- Extracted and printed hidden states for each word token along with its running count label, laying the groundwork for future probing or causal intervention analysis.

*(Analysis could not be completed due to limited compute and the time duration of the task.)*

---

### Planned Methodology: Next Steps

**Train a Linear Probe for Running Count**
Detect whether intermediate representations in GPT-2 contain an explicit encoding of the running count of category-matching words.

**Causal Mediation via Activation Patching**
Test whether specific hidden states cause the final token prediction to reflect the running count.

Procedure:
1. Prompt Construction:
  - Clean Prompt: Contains multiple category-matching items (e.g., 3 fruits).
  - Corrupted Prompt: Replaces some matching items with distractors (e.g., 1 fruit).
2. Run Forward Passes: Collect clean and corrupted activations using the HookedTransformer cache.
3. Patching Operation: Replace specific hidden states in the corrupted run with those from the clean run. Then run forward from that point using run_with_cache(patched_activations) or similar patching utilities in HookedTransformer.
4. Output Comparison
5. Repeat the patching over: layers, positions (focus on early/mid/late list tokens), and multiple examples.

**Analyze Mediation Results**
Identify locations (layer × position) where hidden states mediate the count prediction.










