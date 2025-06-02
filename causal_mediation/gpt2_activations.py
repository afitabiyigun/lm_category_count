# %%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import json

# Load model
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")

# Load annotated dataset
with open("running_counts_dataset.json") as f:
    data = json.load(f)

sample = data[:10]
prompt = (
    "Count the number of words in the following list that match the given type, "
    "and put the numerical answer in parentheses.\n"
    f"Type: {sample['Type']}\nList: [{', '.join(sample['List'])}]\nAnswer: ("
)

activations = {}
def save_activations_hook(act, hook):
    activations[hook.name] = act.detach()

model.run_with_hooks(
    prompt,
    fwd_hooks=[(get_act_name("resid_mid", layer), save_activations_hook) for layer in range(model.cfg.n_layers)],
)

# OPTIONAL: map list-word token positions if needed (GPT-2 uses BPEs)
tokenized = model.to_str_tokens(prompt)
list_words = sample["AnnotatedList"]

# Now collect activations at each word's approximate token position
# You may need better alignment (e.g., matching subword tokens back to words)
for layer in range(model.cfg.n_layers):
    resid = activations[get_act_name("resid_mid", layer)][0]  # [seq_len, d_model]

    print(f"\nLayer {layer} Residual Stream Activations:")
    for word_info in list_words:
        idx = word_info["position"] + 2  # crude offset: header + newline
        if idx < len(resid):
            hidden_vector = resid[idx]  # [d_model]
            print(f"Word: {word_info['word']}, Count: {word_info['running_count']}, Activation Dim: {hidden_vector.shape}")