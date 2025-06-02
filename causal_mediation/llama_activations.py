# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from huggingface_hub import login

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, model=True,)
model = AutoModelForCausalLM.from_pretrained(model_id, model=True, device_map="auto", torch_dtype=torch.float16)
model.eval()

with open("running_counts_dataset.json") as f:
    data = json.load(f)

sample = data[0] 
prompt = (
        "Given the following type of prompt, a sufficiently large language model will be able to answer with the correct number.\n"
        "Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {sample['Type']}\n"
        f"List: {sample['List']}\n"
        "Answer: ("
    )

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
activations = {}
def capture_hidden_states(module, input, output):
    activations["hidden_states"] = output

hook = model.model.layers[-1].register_forward_hook(capture_hidden_states)

with torch.no_grad():
    _ = model(**inputs)
hook.remove()

hidden_states = activations["hidden_states"]  # shape: [batch, seq_len, hidden_dim]
hidden_states = hidden_states[0]  # remove batch dim

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("\n Prompt tokenized as:")
for i, t in enumerate(tokens):
    print(f"{i:2}: {t}")

print("\n🔍 Hidden states for each word in list:")
for word_info in sample["AnnotatedList"]:
    token_pos = word_info["position"] + 2  # rough guess: skip header
    if token_pos < hidden_states.shape[0]:
        vector = hidden_states[token_pos]
        print(f"Word: {word_info['word']}, Count: {word_info['running_count']}, Vec Shape: {vector.shape}")