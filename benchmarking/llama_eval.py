# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch, json, re
import requests

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, token=True, device_map="cpu")

with open("/dataset_creation/matching_count_dataset.json") as f:
    data = json.load(f)

correct = 0
output_log = []
total = 10 

for ex in data[:total]:
    prompt = (
        "Given the following type of prompt, a sufficiently large language model will be able to answer with the correct number.\n"
        "Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {ex['Type']}\n"
        f"List: {ex['List']}\n"
        "Answer: ("
    )
    expected = f"{ex['Answer']})" if not str(ex['Answer']).startswith("(") else str(ex['Answer'])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    match = re.search(r"\(?(\d+)", generated_text)
    predicted = f"{match.group(1)})" if match else "N/A"

    print("\n" + "="*60)
    print("PROMPT:", prompt)
    print("\n RAW MODEL OUTPUT:", generated_text)
    print("\n EXTRACTED ANSWER:", predicted)
    print("EXPECTED ANSWER:", expected)

    if predicted.strip() == expected.strip():
        correct += 1

    output_log.append({
    "Type": ex["Type"],
    "List": ex["List"],
    "Expected": expected,
    "RawOutput": generated_text,
    "Predicted": predicted,
    "Match": predicted.strip() == expected.strip()
    })

with open("llama3_outputs2.json", "w") as out_file:
    json.dump(output_log, out_file, indent=2, ensure_ascii=False)

print(f"\nAccuracy: {correct}/{len(data[:total])} = {correct / len(data[:total]):.2%}")