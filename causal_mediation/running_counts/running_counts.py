# %%
from dataset_creation.dataset import category_wordbank
import json
import os

input_file = "matching_count_dataset.json"
output_file = "running_counts_dataset.json"

def annotate_entry(entry):
    word_list = entry["List"]
    target_type = entry["Type"].lower()
    match_set = category_wordbank.get(target_type, set())

    running_count = 0
    annotated_list = []

    for idx, word in enumerate(word_list):
        is_match = word.lower() in match_set
        if is_match:
            running_count += 1
        annotated_list.append({
            "word": word,
            "position": idx + 1,
            "is_match": is_match,
            "running_count": running_count
        })

    return {
        "Type": entry["Type"],
        "List": word_list,
        "AnnotatedList": annotated_list,
        "Answer": running_count
    }

def main():
    with open(input_file, "r") as f:
        dataset = json.load(f)

    annotated_dataset = [annotate_entry(entry) for entry in dataset]
    print("Saving to:", os.path.abspath(output_file))
    with open(output_file, "w") as f:
        json.dump(annotated_dataset, f, indent=2)

    print(f"Annotated dataset saved to: {output_file}")

if __name__ == "__main__":
    main()