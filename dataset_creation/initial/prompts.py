# %%
# Example Prompt:
# Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.  
# Type: fruit  
# List: [apple table banana orange chair dog]  
# Answer: (3)

import random
from dataset_creation.initial.wordNet import words_by_category
from dataset_creation.initial.conceptNet import conceptnet_words_by_category

def generate_prompt(words_by_category, list_length_range=(2, 20)):
    all_categories = list(words_by_category.keys())

    # pick a target category and the corresponding target words
    target_category = random.choice(all_categories)
    target_words = words_by_category[target_category]

    # choose a few correct words from that category
    correct_words = random.sample(target_words, k=random.randint(1, min(5, len(target_words))))

    # choose a few distractor words from other categores (all categories except the target one)
    distractor_categories = [cat for cat in all_categories if cat != target_category]
    distractors = [] # will store distractor words
    target_set = set(target_words)
    total_length = random.randint(*list_length_range)
    while len(correct_words) + len(distractors) < total_length:
        other_cat = random.choice(distractor_categories) 
        distractor_word = random.choice(words_by_category[other_cat])
        if distractor_word not in target_set: distractors.append(distractor_word)

    # put the correct words and distractors together in a list; shuffle the list
    word_list = correct_words + distractors
    random.shuffle(word_list) 

    # prompt
    prompt = (
        f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n"
        f"Type: {target_category.split('.')[0]}\n"
        f"List: [{', '.join(word_list)}]\n"
        f"Answer: ({len(correct_words)})"
    )

    return prompt

# %%
for _ in range(5):
    print(generate_prompt(words_by_category))
    print()