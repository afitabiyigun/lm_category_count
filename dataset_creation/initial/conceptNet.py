# %%

import requests
import random
from wordfreq import word_frequency
from dataset_creation.initial.config import MIN_FREQ, DISALLOWED_TERMS

def get_conceptnet_examples(category, limit=1000, min_freq=MIN_FREQ):
    """
    Fetch common words that are IsA <category> from ConceptNet.
    Cleans: one-word only, alphabetical, frequent words.
    """
    url = f"http://api.conceptnet.io/query?rel=/r/IsA&end=/c/en/{category}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    words = set()
    for edge in data.get('edges', []):
        start_term = edge['start']['term']  # e.g., /c/en/apple
        if start_term.startswith('/c/en/'):
            word = start_term.split('/')[3].replace('_', ' ').lower()
            if (
                word.isalpha() and
                ' ' not in word and
                word_frequency(word, 'en') > min_freq
            ):
                words.add(word)
    return sorted(words)

# %%
categories = ["fruit", "animal", "vehicle", "color", "instrument", "occupation", "furniture"]
conceptnet_words_by_category = {}

for cat in categories:
    print("")
    conceptnet_words_by_category[cat] = get_conceptnet_examples(cat)
    print(f"{cat}: {len(conceptnet_words_by_category[cat])} words")
    # for word in conceptnet_words_by_category[cat][:]:
    #     print(word, end=', ')
# %%
