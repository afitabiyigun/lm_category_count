# Defining categories of words and extracting them from WordNet
# %%
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from dataset_creation.initial.config import MIN_FREQ, DISALLOWED_TERMS

# %%
for synset in list(wn.all_synsets('n'))[:]:
    print(synset)
# %% extracting category synsets from WordNet, instead of hardcoding them one by one
categories = ["fruit", "animal", "vehicle", "color", "instrument", "occupation", "furniture", "sport", "drink", "plant", "clothing", "body_part", "emotion"]

# def get_top_level_noun_categories(depth=3, min_hyponyms=10):
#     categories = []
#     noun_roots = wn.synset('entity.n.01').hyponyms() # Start from all top-level noun synsets
#     for root in noun_roots:
#         stack = [(root, 1)]
#         while stack:
#             synset, current_depth = stack.pop()
#             if current_depth == depth:
#                 hyponyms = list(synset.closure(lambda s: s.hyponyms()))
#                 if len(hyponyms) >= min_hyponyms:
#                     categories.append(synset)
#             elif current_depth < depth:
#                 stack.extend([(child, current_depth + 1) for child in synset.hyponyms()])
    
#     return categories

# %% defining list of words for each category
import re
from wordfreq import word_frequency

category_keywords = {}

# filter out rare words and clean the wordnet lists
def is_clean_word(word, min_freq=MIN_FREQ, category_keywords=None):
    word = word.lower()
    if word_frequency(word, 'en') <= min_freq: return False
    # if len(word.split()) > 2 or len(word) > 20: return False
    # if not re.match(r'^[a-zA-Z\s\-]+$', word): return False
    # if any(kw.lower() in word for kw in category_keywords): return False
    if any(term in word for term in DISALLOWED_TERMS): return False
    return True

def get_words_for_synset(synset_name, category_keywords=[]):
    synset = wn.synset(synset_name) # fetch the actual synset object from WordNet using the string name (e.g. wn.synset('fruit.n.01')) gives the fruit noun synset)
    # print(synset.definition()) 
    words = []
    # want to get not all hyponyms actually; just leaf nodes that do not have children that can be further classified 
    # synsets that have no further hyponyms (e.g. "health profession", still a category of professions, but not a leaf node/instance like "nurse") 
    for syn in synset.closure(lambda s: s.hyponyms() + s.instance_hyponyms()): # traversing the hierarchy of WordNet concepts, from each synset downward through all hyponyms, to include all subtypes of 
        if syn.hyponyms() or syn.instance_hyponyms(): continue # only leaf nodes, no further hyponym
        if syn.lexname() in ['noun.cognition', 'noun.attribute', 'noun.process']: continue
        if syn.lexname() in ['noun.group', 'noun.cognition', 'noun.attribute', 'noun.communication']:continue        
        if any(term in syn.definition().lower() for term in ["process", "state", "organization", "activity", "service"]): continue
        for name in syn.lemma_names(): # get the names of the lemmas in the synset
            word = name.replace('_', ' ').lower() # multiwords
            if is_clean_word(word, min_freq=MIN_FREQ, category_keywords=category_keywords):
                words.append(word)
    return words
# %%
# automating the process of fetching words for each category
# categories = get_top_level_noun_categories()

words_by_category = {}
for cat in categories:
    # words_by_category[cat.name()] = get_words_for_synset(cat.name())
    synset_name = f"{cat}.n.01"
    inferred_keywords = [cat.lower()] 
    words = get_words_for_synset(synset_name, category_keywords=inferred_keywords)
    if words: words_by_category[cat] = words # only add categories with non-empty lists of words
# %%
# filtered_words_by_category = {}

# for cat, words in words_by_category.items():
#     if len(words) >= 30:
#         sorted_words = sorted(words, key=lambda w: word_frequency(w, 'en'), reverse=True)
#         filtered_words_by_category[cat] = sorted_words[:30]
    
# for cat in filtered_words_by_category:
#     print(f"Category: {cat} — {len(filtered_words_by_category[cat])} words")
#     for word in filtered_words_by_category[cat][:10]:
#         print(f"  {word}")
# %%
for cat in words_by_category:
    print(f"Category: {cat} — {len(words_by_category[cat])} words")
    for word in words_by_category[cat][:10]:
        print(f"  {word}")