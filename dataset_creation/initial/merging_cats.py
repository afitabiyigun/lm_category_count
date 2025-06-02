# %%
from collections import defaultdict
from itertools import combinations

def merge_similar_categories(synsets, threshold=0.85):
    """
    Groups semantically close synsets using Wu-Palmer similarity.
    Returns a dict mapping each synset name to a merged cluster label.
    """
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Pairwise similarity and clustering
    for s1, s2 in combinations(synsets, 2):
        sim = s1.wup_similarity(s2)
        if sim and sim >= threshold:
            union(s1, s2)

    # Group synsets under root
    clusters = defaultdict(list)
    for s in synsets:
        clusters[find(s)].append(s)

    # Choose label for each cluster
    label_map = {}
    for root, members in clusters.items():
        label = root.lemmas()[0].name()  # choose label from root synset
        for s in members:
            label_map[s.name()] = label

    return label_map