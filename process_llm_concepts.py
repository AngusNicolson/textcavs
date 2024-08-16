
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import clip

from textcavs.model import ClipZeroShot

device = "cuda"


def main():
    base_path = Path("data/text_concepts")
    json_paths = [
        base_path / "tulu_4bit_seen_around_00.json",
        base_path / "tulu_4bit_part_of_00.json",
        base_path / "tulu_4bit_superclass_00.json",
    ]
    out_name = "tulu_4bit_00.txt"
    out_path = base_path / out_name

    jsons = []
    for json_path in json_paths:
        with open(json_path, "r") as fp:
            jsons.append(json.load(fp))

    concepts = set()
    for responses in jsons:
        concepts = concepts.union(extract_concepts(responses))

    concepts = list(concepts)
    concepts.sort()

    with open(out_path, "w") as fp:
        fp.write("\n".join(concepts))

    cleaned_concepts = conceptnet_cleaning(concepts)
    cleaned_concepts = remove_similar(cleaned_concepts, threshold=0.9)

    with open(out_path.parent / (out_path.stem + "_cleaned.txt"), "w") as fp:
        fp.write("\n".join(cleaned_concepts))

    print(len(concepts), len(cleaned_concepts))
    print("Done!")


def remove_similar(concepts, threshold=0.9):
    clip_model = ClipZeroShot('ViT-B/16')
    embeddings = get_all_embeddings(clip_model, concepts)

    out = []
    for i, concept in tqdm(enumerate(concepts)):
        similarities = get_similarities(embeddings[i], embeddings)
        highly_similar = list(np.where(similarities > threshold)[0])
        highly_similar_concepts = [concepts[j] for j in highly_similar]
        word_lengths = [len(v) for v in highly_similar_concepts]
        shortest_word = highly_similar_concepts[np.argmin(word_lengths)]
        n_words = len(concept.split(" "))
        if n_words == 1:
            out.append(concept)
        else:
            out.append(shortest_word)

    out = list(set(out))

    return out


def conceptnet_cleaning(all_concepts):
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [v[1:] if v[:2] == "a " else v for v in all_concepts]
    all_concepts = [v[2:] if v[:3] == "an " else v for v in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c != ""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    cleaned_concepts = clean_concepts(all_concepts)
    return cleaned_concepts


def extract_concepts(responses):
    responses = {k: process_indiv_response(v) for k,v in responses.items()}
    all_concepts = set(in_v for v in responses.values() for in_v in v)
    return all_concepts


def process_indiv_response(response):
    out = response.split("<|assistant|>")[-1]
    out = [v.strip() for v in out.split("\n")]
    out = [v[2:] for v in out if v.startswith("- ")]
    if len(out) > 0:
        if out[-1].endswith("</s>"):
            out[-1] = out[-1][:-4]
    return out


def clean_concepts(scenario_concepts):
    """
    Clean the plurals, trailing whitespaces etc.
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    # We use nltk to handle plurals, multiples of the same words etc.
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        c = c.strip()
        c_subwords = c.split(" ")
        # If a concept is made of more than 2 words, we drop it.
        if len(c_subwords) > 2:
            print("skipping long concept", c_prev)
            continue
        # Lemmatize words to help eliminate non-unique concepts etc.
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))
    return scenario_concepts_rec


def get_similarities(target_embedding, embeddings):
    similarities = (target_embedding * embeddings).sum(axis=1)
    return similarities


def get_all_embeddings(clip_model, prompts, to_numpy=True, **kwargs):
    embeddings = []
    for prompt in tqdm(prompts, desc="Getting embeddings"):
        embedding = get_embedding(clip_model, prompt, **kwargs)
        embeddings.append(embedding)
    embeddings = torch.concat(embeddings)
    if to_numpy:
        embeddings = embeddings.cpu().numpy()
    return embeddings


def get_embedding(clip_model, prompt, normalise=True):
    with torch.no_grad():
        tokens = clip.tokenize(prompt)
        embedding = clip_model.encode_text(tokens.to(device))
        if normalise:
            embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding


if __name__ == "__main__":
    main()
