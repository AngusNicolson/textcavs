
import numpy as np
import torch
from tqdm import tqdm

from textcavs.feature_converter import FeatureConverter

device = "cuda"


def get_text_cavs(clip_model, feature_converter: FeatureConverter, concepts, prompts, batch_size=128):
    cavs = []
    # If more than one prompt, that is our batch. CAV is mean of batch.
    if len(prompts) > 1:
        for c in tqdm(concepts):
            with torch.no_grad():
                tokens = clip_model.tokenize([prompt.format(c) for prompt in prompts])
                c_vecs = clip_model.encode_text(tokens.to(device))
                c_vec = c_vecs.mean(0)
                cavs.append(c_vec)
        out = torch.stack(cavs)
    # Otherwise, batch the concepts and CAV is simply the encoded concept.
    else:
        batched_concepts = [concepts[i:i + batch_size] for i in range(0, len(concepts), batch_size)]
        for c in tqdm(batched_concepts):
            with torch.no_grad():
                tokens = clip_model.tokenize(c)
                c_vecs = clip_model.encode_text(tokens.to(device))
                cavs.append(c_vecs)
        out = torch.concat(cavs)

    with torch.no_grad():
        if feature_converter.variance_coefs["clip_text"] is not None:
            out = out * feature_converter.variance_coefs["clip_text"]
        out = feature_converter.to_model(out)
    out /= out.norm(dim=-1, keepdim=True)
    return out


def get_gradients(mymodel, target_class, bottleneck, examples):
    """Return the list of gradients.

    Args:
    mymodel: a model class instance
    target_class: one target class
    concept: one concept
    bottleneck: bottleneck layer name
    examples: an array of examples of the target class where examples[i]
      corresponds to class_acts[i]

    Returns:
    list of gradients
    """
    class_id = mymodel.label_to_id(target_class)
    grads = []
    for i in range(len(examples)):
        example = examples[i].to(device)
        grad = mymodel.get_gradient(example, class_id, bottleneck).cpu()
        grads.append(np.reshape(grad, -1))
    return grads
