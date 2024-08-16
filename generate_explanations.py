
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import clip
import matplotlib.pyplot as plt
import seaborn as sns

from textcavs.feature_converter import FeatureConverter
from textcavs.utils import MIMIC_CLASS_NAMES, TARGET_CLASS_NAMES, CLASS_NAMES_PATH
from textcavs.model import get_model, get_clip_model
from textcavs.explanations import get_text_cavs, get_gradients

device = "cuda"

torch.set_float32_matmul_precision("high")


def main(args):
    layer = args.layer
    model_path = args.model
    exp_name = args.exp_name
    dataset = args.dataset.lower()
    center_crop = not args.no_center_crop
    clip_model = args.clip_model
    concept_names = args.concept_names
    concept_dir = Path(args.concept_dir)
    concept_path = concept_dir / f"{concept_names}.txt"
    out_suffix = args.out_suffix

    # What are the class names the model expects?
    # And which of these classes do you want to generate explanations for?
    if dataset == "imagenet":
        with open(CLASS_NAMES_PATH, "r") as fp:
            class_names = fp.read().split("\n")

        if args.target_classes is None:
            target_classes = TARGET_CLASS_NAMES
        else:
            target_classes = [v.strip() for v in args.target_classes.split(",")]
    elif dataset == "mimic-cxr":
        if args.class_names is None:
            class_names = MIMIC_CLASS_NAMES
        else:
            class_names = [v.strip() for v in args.class_names.split(",")]

        if args.target_classes is None:
            target_classes = class_names
        else:
            target_classes = [v.strip() for v in args.target_classes.split(",")]
    else:
        raise ValueError(f"Dataset {dataset} not recognised!")

    base_outdir = Path(args.out_dir)
    feature_converter_dir = base_outdir / exp_name

    outdir = feature_converter_dir / "outputs"
    outdir.mkdir(exist_ok=True)

    with open(concept_path, "r") as fp:
        concepts = fp.read().split("\n")

    feature_converter = FeatureConverter()
    feature_converter.load_model(feature_converter_dir)
    model, preprocess = get_model([layer], model_path=model_path, center_crop=center_crop, class_names=class_names)
    model.model.to("cuda")
    clip_model = get_clip_model(clip_model)
    clip_model.to("cuda")

    if dataset == "imagenet":
        prompts = [
            'a photo of {}.',
            "an image of {}.",
            "art depicting {}.",
            "{}",
            "an example {}.",
            "something similar to {}.",
        ]
    else:
        prompts = ["{}"]

    cavs = get_text_cavs(clip_model, feature_converter, concepts, prompts)
    cavs = cavs.cpu().numpy()

    ordered_concepts = {}
    for target in target_classes:
        print(f"Getting explanations for {target}.")

        ordered_concepts[target], directional_dirs, gradient_cosines = get_concept_sensitivity_order(
            target,
            model,
            layer,
            cavs,
            concepts
        )
        # True value of class name not needed anymore so replace spaces with _ for plots
        target = target.replace(" ", "_")
        ordered_concepts[f"{target}_vals"] = directional_dirs
        ordered_concepts[f"{target}_grad_cosine"] = gradient_cosines

        if args.plot_dists:
            fig = sns.displot(directional_dirs)
            fig.axes[0, 0].set_title(target)
            fig.axes[0, 0].set_xlabel("Directional Derivative")
            plt.savefig(outdir / f"directional_derivative_distribution_{target}.png", bbox_inches="tight")

            fig = sns.displot(gradient_cosines)
            fig.axes[0, 0].set_title(target)
            fig.axes[0, 0].set_xlabel("Gradient Cosine")
            plt.savefig(outdir / f"gradient_cosine_distribution_{target}.png", bbox_inches="tight")

    cav_df = pd.DataFrame(ordered_concepts)
    out_path = outdir / f"ordered_by_directional_derivative_concepts_for_each_class_{concept_names}{out_suffix}.csv"
    cav_df.to_csv(
        out_path
    )
    print(f"Explanations saved to {out_path}")
    print("Done!")


def get_concept_sensitivity_order(target, model, layer, cavs, concepts):
    img = torch.zeros((1, 3, 224, 224)).to(device)

    gradients = get_gradients(model, target, layer, img)[0]
    gradients = gradients.cpu().numpy()
    normalised_gradients = gradients / np.linalg.norm(gradients)

    gradient_cosines = []
    directional_dirs = []
    for cav in cavs:
        directional_dirs.append((gradients * cav).sum())
        gradient_cosines.append((normalised_gradients * cav).sum())
    directional_dirs = np.array(directional_dirs)
    gradient_cosines = np.array(gradient_cosines)
    sensitivity_order = np.argsort(directional_dirs)[::-1]
    concepts_in_order = [concepts[i] for i in sensitivity_order]
    return concepts_in_order, directional_dirs[sensitivity_order], gradient_cosines[sensitivity_order]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .pth, if None uses pretrained ImageNet ResNet50"
    )
    parser.add_argument(
        "--exp-name",
        default="example_00",
        help="Experiment name (for savefiles)"
    )
    parser.add_argument(
        "--concept-names",
        default="tulu_4bit_00_cleaned",
        help="The name of the concepts file"
    )
    parser.add_argument(
        "--concept-dir",
        default="data/text_concepts",
        help="Directory containing the concept.txt files"
    )
    parser.add_argument(
        "--clip-model",
        default='ViT-B/16',
        help="CLIP model to use"
    )
    parser.add_argument(
        "--layer",
        default="avgpool",
        help="Layer to extract model features/gradients"
    )
    parser.add_argument(
        "--out-dir",
        default="models/feature_converters",
        help="Output directory"
    )
    parser.add_argument(
        "--plot-dists",
        action="store_true",
        help="Plot directional derivative distributions for each class"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        help="Name of dataset"
    )
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Whether to center crop the images or not"
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optionally define the class names for the model if non-standard "
             "(comma separated)"
    )
    parser.add_argument(
        "--target-classes",
        default=None,
        help="Optionally provide a subset of class names to generate explanations for "
             "(comma separated)"
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Optionally add a suffix to the output .csv. "
             "Usually used in combination with target-classes."
    )
    main(parser.parse_args())
