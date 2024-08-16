
from pathlib import Path
from argparse import ArgumentParser
import yaml

import numpy as np
import torch
import clip
import seaborn as sns
import matplotlib.pyplot as plt

from textcavs.feature_converter import FeatureConverter
from textcavs.model import get_model, get_clip_model
from textcavs.utils import get_imagenet_dataset, CLIP_IMAGENET_TRANSFORMATION, RESIZE_TRANSFORMATION
from textcavs.datasets import get_mimic_dataset

from train_feature_converter import obtain_ftrs, get_text_embeddings

device = 'cuda'


def main(args):
    model_path = args.model
    layer = args.layer
    text_path = Path(args.text_path)
    batch_size = args.batch_size
    exp_name = args.exp_name
    dataset_name = args.dataset
    center_crop = not args.no_center_crop

    with open(text_path, "r") as fp:
        text_data = fp.read().split("\n")

    base_outdir = Path(args.out_dir)
    clip_dir = base_outdir / args.clip_model
    clip_dir.mkdir(exist_ok=True, parents=True)

    outdir = base_outdir / exp_name
    outdir.mkdir(exist_ok=True)

    args_dict = vars(args)
    with open(outdir / "args.yaml", "w") as fp:
        yaml.dump(args_dict, fp)

    # Setup model and feature extraction
    model, preprocess = get_model([layer], model_path=model_path, center_crop=center_crop)
    model.model.to(device)
    clip_model = get_clip_model(args.clip_model)
    clip_model.to(device)

    if dataset_name == "imagenet":
        print("Loading ImageNet train set.")
        dataset = get_imagenet_dataset("train", CLIP_IMAGENET_TRANSFORMATION, 0.2)
    elif dataset_name == "mimic-cxr":
        print("Loading MIMIC-CXR train set.")
        dataset = get_mimic_dataset("train", RESIZE_TRANSFORMATION)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognised!!")

    clip_embedding_path = clip_dir / f"clip_embeddings_{dataset_name}.npy"
    text_embedding_path = clip_dir / f"clip_text_embeddings_{text_path.stem}.npy"
    model_embedding_path = outdir / f"model_embeddings_{dataset_name}.npy"

    if clip_embedding_path.exists():
        print("Loading CLIP embeddings from disk.")
        clip_embeddings = np.load(str(clip_embedding_path))
    else:
        print("Obtaining CLIP embeddings...")
        clip_embeddings = obtain_ftrs(clip_model, dataset)
        np.save(str(clip_embedding_path), clip_embeddings)

    if text_embedding_path.exists():
        print("Loading text embeddings from disk.")
        text_embeddings = np.load(str(text_embedding_path))
    else:
        print("Obtaining text embeddings...")
        text_embeddings = get_text_embeddings(clip_model, text_data)
        np.save(str(text_embedding_path), text_embeddings)

    if model_embedding_path.exists():
        print("Loading model embeddings from disk.")
        model_embeddings = np.load(str(model_embedding_path))
    else:
        print("Obtaining model embeddings...")
        model_embeddings = obtain_ftrs(model, dataset)
        np.save(str(model_embedding_path), model_embeddings)

    feature_converter = FeatureConverter()
    feature_converter.load_model(outdir)

    dataloader = feature_converter.get_dataloader(
        clip_embeddings * feature_converter.variance_coefs["clip"],
        text_embeddings * feature_converter.variance_coefs["clip_text"],
        model_embeddings * feature_converter.variance_coefs["target"],
        batch_size=batch_size
    )
    for features in dataloader:
        break
    clip_img_features, clip_text_features, target_img_features = features

    sns.displot(clip_img_features.cpu().flatten())
    plt.show()

    sns.displot(target_img_features.cpu().flatten())
    plt.show()

    non_zero_target_features = target_img_features.cpu().flatten()
    non_zero_target_features = non_zero_target_features[non_zero_target_features != 0]
    sns.displot(non_zero_target_features)
    plt.show()

    with torch.no_grad():
        converted_target_features = feature_converter.to_model(clip_img_features.cuda())
        sns.displot(converted_target_features.cpu().flatten())
        plt.show()

        non_zero_converted_features = converted_target_features.cpu().flatten()
        non_zero_converted_features = non_zero_converted_features[non_zero_converted_features != 0]
        sns.displot(non_zero_converted_features)
        plt.show()

        converted_clip_features = feature_converter.to_clip(target_img_features.cuda())
        sns.displot(converted_clip_features.cpu().flatten())
        plt.show()

    metrics = feature_converter.test(dataloader)
    print(metrics)

    print("Done!")


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
        "--out-dir",
        default="models/feature_converters",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        help="Dataset to train feature converter on."
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="No. epochs to train for"
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="No. images per batch"
    )
    parser.add_argument(
        "--text-path",
        default="./data/text_concepts/tulu_4bit_00.txt",
        help="Path to text examples to use in cycle consistency loss"
    )
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Whether to center crop the images or not"
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
    main(parser.parse_args())
