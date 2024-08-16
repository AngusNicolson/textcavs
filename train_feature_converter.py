
from pathlib import Path
from argparse import ArgumentParser
import yaml

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import clip

from textcavs.feature_converter import FeatureConverter
from textcavs.model import get_model, get_clip_model
from textcavs.utils import get_imagenet_dataset, CLIP_IMAGENET_TRANSFORMATION, RESIZE_TRANSFORMATION
from textcavs.datasets import get_mimic_dataset

device = 'cuda'


def main(args):
    model_path = args.model
    layer = args.layer
    text_path = Path(args.text_path)
    n_epochs = args.epochs
    batch_size = args.batch_size
    exp_name = args.exp_name
    dataset_name = args.dataset
    center_crop = not args.no_center_crop
    forwards_relu = args.forwards_relu
    mlp = args.mlp

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
        print("Loading ImageNet training set.")
        dataset = get_imagenet_dataset("train", CLIP_IMAGENET_TRANSFORMATION, 0.2)
    elif dataset_name == "mimic-cxr":
        print("Loading MIMIC-CXR training set.")
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
    try:
        feature_converter.load_model(outdir)
        raise ValueError("Model already exists!")
    except FileExistsError:
        print("Training feature converter...")

    metrics = feature_converter.train(
        clip_embeddings,
        text_embeddings,
        model_embeddings,
        batch_size=batch_size,
        epochs=n_epochs,
        forwards_relu=forwards_relu,
        mlp=mlp
    )
    feature_converter.save_model(outdir)

    plot_metrics(
        metrics,
        ["mse", "mse_forwards", "mse_backwards"],
        savefig=outdir / "mse.png"
    )
    plot_metrics(
        metrics,
        ["cycle", "cycle_target", "cycle_clip", "cycle_text"],
        savefig=outdir / "cycle.png",
    )
    plot_metrics(
        metrics,
        ["loss", "mse", "cycle"],
        savefig=outdir / "loss.png",
    )
    plot_metrics(
        metrics,
        ["lr"],
        savefig=outdir / "lr.png",
    )

    print("Done!")


def plot_metrics(metrics_dict, keys, ylabel=None, xlabel="Epoch", savefig=None):
    fig, ax = plt.subplots()
    for key in keys:
        ax.plot(metrics_dict[key], label=key)
    if len(keys) > 1:
        plt.legend()
    if ylabel is None:
        ax.set_ylabel(keys[0])
    else:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, bbox_inches="tight")


def get_text_embeddings(clip_model, text_data, batch_size=16):
    out = []
    n = len(text_data) // batch_size
    for i in tqdm(range(n)):
        text = text_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            tokens = clip_model.tokenize(text)
            embedding = clip_model.encode_text(tokens.to(device))
            embedding = embedding.cpu().numpy()
            out.append(embedding)
    out = np.concatenate(out)
    return out


def obtain_ftrs(model, dset, batch_size=64):
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)
    return obtain_reps_given_loader(model, loader)


def obtain_reps_given_loader(model, loader):
    all_reps = []
    for imgs, _ in tqdm(loader):
        if model.has_normalizer:
            imgs = model.get_normalizer(imgs)

        imgs = imgs.to(device)
        with torch.no_grad():
            reps = model.forward_features(imgs).flatten(1)
            reps = [x.detach().cpu().numpy() for x in reps]

        all_reps.extend(reps)

    all_reps = np.stack(all_reps)
    return all_reps


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
    parser.add_argument(
        "--forwards-relu",
        action="store_true",
        help="Add ReLU activation to feature converter (CLIP --> target)"
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Use an MLP, instead of a single linear layer"
    )
    main(parser.parse_args())
