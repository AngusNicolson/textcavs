
import copy
from typing import Sequence, Optional, Callable

import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
import torchvision.transforms as T
import numpy as np
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoModel, AutoTokenizer


from textcavs.utils import IMAGENET_MEAN, IMAGENET_STD


class ModelWrapper(torch.nn.Module):
    """Model wrapper to hold pytorch image models and set up the needed
    hooks to access the activations and grads.
    """

    def __init__(self, model: torch.nn.Module, bottlenecks: dict, labels: Sequence[str],
                 *args, **kwargs):
        """Initialize wrapper with model and set up the hooks to the bottlenecks.
        Args:
            model (nn.Module): Model to test
            bottlenecks (dict): Dictionary attaching names to the layers to hook into. Expects, at least, an input,
                logit and prediction.
            labels (list): Class labels in order the model expects
        """
        super().__init__(*args, **kwargs)
        self.ends = None
        self.y_input = None
        self.loss = None
        self.bottlenecks_gradients = None
        self.bottlenecks_tensors = {}
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.shape = (224, 224)
        self.labels = labels
        self.bottlenecks = []

        def save_activation(name):
            """Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            """

            def hook(module, input, output):
                """Saves the activation hook to dictionary"""
                self.bottlenecks_tensors[name] = output

            return hook

        for name, mod in self.model.named_modules():
            if name in bottlenecks.keys():
                mod.register_forward_hook(save_activation(bottlenecks[name]))
                self.bottlenecks.append(name)

    def _make_gradient_tensors(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """
        Makes gradient tensor for logit y w.r.t. layer with activations

        Args:
            x (tensor): Model input
            y (int): Index of logit (class)
            bottleneck_name (string): Name of layer activations
        Returns:
            (torch.tensor): Gradients of logit w.r.t. to activations
        """
        out = self.model(x.unsqueeze(0))
        acts = self.bottlenecks_tensors[bottleneck_name]
        return grad(out[:, y], acts)[0]

    def __call__(self, x: torch.Tensor):
        """Calls prediction on wrapped model."""
        self.ends = self.model(x)
        return self.ends

    def get_gradient(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """Returns the gradient at a given bottle_neck.
        Args:
            x: Model input
            y: Index of the logit layer (class)
            bottleneck_name: Name of the bottleneck to get gradients w.r.t.
        Returns:
            (torch.tensor): Tensor containing the gradients at layer.
        """
        self.y_input = y
        return self._make_gradient_tensors(x, y, bottleneck_name)

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)


def to_rgb(img):
    return img.convert("RGB")


def get_model(bottlenecks, model_path=None, class_names=None, center_crop=True):
    layer = bottlenecks[0]
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    if model_path is not None:
        state_dict = torch.load(model_path)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]

            for k, v in state_dict.items():
                break
            if k == "model.conv1.weight":
                state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
            fc_shape = state_dict["fc.weight"].shape
            model.fc = nn.Linear(fc_shape[1], fc_shape[0])
        model.load_state_dict(
            state_dict
        )
    if type(bottlenecks) is not dict:
        bottlenecks = {v: v for v in bottlenecks}

    model = ModelWrapper(model, bottlenecks, class_names)
    model.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = T.Normalize(mean=mean, std=std)

    if center_crop:
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            to_rgb,
            T.ToTensor(),
            normalize
        ])
    else:
        preprocess = T.Compose([
            T.Resize(224),
            to_rgb,
            T.ToTensor(),
            normalize
        ])

    def get_forward_features(x):
        model(x)
        return model.bottlenecks_tensors[layer]

    model.forward_features = lambda x: get_forward_features(x)
    model.get_normalizer = T.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    model.has_normalizer = True

    return model, preprocess


class ClipZeroShot(torch.nn.Module):
    def __init__(self, mtype):
        super(ClipZeroShot, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = T.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False

    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)
        return image_features

    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)

    def tokenize(self, text):
        return clip.tokenize(text)


class OpenCLIPModel(torch.nn.Module):
    def __init__(self, url_str):
        super(OpenCLIPModel, self).__init__()
        self.clip_model, self.clip_preprocess = create_model_from_pretrained(url_str)
        self.tokenizer = get_tokenizer(url_str)
        self.to_pil = T.ToPILImage()
        self.mtype = url_str
        self.has_normalizer = False

    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)
        return image_features

    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)

    def tokenize(self, text):
        return self.tokenizer(text)


class MLP(nn.Module):
    """
    Fully connected layers to map between image embeddings and projection space where pairs of images are compared.

    :param input_dim: Input embedding feature size
    :param hidden_dim: Hidden layer size in MLP
    :param output_dim: Output projection size
    :param use_1x1_convs: Use 1x1 conv kernels instead of 2D linear transformations for speed and memory efficiency.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, use_1x1_convs: bool = False
    ) -> None:
        super().__init__()

        if use_1x1_convs:
            linear_proj_1_args = {'in_channels': input_dim, 'out_channels': hidden_dim, 'kernel_size': 1, 'bias': False}
            linear_proj_2_args = {'in_channels': hidden_dim, 'out_channels': output_dim, 'kernel_size': 1, 'bias': True}
            normalisation_layer: Callable = nn.BatchNorm2d
            projection_layer: Callable = nn.Conv2d
        else:
            linear_proj_1_args = {'in_features': input_dim, 'out_features': hidden_dim, 'bias': False}
            linear_proj_2_args = {'in_features': hidden_dim, 'out_features': output_dim, 'bias': True}
            normalisation_layer = nn.BatchNorm1d
            projection_layer = nn.Linear

        self.output_dim = output_dim
        self.input_dim = input_dim
        if hidden_dim is not None:
            self.model = nn.Sequential(
                projection_layer(**linear_proj_1_args),
                normalisation_layer(hidden_dim),
                nn.ReLU(inplace=True),
                projection_layer(**linear_proj_2_args),
            )
        else:
            self.model = nn.Linear(input_dim, output_dim)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of the multi-layer perceptron"""
        x = self.model(x)
        return x


class ImageModel(nn.Module):
    def __init__(self, joint_feature_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder, self.preprocess = get_model(["avgpool"])
        self.feature_size = 2048
        self.projector = MLP(
            input_dim=self.feature_size,
            output_dim=joint_feature_size,
            hidden_dim=joint_feature_size,
            use_1x1_convs=True,
        )

    def forward(self, x):
        out = self.encoder.forward_features(x)
        out = self.projector(out)
        return out


class BioViLModel(torch.nn.Module):
    def __init__(self, local_dir, image_model_name="biovil_image_resnet50_proj_size_128.pt", proj_size=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
        self.image_model = self.load_image_model(local_dir, image_model_name, proj_size)
        self.has_normalizer = False

    @staticmethod
    def load_image_model(local_dir, image_model_name, proj_size=128):
        image_model = ImageModel(proj_size)
        image_model_sd_base = torch.load(local_dir + f"/{image_model_name}")
        image_model_sd = {}
        for k, v in image_model_sd_base.items():
            if k.startswith("encoder.encoder"):
                new_k = k[8:]
                new_k = new_k.split(".")
                new_k[1] = "model." + new_k[1]
                new_k = ".".join(new_k)
            else:
                new_k = k
            image_model_sd[new_k] = v
        image_model.load_state_dict(image_model_sd)
        return image_model

    def forward_features(self, img):
        image_features = self.image_model(img)
        return image_features

    def encode_text(self, tokenizer_output):
        out = self.text_model.get_projected_text_embeddings(
            input_ids=tokenizer_output.input_ids,
            attention_mask=tokenizer_output.attention_mask
        )
        return out

    def tokenize(self, text):
        out = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
        return out


def get_clip_model(model_name):
    if model_name == "BiomedCLIP":
        url_str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        model = OpenCLIPModel(url_str)
    elif model_name == "BioViL":
        local_dir = "/home/lina3782/dev/BiomedVLP-CXR-BERT-specialized"
        model = BioViLModel(local_dir)
    else:
        model = ClipZeroShot(model_name)
    return model

