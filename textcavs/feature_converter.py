
from copy import deepcopy
import json
from pathlib import Path

import torch
import numpy as np


class FeatureConverter:
    def __init__(self):
        self.h = None
        self.h_inv = None
        self.cycle_coef = 0.1
        self.mse_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.MSELoss()
        self.init_result = None
        self.target_variance = 4.5
        self.variance_coefs = {
            "target": None,
            "clip": None,
            "clip_text": None,
        }

    def to_model(self, features):
        return self.h(features.float())

    def to_clip(self, features):
        return self.h_inv(features.float())

    def get_variance_coef(self, features):
        var = self.get_variance(features)
        c = self.target_variance / var
        c = c**0.5
        return c

    def loss(self, clip_img_features, clip_text_features, target_img_features):
        # MSE Loss
        out_target = self.h_inv(target_img_features)
        out_clip = self.h(clip_img_features)

        mse_loss_backwards = self.mse_loss(out_target, clip_img_features)
        mse_loss_forwards = self.mse_loss(out_clip, target_img_features)

        # Cycle Loss
        recovered_target = self.h(out_target)
        recovered_clip = self.h_inv(out_clip)
        out_text = self.h(clip_text_features)
        recovered_text = self.h_inv(out_text)

        cycle_loss_target = self.cycle_loss(recovered_target, target_img_features)
        cycle_loss_clip = self.cycle_loss(recovered_clip, clip_img_features)
        cycle_loss_text = self.cycle_loss(recovered_text, clip_text_features)

        # Combining the losses
        mse_loss = mse_loss_forwards + mse_loss_backwards
        cycle_loss = cycle_loss_clip + cycle_loss_target + cycle_loss_text

        loss = mse_loss + self.cycle_coef*cycle_loss

        output = {
            "loss": loss,
            "mse": mse_loss,
            "mse_forwards": mse_loss_forwards,
            "mse_backwards": mse_loss_backwards,
            "cycle": cycle_loss,
            "cycle_target": cycle_loss_target,
            "cycle_clip": cycle_loss_clip,
            "cycle_text": cycle_loss_text,
        }
        return output

    @staticmethod
    def get_dataloader(
            clip_img_features: np.ndarray,
            clip_text_features: np.ndarray,
            target_img_features: np.ndarray,
            batch_size: int = 100,
    ):
        clip_img_features = torch.from_numpy(clip_img_features).float()
        clip_text_features = torch.from_numpy(clip_text_features).float()
        target_img_features = torch.from_numpy(target_img_features).float()

        if len(clip_img_features) > len(clip_text_features):
            print(f"Number of text prompts ({len(clip_text_features)}) "
                  f"less than number of images. ({len(clip_img_features)}) "
                  "Will reuse text data during each epoch.")
            clip_text_features = FeatureConverter.extend_text_features(
                clip_text_features,
                clip_img_features.shape[0]
            )
        elif len(clip_img_features) < len(clip_text_features):
            print(f"Number of text prompts ({len(clip_text_features)}) "
                  f"greater than number of images ({len(clip_img_features)}). "
                  "Will use random subset of text data.")
            rng = np.random.default_rng(42)
            text_indices = rng.choice(
                np.arange(len(clip_text_features)),
                clip_img_features.shape[0],
                replace=False,
            )
            clip_text_features = clip_text_features[text_indices]

        dataset = torch.utils.data.TensorDataset(
            clip_img_features,
            clip_text_features,
            target_img_features,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        return dataloader

    @staticmethod
    def extend_text_features(text_features, target_length):
        n_repeats = target_length // text_features.shape[0]
        remaining_elements = target_length - (n_repeats * text_features.shape[0])
        out = text_features.repeat(n_repeats, 1)

        if remaining_elements > 0:
            out = torch.cat([out, text_features[:remaining_elements]], dim=0)
        return out

    def train(
            self,
            clip_img_features: np.ndarray,
            clip_text_features: np.ndarray,
            target_img_features: np.ndarray,
            bias=True,
            batch_size=256,
            epochs=20,
            clip_grad=1.0,
            forwards_relu=False,
            backwards_relu=False,
            mlp=False,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Scale feature spaces to have the same variance
        self.variance_coefs["clip"] = self.get_variance_coef(clip_img_features)
        self.variance_coefs["clip_text"] = self.get_variance_coef(clip_text_features)
        self.variance_coefs["target"] = self.get_variance_coef(target_img_features)

        dataloader = self.get_dataloader(
            clip_img_features * self.variance_coefs["clip"],
            clip_text_features * self.variance_coefs["clip_text"],
            target_img_features * self.variance_coefs["target"],
            batch_size,
        )

        forwards_shapes = (clip_img_features.shape[1], target_img_features.shape[1])
        backwards_shapes = (target_img_features.shape[1], clip_img_features.shape[1])
        if mlp:
            self.h = MLP(*forwards_shapes, final_relu=forwards_relu, bias=bias)
            self.h_inv = MLP(*backwards_shapes, final_relu=backwards_relu, bias=bias)
        else:
            if forwards_relu:
                self.h = ReluModel(*forwards_shapes, bias=bias)
            else:
                self.h = LinearModel(*forwards_shapes, bias=bias)
            if backwards_relu:
                self.h_inv = ReluModel(*backwards_shapes, bias=bias)
            else:
                self.h_inv = LinearModel(*backwards_shapes, bias=bias)

        #lr = 0.05
        lr = 1e-4
        momentum = 0.9
        wd = 5e-4
        t_max = 200

        optimizer_h = torch.optim.Adam(self.h.parameters(), lr=lr, weight_decay=wd)
        scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h, T_max=t_max)

        optimizer_h_inv = torch.optim.Adam(self.h_inv.parameters(), lr=lr, weight_decay=wd)
        scheduler_h_inv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h_inv, T_max=t_max)

        self.h.to(device)
        self.h_inv.to(device)

        init_metrics = self.test(dataloader)
        print(
            f'Initial '
            f'MSE: {init_metrics["mse"]:.3f}, '
            f'h R^2: {init_metrics["forwards_r"]:.3f}, '
            f'h_inv R^2: {init_metrics["backwards_r"]:.3f}, '
            f'cycle: {init_metrics["cycle"]:.3f}')

        self.init_result = init_metrics
        self.h.train()
        self.h_inv.train()

        all_loss = {}
        lrs = []
        for epoch in range(epochs):
            e_loss, num_of_batches = None, 0
            learning_rate = optimizer_h.param_groups[0]['lr']
            lrs.append(learning_rate)

            for batch_idx, features in enumerate(dataloader):
                num_of_batches += 1
                features = [v.to(device) for v in features]

                optimizer_h.zero_grad()
                optimizer_h_inv.zero_grad()

                loss_output = self.loss(*features)
                loss = loss_output["loss"]

                if e_loss is None:
                    e_loss = {k: v.item() for k, v in loss_output.items()}
                else:
                    for k in loss_output.keys():
                        e_loss[k] += loss_output[k].item()

                loss.backward()
                # Clip gradients to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(self.h.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(self.h_inv.parameters(), clip_grad)

                optimizer_h.step()
                optimizer_h_inv.step()

            e_loss = {k: v / num_of_batches for k, v in e_loss.items()}

            print_text = ", ".join([f"{k}: {v:.3f}" for k, v in e_loss.items()])
            print(f'Epoch: {epoch}, lr: {learning_rate:.5f}, {print_text}')

            for k, v in e_loss.items():
                if k not in all_loss.keys():
                    all_loss[k] = [v]
                else:
                    all_loss[k].append(v)

            scheduler_h.step()
            scheduler_h_inv.step()

        all_loss["lr"] = lrs
        final_metrics = self.test(dataloader)
        print(
            f'Final '
            f'MSE: {final_metrics["mse"]:.3f}, '
            f'h R^2: {final_metrics["forwards_r"]:.3f}, '
            f'h_inv R^2: {final_metrics["backwards_r"]:.3f}, '
            f'cycle: {final_metrics["cycle"]:.3f}'
        )
        return all_loss

    def test(
            self,
            dataloader
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.h.eval()
        self.h_inv.eval()

        num_of_batches = 0
        r_squared = {"forward": [], "backward": []}
        outputs = None
        with torch.no_grad():
            for batch_idx, features in enumerate(dataloader):
                num_of_batches += 1
                features = [v.to(device) for v in features]

                loss_output = self.loss(*features)
                loss_output = {k: v.item() for k, v in loss_output.items()}

                r_squared_forward, r_squared_backward = self.get_r_squared(*features)
                r_squared["forward"].append(r_squared_forward)
                r_squared["backward"].append(r_squared_backward)

                if outputs is None:
                    outputs = deepcopy(loss_output)
                else:
                    for k in loss_output.keys():
                        outputs[k] += loss_output[k]

        outputs = {k: v/num_of_batches for k, v in outputs.items()}

        outputs["forwards_r"] = torch.tensor(r_squared["forward"]).mean()
        outputs["backwards_r"] = torch.tensor(r_squared["backward"]).mean()
        return outputs

    def get_r_squared(self, clip_img_features, clip_text_features, target_img_features):

        clip_output = self.h_inv(target_img_features)
        target_output = self.h(clip_img_features)

        ss_res_forwards = ((target_output - target_img_features)**2).sum()
        target_mean = target_img_features.mean(dim=0)
        ss_tot_forwards = ((target_output - target_mean)**2).sum()
        r_squared_forwards = 1 - (ss_res_forwards / ss_tot_forwards)

        ss_res_backwards = ((clip_output - clip_img_features)**2).sum()
        clip_mean = clip_img_features.mean(dim=0)
        ss_tot_backwards = ((clip_output - clip_mean) ** 2).sum()
        r_squared_backwards = 1 - (ss_res_backwards / ss_tot_backwards)

        return r_squared_forwards, r_squared_backwards

    @staticmethod
    def get_variance(y: np.ndarray):
        ey = np.mean(y)
        ey2 = np.mean(np.square(y))
        return ey2 - ey**2

    def save_model(self, outdir: Path):
        torch.save(self.h, outdir / "h.pth")
        torch.save(self.h_inv, outdir / "h_inv.pth")
        with open(outdir / "variance_coefs.json", "w") as fp:
            json.dump(self.variance_coefs, fp, indent=2)

    def load_model(self, outdir: Path):
        h_path = outdir / "h.pth"
        h_inv_path = outdir / "h_inv.pth"
        coefs_path = outdir / "variance_coefs.json"

        if h_path.exists() and h_inv_path.exists():
            self.h = torch.load(h_path)
            self.h_inv = torch.load(h_inv_path)
        else:
            raise FileExistsError(f"One of {h_path} or {h_inv_path} is missing")

        if coefs_path.exists():
            with open(coefs_path, "r") as fp:
                self.variance_coefs = json.load(fp)


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


class ReluModel(LinearModel):
    def forward(self, x):
        out = self.linear(x)
        out = torch.relu(out)
        return out

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, bias=True, final_relu=False):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = torch.nn.Linear(hidden_size, output_size, bias=bias)
        self.final_relu = final_relu

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        if self.final_relu:
            out = self.relu(out)
        return out

