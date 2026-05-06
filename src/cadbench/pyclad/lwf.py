"""Learning without Forgetting (LwF) strategy for PyCLAD autoencoders."""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.strategies.strategy import (
    ConceptAgnosticStrategy,
    ConceptIncrementalStrategy,
)


class LwFStrategy(ConceptIncrementalStrategy, ConceptAgnosticStrategy):
    def __init__(
        self,
        model: Model,
        alpha: float = 0.5,
        distill_mode: str = "hybrid",
        adaptive_balance: bool = True,
        epochs: int | None = None,
    ):
        if distill_mode not in {"latent", "reconstruction", "hybrid"}:
            raise ValueError("distill_mode must be 'latent', 'reconstruction', or 'hybrid'")
        if epochs is not None and int(epochs) <= 0:
            raise ValueError("epochs must be positive")

        self._model = model
        self._old_model: Model | None = None
        self._alpha = alpha
        self._distill_mode = distill_mode
        self._adaptive_balance = adaptive_balance
        self._epochs = int(epochs) if epochs is not None else None
        self._task_count = 0

        self._module()

    def learn(self, data, *args, **kwargs):
        if self._old_model is None:
            owner = next((model for model in self._iter_wrapped() if hasattr(model, "epochs")), None)
            if self._epochs is None or owner is None:
                self._model.fit(data)
            else:
                previous_epochs = owner.epochs
                owner.epochs = self._epochs
                try:
                    self._model.fit(data)
                finally:
                    owner.epochs = previous_epochs
        else:
            self._fit_with_distillation(data)
        self._task_count += 1
        self._old_model = self._clone_model()

    def predict(self, data, *args, **kwargs) -> tuple:
        return self._model.predict(data)

    def name(self) -> str:
        return "LwF"

    def additional_info(self):
        return {
            "model": self._model.name(),
            "alpha": self._alpha,
            "distill_mode": self._distill_mode,
            "adaptive_balance": self._adaptive_balance,
            "epochs": self._epochs if self._epochs is not None else int(self._resolve_attr("epochs", 20) or 20),
            "task_count": self._task_count,
            "has_old_model": self._old_model is not None,
        }

    def _iter_wrapped(self, model=None):
        current = self._model if model is None else model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = getattr(current, "model", None)

    def _module(self, model=None, *, required=True):
        for current in self._iter_wrapped(model):
            module = current if isinstance(current, nn.Module) else getattr(current, "module", None)
            if isinstance(module, nn.Module):
                return module
        if required:
            raise TypeError("LwFStrategy requires a PyCLAD autoencoder-like torch `.module`.")
        return None

    def _resolve_attr(self, name, default, model=None):
        for current in self._iter_wrapped(model):
            for target in (current, getattr(current, "module", None)):
                if hasattr(target, name):
                    return getattr(target, name)
        return default

    def _resolve_shuffle(self) -> bool:
        models = [*self._iter_wrapped(), self._module()]
        return not any("TemporalAutoencoder" in model.__class__.__name__ for model in models)

    def _prepare_data(self, data):
        transformed = np.asarray(data)
        if any(current.__class__.__name__ == "FlattenTimeSeriesAdapter" for current in self._iter_wrapped()):
            transformed = transformed.reshape(transformed.shape[0], -1)
        return torch.from_numpy(np.array(transformed, dtype=np.float32, copy=True))

    def _clone_model(self) -> Model:
        cloned = copy.deepcopy(self._model)
        module = self._module(cloned)
        module.eval()
        module.requires_grad_(False)
        return cloned

    def _resolve_distill_mode(self) -> str:
        if self._old_model is None:
            raise RuntimeError("LwF distillation requires a previous-task teacher.")
        if self._distill_mode == "reconstruction":
            return self._distill_mode
        module = self._module(required=False)
        old_module = self._module(self._old_model, required=False)
        if not all(module is not None and hasattr(module, "encoder") for module in (module, old_module)):
            return "reconstruction"
        return self._distill_mode

    @staticmethod
    def _latent_values(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and all(torch.is_tensor(item) for item in output):
            return tuple(output)
        raise TypeError("Latent distillation requires encoder outputs to be tensors.")

    def _encode_with_model(self, model, batch):
        module = self._module(model)
        if not hasattr(module, "encoder"):
            raise TypeError("Latent distillation requires module.encoder.")
        return self._latent_values(module.encoder(batch))

    @staticmethod
    def _is_variational(module) -> bool:
        return "Variational" in module.__class__.__name__ and hasattr(module, "encoder") and hasattr(module, "decoder")

    def _forward_module(self, module, batch, *, deterministic_variational=False):
        if deterministic_variational and self._is_variational(module):
            latent = self._latent_values(module.encoder(batch))
            mean = latent[0] if isinstance(latent, tuple) else latent
            extras = latent if isinstance(latent, tuple) else tuple()
            return module.decoder(mean), extras

        output = module(batch)
        return (output[0], tuple(output[1:])) if isinstance(output, (tuple, list)) else (output, tuple())

    def _reconstruction_loss(self, module, batch, x_hat, extras):
        train_loss = getattr(module, "train_loss", None)
        if callable(train_loss):
            if len(extras) >= 2:
                try:
                    return train_loss(batch, x_hat, *extras[:2])
                except TypeError:
                    pass
            return train_loss(x_hat, batch)
        return F.mse_loss(x_hat, batch)

    def _training_batch_step(self, batch, *, include_latent=False):
        module = self._module()
        x_hat, extras = self._forward_module(module, batch)
        rec_loss = self._reconstruction_loss(module, batch, x_hat, extras)
        z = self._encode_with_model(self._model, batch) if include_latent and hasattr(module, "encoder") else None
        return rec_loss, x_hat, z, extras

    def _compute_distill_loss(self, batch, distill_mode, x_hat_new, z_new):
        reconstruction_loss = None
        if distill_mode in {"reconstruction", "hybrid"}:
            if self._is_variational(self._module()):
                x_hat_new, _ = self._forward_module(self._module(), batch, deterministic_variational=True)
            with torch.no_grad():
                x_hat_old, _ = self._forward_module(
                    self._module(self._old_model), batch, deterministic_variational=True
                )
            reconstruction_loss = F.mse_loss(x_hat_new, x_hat_old)
            if distill_mode == "reconstruction":
                return reconstruction_loss

        with torch.no_grad():
            z_old = self._encode_with_model(self._old_model, batch)
        new_values = z_new if isinstance(z_new, tuple) else (z_new,)
        old_values = z_old if isinstance(z_old, tuple) else (z_old,)
        if len(new_values) != len(old_values) or any(value is None for value in new_values):
            raise ValueError("Current and teacher latent outputs must have the same structure.")
        latent_loss = sum(F.mse_loss(current, teacher) for current, teacher in zip(new_values, old_values)) / len(
            new_values
        )
        return latent_loss if distill_mode == "latent" else (latent_loss + reconstruction_loss) / 2

    def _weight_distill_loss(self, rec_loss, distill_loss):
        if not self._adaptive_balance:
            return self._alpha * distill_loss
        distill_value = distill_loss.detach()
        if torch.all(distill_value <= torch.finfo(distill_value.dtype).eps):
            return self._alpha * distill_loss
        scale = rec_loss.detach() / distill_value.clamp_min(torch.finfo(distill_value.dtype).eps)
        return self._alpha * scale * distill_loss

    def _fit_with_distillation(self, data):
        if self._old_model is None:
            raise RuntimeError("LwF distillation requires a previous-task teacher.")

        module = self._module()
        old_module = self._module(self._old_model)
        distill_mode = self._resolve_distill_mode()
        if (explicit_device := self._resolve_attr("device", None)) is not None:
            device = torch.device(explicit_device)
        else:
            try:
                device = next(module.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        old_module.eval()
        old_module.requires_grad_(False)
        old_module.to(device)
        module.to(device)
        module.train()

        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise TypeError("LwFStrategy requires at least one trainable parameter.")

        optimizer = torch.optim.Adam(parameters, lr=float(self._resolve_attr("lr", 1e-2) or 1e-2))
        include_latent = distill_mode in {"latent", "hybrid"}
        dataloader = DataLoader(
            TensorDataset(self._prepare_data(data)),
            batch_size=int(self._resolve_attr("batch_size", 32) or 32),
            shuffle=self._resolve_shuffle(),
            num_workers=0,
        )

        epochs = self._epochs if self._epochs is not None else int(self._resolve_attr("epochs", 20) or 20)
        for _ in range(epochs):
            for (batch,) in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                rec_loss, x_hat_new, z_new, _ = self._training_batch_step(batch, include_latent=include_latent)
                distill_loss = self._compute_distill_loss(batch, distill_mode, x_hat_new, z_new)
                total_loss = rec_loss + self._weight_distill_loss(rec_loss, distill_loss)
                total_loss.backward()
                optimizer.step()

        if hasattr(self._model, "_auto_threshold") and self._model._auto_threshold:
            self._model._calibrate_threshold(data)
