from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from absl import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

import bnaf

DEVICE = "cpu"


# BNAF for Domias
def load_dataset(
    data_train: Optional[np.ndarray] = None,
    data_valid: Optional[np.ndarray] = None,
    data_test: Optional[np.ndarray] = None,
    device: Any = DEVICE,
    batch_dim: int = 50,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    if data_train is not None:
        dataset_train = torch.utils.data.TensorDataset(
            torch.from_numpy(data_train).float().to(device)
        )
        if data_valid is None:
            logging.debug("No validation set passed")
            data_valid = np.random.randn(*data_train.shape)
        if data_test is None:
            logging.debug("No test set passed")
            data_test = np.random.randn(*data_train.shape)

        dataset_valid = torch.utils.data.TensorDataset(
            torch.from_numpy(data_valid).float().to(device)
        )

        dataset_test = torch.utils.data.TensorDataset(
            torch.from_numpy(data_test).float().to(device)
        )
    else:
        raise RuntimeError()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_dim, shuffle=True
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_dim, shuffle=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_dim, shuffle=False
    )

    return data_loader_train, data_loader_valid, data_loader_test


def create_model(
    n_dims: int,
    n_flows: int = 5,
    n_layers: int = 3,
    hidden_dim: int = 32,
    residual: Optional[str] = "gated",  # [None, "normal", "gated"]
    verbose: bool = False,
    device: Any = DEVICE,
    batch_dim: int = 50,
) -> nn.Module:
    flows: List = []
    for f in range(n_flows):
        layers: List = []
        for _ in range(n_layers - 1):
            layers.append(
                bnaf.MaskedWeight(
                    n_dims * hidden_dim,
                    n_dims * hidden_dim,
                    dim=n_dims,
                )
            )
            layers.append(bnaf.Tanh())

        flows.append(
            bnaf.BNAF(
                *(
                    [
                        bnaf.MaskedWeight(n_dims, n_dims * hidden_dim, dim=n_dims),
                        bnaf.Tanh(),
                    ]
                    + layers
                    + [bnaf.MaskedWeight(n_dims * hidden_dim, n_dims, dim=n_dims)]
                ),
                res=residual if f < n_flows - 1 else None,
            )
        )

        if f < n_flows - 1:
            flows.append(bnaf.Permutation(n_dims, "flip"))

    model = bnaf.Sequential(*flows).to(device)

    return model


def load_model(
    model: nn.Module,
    optimizer: Any,
    workspace: Path = Path("workspace"),
) -> Callable:
    def f() -> None:
        if workspace.exists():
            return

        logging.info("Loading model..")
        if (workspace / "checkpoint.pt").exists():
            checkpoint = torch.load(workspace / "checkpoint.pt")  # nosec B614
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    return f


def save_model(
    model: nn.Module,
    optimizer: Any,
    epoch: int,
    save: bool = False,
    workspace: Path = Path("workspace"),
) -> Callable:
    workspace.mkdir(parents=True, exist_ok=True)

    def f() -> None:
        if save:
            logging.debug("Saving model..")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                workspace / "DomiasMIA_bnaf_checkpoint.pt",
            )  # nosec B614

    return f


def train(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    data_loader_train: torch.utils.data.DataLoader,
    data_loader_valid: torch.utils.data.DataLoader,
    data_loader_test: torch.utils.data.DataLoader,
    workspace: Path = Path("workspace"),
    start_epoch: int = 0,
    device: Any = DEVICE,
    epochs: int = 50,
    save: bool = False,
    clip_norm: float = 0.1,
) -> Callable:
    epoch = start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        t = tqdm(data_loader_train, smoothing=0, ncols=80, disable=True)
        # train_loss: torch.Tensor = []
        running_train_loss = 0.0

        for (x_mb,) in t:
            loss = -compute_log_p_x(model, x_mb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            t.set_postfix(loss="{:.2f}".format(loss.item()), refresh=False)
            # train_loss.append(loss) 
            running_train_loss += loss.item()

        # train_loss = torch.stack(train_loss).mean()
        train_loss = running_train_loss / len(data_loader_train)
        optimizer.swap()
        validation_loss = -torch.stack(
            [
                compute_log_p_x(model, x_mb).mean().detach()
                for x_mb, in data_loader_valid
            ],
            -1,
        ).mean()
        optimizer.swap()

        logging.debug(
            "Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}".format(
                epoch + 1,
                start_epoch + epochs,
                train_loss.item(),
                validation_loss.item(),
            )
        )

        stop = scheduler.step(
            validation_loss,
            callback_best=save_model(
                model, optimizer, epoch + 1, save=save, workspace=workspace
            ),
            callback_reduce=load_model(model, optimizer, workspace=workspace),
        )

        if stop:
            break

    load_model(model, optimizer, workspace=workspace)()
    optimizer.swap()
    # validation_loss = -torch.stack(
    #     [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_valid],
    #     -1,
    # ).mean()
    # test_loss = -torch.stack(
    #     [compute_log_p_x(model, x_mb).mean().detach() for x_mb, in data_loader_test], -1
    # ).mean()

    validation_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for x_mb, in data_loader_valid:
            loss = -compute_log_p_x(model, x_mb).mean()
            validation_loss += loss.item()

    validation_loss /= len(data_loader_valid)

    # Apply the same pattern for the test loss
    test_loss = 0.0
    with torch.no_grad():
        for x_mb, in data_loader_test:
            loss = -compute_log_p_x(model, x_mb).mean()
            test_loss += loss.item()

    test_loss /= len(data_loader_test)

    logging.debug(
        f"""
        ###### Stop training after {epoch + 1} epochs!
        Validation loss: {validation_loss.item():4.3f}
        Test loss:       {test_loss.item():4.3f}
        """
    )

    if save:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            workspace / "checkpoint.pt",
        )  # nosec B614
        logging.debug(
            f"""
            ###### Stop training after {epoch + 1} epochs!
            Validation loss: {validation_loss.item():4.3f}
            Test loss:       {test_loss.item():4.3f}
            """
        )

    def p_func(x: np.ndarray) -> np.ndarray:
        return np.exp(compute_log_p_x(model, x))

    return p_func


def density_estimator_trainer(
    data_train: np.ndarray,
    data_val: Optional[np.ndarray] = None,
    data_test: Optional[np.ndarray] = None,
    batch_dim: int = 50,
    flows: int = 5,
    layers: int = 3,
    hidden_dim: int = 32,
    residual: Optional[str] = "gated",  # [None, "normal", "gated"]
    workspace: Path = Path("workspace"),
    decay: float = 0.5,
    patience: int = 20,
    cooldown: int = 10,
    min_lr: float = 5e-4,
    early_stopping: int = 100,
    device: Any = DEVICE,
    epochs: int = 50,
    learning_rate: float = 1e-2,
    clip_norm: float = 0.1,
    polyak: float = 0.998,
    save: bool = True,
    load: bool = True,
) -> Tuple[Callable, nn.Module]:
    logging.debug("Loading dataset..")
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(
        data_train,
        data_val,
        data_test,
        device=device,
        batch_dim=batch_dim,
    )

    if save:
        logging.debug("Creating directory experiment..")
        workspace.mkdir(parents=True, exist_ok=True)

    logging.debug("Creating BNAF model..")
    model = create_model(
        data_train.shape[1],
        batch_dim=batch_dim,
        n_flows=flows,
        n_layers=layers,
        hidden_dim=hidden_dim,
        verbose=True,
        device=device,
    )

    logging.debug("Creating optimizer..")
    optimizer = bnaf.Adam(
        model.parameters(), lr=learning_rate, amsgrad=True, polyak=polyak
    )

    logging.debug("Creating scheduler..")

    scheduler = bnaf.ReduceLROnPlateau(
        optimizer,
        factor=decay,
        patience=patience,
        cooldown=cooldown,
        min_lr=min_lr,
        # verbose=True,
        early_stopping=early_stopping,
        threshold_mode="abs",
    )

    if load:
        load_model(model, optimizer, workspace=workspace)()

    logging.debug("Training..")
    p_func = train(
        model,
        optimizer,
        scheduler,
        data_loader_train,
        data_loader_valid,
        data_loader_test,
        workspace=workspace,
        device=device,
        epochs=epochs,
        save=save,
        clip_norm=clip_norm,
    )
    return p_func, model


def compute_log_p_x(model: nn.Module, x_mb: torch.Tensor) -> torch.Tensor:
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = (
        torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
        .log_prob(y_mb)
        .sum(-1)
    )
    return log_p_y_mb + log_diag_j_mb


class DomiasMIABNAF:
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_BNAF"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any = DEVICE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, p_G_model = density_estimator_trainer(
            synth_set.values,
            synth_val_set.values[: int(0.5 * synth_val_set.shape[0])],
            synth_val_set.values[int(0.5 * synth_val_set.shape[0]) :],
            device=device
        )
        _, p_R_model = density_estimator_trainer(reference_set, device=device)
        p_G_evaluated = np.exp(
            compute_log_p_x(p_G_model, torch.as_tensor(X_test).float().to(device))
            .cpu()
            .detach()
            .numpy()
        )
        p_R_evaluated = np.exp(
            compute_log_p_x(p_R_model, torch.as_tensor(X_test).float().to(device))
            .cpu()
            .detach()
            .numpy()
        )
        return p_G_evaluated, p_R_evaluated
