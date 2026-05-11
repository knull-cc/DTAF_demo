import argparse
import copy
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dtaf import DTAF, DTAFConfig


class StandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_ = data.mean(axis=0)
        self.scale_ = data.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.scale_

    def inverse_target(self, data: np.ndarray, target_indices: Sequence[int]) -> np.ndarray:
        target_indices = np.asarray(target_indices)
        return data * self.scale_[target_indices] + self.mean_[target_indices]

    def state_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Sequence[float]]) -> "StandardScaler":
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(state["scale"], dtype=np.float32)
        return scaler


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        dates: np.ndarray,
        seq_len: int,
        pred_len: int,
        border1: int,
        border2: int,
    ):
        self.data = data.astype(np.float32)
        self.dates = dates
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.border1 = border1
        self.border2 = border2
        self.length = border2 - border1 - seq_len - pred_len + 1
        if self.length <= 0:
            raise ValueError(
                "Split is too short for the requested seq_len and pred_len. "
                "Use shorter windows or provide more data."
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        s_begin = self.border1 + index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        return (
            torch.from_numpy(self.data[s_begin:s_end]),
            torch.from_numpy(self.data[r_begin:r_end]),
            index,
        )

    def target_dates(self, index: int) -> np.ndarray:
        s_begin = self.border1 + index
        r_begin = s_begin + self.seq_len
        return self.dates[r_begin : r_begin + self.pred_len]


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.best_state = None

    def step(self, metric: float, model: nn.Module) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args) -> torch.device:
    if args.use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    if args.use_mps and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def print_separator() -> None:
    print("-" * 80, flush=True)


def print_section(title: str) -> None:
    print_separator()
    print(title, flush=True)
    print_separator()


def print_args(args) -> None:
    print_section("loaded arguments")
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}", flush=True)


def format_metrics(metrics: Dict[str, float]) -> str:
    return f"mae {metrics['mae']:.6f} | mse {metrics['mse']:.6f}"


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args) -> None:
    if args.lradj == "none":
        return
    if args.lradj == "type1":
        lr = args.lr * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == "type2":
        lr_by_epoch = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        if epoch not in lr_by_epoch:
            return
        lr = lr_by_epoch[epoch]
    else:
        raise ValueError("lradj must be one of: none, type1, type2")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"learning_rate {lr:.8g}", flush=True)


def read_itransformer_csv(
    args,
    selected_columns: Optional[Sequence[str]] = None,
    target_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[str]]:
    path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if len(df.columns) == 0 or df.columns[0].lower() != "date":
        raise ValueError(
            "DTAF expects iTransformer-style CSV files: first column must be 'date', "
            "remaining columns must be numeric variables."
        )

    dates = pd.to_datetime(df.iloc[:, 0]).to_numpy()
    all_feature_columns = list(df.columns[1:])
    if selected_columns is None:
        if args.target not in all_feature_columns:
            raise ValueError(
                f"target '{args.target}' is not in the dataset columns: {all_feature_columns}"
            )
        if args.features == "S":
            selected_columns = [args.target]
            target_names = [args.target]
        elif args.features == "M":
            selected_columns = all_feature_columns
            target_names = list(selected_columns)
        elif args.features == "MS":
            selected_columns = all_feature_columns
            target_names = [args.target]
        else:
            raise ValueError("features must be one of: S, M, MS")
    else:
        missing = [col for col in selected_columns if col not in all_feature_columns]
        if missing:
            raise ValueError(f"Checkpoint columns are missing from the dataset: {missing}")
        selected_columns = list(selected_columns)
        target_names = list(target_names)

    values_df = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
    bad_columns = [col for col in selected_columns if values_df[col].isna().any()]
    if bad_columns:
        raise ValueError(f"Non-numeric or missing values found in columns: {bad_columns}")

    target_indices = [selected_columns.index(col) for col in target_names]
    values = values_df.to_numpy(dtype=np.float32)
    return values, dates, list(selected_columns), target_indices, list(target_names)


def split_borders(n: int, args) -> Dict[str, Tuple[int, int]]:
    if args.split == "ett":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
        if n < test_end:
            raise ValueError(
                f"ETT split requires at least {test_end} rows, but dataset has {n}."
            )
        return {
            "train": (0, train_end),
            "val": (train_end - args.seq_len, val_end),
            "test": (val_end - args.seq_len, test_end),
        }

    train_ratio, val_ratio, test_ratio = args.train_ratio, args.val_ratio, args.test_ratio
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("--train_ratio + --val_ratio + --test_ratio must equal 1.0")

    num_train = int(n * train_ratio)
    num_test = int(n * test_ratio)
    num_val = n - num_train - num_test
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    return {
        "train": (0, num_train),
        "val": (max(0, num_train - args.seq_len), num_train + num_val),
        "test": (max(0, n - num_test - args.seq_len), n),
    }


def make_loaders(
    values: np.ndarray,
    dates: np.ndarray,
    scaler: StandardScaler,
    args,
    shuffle_train: bool = True,
) -> Tuple[Dict[str, TimeSeriesDataset], Dict[str, DataLoader]]:
    scaled = scaler.transform(values).astype(np.float32)
    borders = split_borders(len(values), args)
    datasets = {
        name: TimeSeriesDataset(
            scaled,
            dates,
            args.seq_len,
            args.pred_len,
            border1,
            border2,
        )
        for name, (border1, border2) in borders.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        ),
    }
    return datasets, loaders


def channel_mixup(x: torch.Tensor, y: torch.Tensor, sigma: float):
    if sigma <= 0 or x.shape[-1] < 2:
        return x, y

    batch_size, seq_len, num_channels = x.shape
    horizon = y.shape[1]
    perm = torch.randint(
        low=0,
        high=num_channels,
        size=(batch_size, num_channels),
        device=x.device,
    ).unsqueeze(-2)
    lam = torch.normal(
        mean=0.0,
        std=sigma,
        size=(batch_size, 1, num_channels),
        device=x.device,
    )
    x_perm = x.gather(-1, perm.repeat(1, seq_len, 1))
    y_perm = y.gather(-1, perm.repeat(1, horizon, 1))
    return x + lam * x_perm, y + lam * y_perm


def kl_loss(stables: torch.Tensor, sample_num: int) -> torch.Tensor:
    if sample_num > 0:
        shuffle = torch.randint(
            low=0,
            high=stables.shape[0],
            size=(stables.shape[0],),
            device=stables.device,
        )
        shuffle = (
            shuffle.unsqueeze(-1)
            .unsqueeze(-1)[:sample_num]
            .repeat(1, stables.shape[1], stables.shape[2])
        )
        stables = torch.gather(stables, dim=0, index=shuffle)
    probs = stables.softmax(dim=-1)
    log_probs = torch.log(probs + 1e-8)
    p_i = probs.unsqueeze(2)
    log_p_i = log_probs.unsqueeze(2)
    log_q_j = log_probs.unsqueeze(1)
    return (p_i * (log_p_i - log_q_j)).sum(dim=-1).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_indices: Sequence[int],
    args,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.L1Loss()
    mse = nn.MSELoss()
    losses = []
    target_indices_tensor = torch.as_tensor(target_indices, dtype=torch.long, device=device)

    for x, y, _ in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        x, y = channel_mixup(x, y, args.sigma)

        optimizer.zero_grad()
        output, stables = model(x)
        pred = output.index_select(-1, target_indices_tensor)
        target = y.index_select(-1, target_indices_tensor)
        loss = criterion(pred, target)

        if args.r_dropout > 0 or args.kl > 0:
            output_r, stables_r = model(x)
            pred_r = output_r.index_select(-1, target_indices_tensor)
            loss = loss + criterion(pred_r, target) / 2
            if args.r_dropout > 0:
                loss = loss + args.r_dropout * mse(pred, pred_r)
            if args.kl > 0:
                loss = loss + args.kl * (
                    kl_loss(stables, args.sample_num)
                    + kl_loss(stables_r, args.sample_num) / 2
                )

        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset: TimeSeriesDataset,
    scaler: StandardScaler,
    target_indices: Sequence[int],
    target_names: Sequence[str],
    device: torch.device,
    collect_predictions: bool = False,
):
    model.eval()
    target_indices_tensor = torch.as_tensor(target_indices, dtype=torch.long, device=device)
    preds, trues = [], []
    rows = []

    for x, y, indices in loader:
        x = x.float().to(device)
        output, _ = model(x)
        pred = output.index_select(-1, target_indices_tensor).cpu().numpy()
        true = y.index_select(-1, target_indices_tensor.cpu()).numpy()
        preds.append(pred)
        trues.append(true)

        if collect_predictions:
            for batch_pos, dataset_index in enumerate(indices.tolist()):
                dates = dataset.target_dates(dataset_index)
                for step, date in enumerate(dates):
                    row = {
                        "sample": int(dataset_index),
                        "step": step + 1,
                        "date": pd.Timestamp(date).isoformat(),
                    }
                    for col_pos, name in enumerate(target_names):
                        row[f"pred_{name}"] = float(pred[batch_pos, step, col_pos])
                        row[f"true_{name}"] = float(true[batch_pos, step, col_pos])
                    rows.append(row)

    pred_all = np.concatenate(preds, axis=0)
    true_all = np.concatenate(trues, axis=0)
    mse = float(np.mean((pred_all - true_all) ** 2))
    mae = float(np.mean(np.abs(pred_all - true_all)))
    return {"mse": mse, "mae": mae}, rows


def build_model(args, n_features: int) -> DTAF:
    config = DTAFConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=n_features,
        d_model=args.d_model,
        e_layers=args.e_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=args.dropout,
        heads=args.heads,
        k=args.k,
        moving_avg=args.moving_avg,
        aggregated_norm=args.aggregated_norm,
        expert_num=args.expert_num,
        kan_div=args.kan_div,
    )
    return DTAF(config)


def save_checkpoint(
    path: str,
    model: nn.Module,
    model_config: DTAFConfig,
    scaler: StandardScaler,
    selected_columns: Sequence[str],
    target_indices: Sequence[int],
    target_names: Sequence[str],
    metrics: Dict[str, float],
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "scaler": scaler.state_dict(),
            "selected_columns": list(selected_columns),
            "target_indices": list(target_indices),
            "target_names": list(target_names),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_path(args) -> str:
    return args.checkpoint or os.path.join(args.output_dir, "dtaf_best.pt")


def train(args) -> None:
    print_args(args)
    set_seed(args.seed)
    device = choose_device(args)
    values, dates, selected_columns, target_indices, target_names = read_itransformer_csv(args)
    borders = split_borders(len(values), args)

    scaler = StandardScaler()
    scaler.fit(values[borders["train"][0] : borders["train"][1]])
    datasets, loaders = make_loaders(values, dates, scaler, args)

    model = build_model(args, len(selected_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(args.patience)

    print_section("run summary")
    print(f"device: {device}", flush=True)
    print(f"features: {args.features} | columns: {len(selected_columns)} | targets: {target_names}", flush=True)
    print(f"windows: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}", flush=True)
    print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    print_section("training")
    best_metrics = None
    best_epoch = None
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, target_indices, args, device
        )
        val_metrics, _ = evaluate(
            model,
            loaders["val"],
            datasets["val"],
            scaler,
            target_indices,
            target_names,
            device,
        )
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.6f} | "
            f"val_{format_metrics(val_metrics)}",
            flush=True,
        )
        improved = stopper.best is None or val_metrics["mae"] < stopper.best - stopper.min_delta
        if stopper.step(val_metrics["mae"], model):
            print(f"early stopping at epoch {epoch}", flush=True)
            break
        if improved:
            best_metrics = val_metrics
            best_epoch = epoch
        adjust_learning_rate(optimizer, epoch, args)

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    if best_metrics is not None:
        print_section("best validation")
        print(f"epoch {best_epoch:03d} | val_{format_metrics(best_metrics)}", flush=True)

    print_section("test")
    test_metrics, rows = evaluate(
        model,
        loaders["test"],
        datasets["test"],
        scaler,
        target_indices,
        target_names,
        device,
        collect_predictions=args.save_predictions,
    )
    print(f"test_{format_metrics(test_metrics)}", flush=True)

    path = checkpoint_path(args)
    save_checkpoint(
        path,
        model,
        model.config,
        scaler,
        selected_columns,
        target_indices,
        target_names,
        {"best_epoch": best_epoch, "val": best_metrics, "test": test_metrics},
    )
    print(f"checkpoint saved: {path}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "val": best_metrics, "test": test_metrics}, f, indent=2)
    print(f"metrics saved: {metrics_path}", flush=True)

    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        print(f"test predictions saved: {pred_path}", flush=True)
    print_separator()


def test(args) -> None:
    device = choose_device(args)
    ckpt = load_checkpoint(checkpoint_path(args), device)
    args.seq_len = ckpt["model_config"]["seq_len"]
    args.pred_len = ckpt["model_config"]["pred_len"]
    print_args(args)
    print_section("checkpoint")
    print(f"checkpoint: {checkpoint_path(args)}", flush=True)
    print(f"checkpoint_metrics: {ckpt.get('metrics')}", flush=True)
    values, dates, selected_columns, target_indices, target_names = read_itransformer_csv(
        args, ckpt["selected_columns"], ckpt["target_names"]
    )
    scaler = StandardScaler.from_state_dict(ckpt["scaler"])
    datasets, loaders = make_loaders(values, dates, scaler, args, shuffle_train=False)

    model = DTAF(DTAFConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    print_section("test")
    print(f"device: {device}", flush=True)
    print(f"features: {args.features} | columns: {len(selected_columns)} | targets: {target_names}", flush=True)
    print(f"windows: test={len(datasets['test'])}", flush=True)
    metrics, rows = evaluate(
        model,
        loaders["test"],
        datasets["test"],
        scaler,
        target_indices,
        target_names,
        device,
        collect_predictions=args.save_predictions,
    )
    print(f"test_{format_metrics(metrics)}", flush=True)
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        print(f"test predictions saved: {pred_path}", flush=True)
    print_separator()


def pandas_freq(freq: str) -> str:
    return {
        "t": "min",
        "h": "h",
        "s": "s",
        "d": "D",
        "b": "B",
        "w": "W",
        "m": "M",
    }.get(freq.lower(), freq)


@torch.no_grad()
def predict(args) -> None:
    device = choose_device(args)
    ckpt = load_checkpoint(checkpoint_path(args), device)
    args.seq_len = ckpt["model_config"]["seq_len"]
    args.pred_len = ckpt["model_config"]["pred_len"]
    print_args(args)
    print_section("checkpoint")
    print(f"checkpoint: {checkpoint_path(args)}", flush=True)
    print(f"checkpoint_metrics: {ckpt.get('metrics')}", flush=True)
    values, dates, selected_columns, target_indices, target_names = read_itransformer_csv(
        args, ckpt["selected_columns"], ckpt["target_names"]
    )
    if len(values) < args.seq_len:
        raise ValueError("Dataset is shorter than seq_len.")

    scaler = StandardScaler.from_state_dict(ckpt["scaler"])
    model = DTAF(DTAFConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = scaler.transform(values).astype(np.float32)[-args.seq_len :]
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    output, _ = model(x)
    pred = output[:, :, target_indices].cpu().numpy()[0]
    pred = scaler.inverse_target(pred, target_indices)

    future_dates = pd.date_range(
        start=pd.Timestamp(dates[-1]),
        periods=args.pred_len + 1,
        freq=pandas_freq(args.freq),
    )[1:]
    rows = {"date": future_dates}
    for col_pos, name in enumerate(target_names):
        rows[name] = pred[:, col_pos]

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "future_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print_section("predict")
    print(f"device: {device}", flush=True)
    print(f"future_predictions saved: {pred_path}", flush=True)
    print_separator()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and run DTAF on iTransformer-format CSV data.")

    parser.add_argument("--mode", choices=["train", "test", "predict"], default="train")
    parser.add_argument("--is_training", type=int, default=None)
    parser.add_argument("--root_path", type=str, default="./dataset")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", "--lr", dest="lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--lradj", type=str, default="none", choices=["none", "type1", "type2"])

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--heads", "--n_heads", dest="heads", type=int, default=2)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--aggregated_norm", type=int, default=1)
    parser.add_argument("--expert_num", type=int, default=2)
    parser.add_argument("--kan_div", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--r_dropout", type=float, default=1e-4)
    parser.add_argument("--kl", type=float, default=0.0)
    parser.add_argument("--sample_num", type=int, default=0)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split", type=str, default="ratio", choices=["ratio", "ett"])
    parser.add_argument("--drop_last", action="store_true")

    parser.add_argument("--checkpoints", "--output_dir", dest="output_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_mps", action="store_true")

    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--model", type=str, default="DTAF")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--des", type=str, default="")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--dec_in", type=int, default=None)
    parser.add_argument("--c_out", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--embed", type=str, default=None)
    parser.add_argument("--distil", action="store_true")
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--inverse", action="store_true")

    args = parser.parse_args()
    if args.is_training is not None:
        args.mode = "train" if args.is_training else "test"
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
