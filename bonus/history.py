import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

import numpy as np


def loss_acc_f1(y_true: np.ndarray,
                y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Compute cross-entropy loss, accuracy and macro F1-score.
    """
    loss = -np.mean(
        np.sum(y_true * np.log(y_pred + 1e-8), axis=1)
    )

    pred_class = np.argmax(y_pred, axis=1)
    true_class = np.argmax(y_true, axis=1)
    acc = np.mean(pred_class == true_class)

    num_classes = y_true.shape[1]
    f1_scores: List[float] = []

    for cls in range(num_classes):
        tp = np.sum((pred_class == cls) & (true_class == cls))
        fp = np.sum((pred_class == cls) & (true_class != cls))
        fn = np.sum((pred_class != cls) & (true_class == cls))

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        f1_scores.append(f1)

    f1 = float(np.mean(f1_scores))

    return float(loss), float(acc), f1


def same_config(config1: Dict[str, Any],
                config2: Dict[str, Any]) -> bool:
    """
    Compare two model configurations.

    Only compares selected relevant keys.
    """
    keys_to_compare = [
        "layer",
        "activations",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer"
    ]

    for key in keys_to_compare:
        if config1.get(key) != config2.get(key):
            return False

    return True


def append_history(history: Dict[str, Any],
                   config: Dict[str, Any],
                   filename: str = "histories.json") -> None:
    """
    Append training history to a JSON file if config is not duplicated.
    """
    history["timestamp"] = datetime.now().isoformat()
    history["config"] = config

    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([history], file, indent=4)
        return

    with open(filename, "r", encoding="utf-8") as file:
        all_histories = json.load(file)

    for item in all_histories:
        if "config" in item and same_config(item["config"], config):
            return

    all_histories.append(history)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(all_histories, file, indent=4)


def build_history(history: Optional[Dict[str, Any]],
                  epoch: int,
                  y_train: np.ndarray,
                  y_val: np.ndarray,
                  y_pred: np.ndarray,
                  y_val_pred: np.ndarray,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update or initialize the training history.
    """
    if history is None:
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "best": {
                "epoch": 0,
                "val_loss": float("inf"),
                "val_acc": 0.0,
                "val_f1": 0.0
            }
        }

    train_loss, train_acc, train_f1 = loss_acc_f1(
        y_train, y_pred
    )

    val_loss, val_acc, val_f1 = loss_acc_f1(
        y_val, y_val_pred
    )

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch}/{config['epochs']} | "
        f"loss: {train_loss:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"acc: {train_acc:.4f} | "
        f"val_acc: {val_acc:.4f} | "
        f"f1: {train_f1:.4f} | "
        f"val_f1: {val_f1:.4f}"
    )

    if val_loss < history["best"]["val_loss"]:
        history["best"] = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        }

    if epoch == config["epochs"]:
        append_history(history, config)

    return history


def load_history(filename: str = "histories.json") -> List[Dict[str, Any]]:
    """
    Load all training histories from file.
    """
    if not os.path.exists(filename):
        return []

    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """
    Test loading histories.
    """
    histories = load_history()
    print(histories)


if __name__ == "__main__":
    main()
