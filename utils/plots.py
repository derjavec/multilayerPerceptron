import matplotlib.pyplot as plt


def loss_plot(loss_list, val_loss_list):
    """
    Plot multiple training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))

    for idx, (train_loss, val_loss) in enumerate(zip(loss_list, val_loss_list)):
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, marker='o', linewidth=2,
                 label=f'Training Loss Exp {idx+1}')
        plt.plot(epochs, val_loss, marker='o', linewidth=2,
                 label=f'Validation Loss Exp {idx+1}')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs. Validation Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def acc_plot(acc_list, val_acc_list) -> None:
    """
    Plot multiple training and validation accuracy curves.

    Parameters
    ----------
    acc_list : list of lists
        Each sublist contains training accuracy per epoch for an experiment.
    val_acc_list : list of lists
        Each sublist contains validation accuracy per epoch for an experiment.
    """
    plt.figure(figsize=(10, 6))

    for idx, (train_acc, val_acc) in enumerate(zip(acc_list, val_acc_list)):
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, marker='o', linewidth=2,
                 label=f'Training Accuracy Exp {idx+1}')
        plt.plot(epochs, val_acc, marker='o', linewidth=2,
                 label=f'Validation Accuracy Exp {idx+1}')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training vs. Validation Accuracy", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def f1_plot(f1_list, val_f1_list) -> None:
    """
    Plot multiple training and validation F1-score curves.

    Parameters
    ----------
    f1_list : list of lists
        Each sublist contains training F1-score per epoch for an experiment.
    val_f1_list : list of lists
        Each sublist contains validation F1-score per epoch for an experiment.
    """
    plt.figure(figsize=(10, 6))

    for idx, (train_f1, val_f1) in enumerate(zip(f1_list, val_f1_list)):
        epochs = range(1, len(train_f1) + 1)
        plt.plot(epochs, train_f1, marker='o', linewidth=2,
                 label=f'Training F1 Exp {idx+1}')
        plt.plot(epochs, val_f1, marker='o', linewidth=2,
                 label=f'Validation F1 Exp {idx+1}')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    plt.title("Training vs. Validation F1-score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

