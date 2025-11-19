import matplotlib.pyplot as plt


def loss_plot(loss, val_loss):
    """
    Plot training and validation loss per epoch.
    """
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, marker='o', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, marker='o', linewidth=2,
             label='Validation Loss')

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs. Validation Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def acc_plot(acc, val_acc):
    """
    Plot training and validation accuracy per epoch.
    """
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, marker='o', linewidth=2, label='Training Accuracy')
    plt.plot(
        epochs,
        val_acc,
        marker='o',
        linewidth=2,
        label='Validation Accuracy'
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training vs. Validation Accuracy", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
