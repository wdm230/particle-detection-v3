import json
import matplotlib.pyplot as plt

def plot_history(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5), dpi=500)

    plt.subplot(1,1,1)
    plt.plot(epochs, history['loss'], label='Loss', linewidth=5)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Training Loss', fontsize=20)

    plt.tight_layout()
    plt.savefig("data/training_history/loss.png")


# Example usage
history_file = 'data/training_history/history.json'
plot_history(history_file)
