import numpy as np
import torch
import warnings
from sklearn.datasets import fetch_openml
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues
import matplotlib.pyplot as plt


def plot_examples(id_iter, nrows=1, ncols=1):
    for count, id in enumerate(id_iter):
        plt.subplot(nrows, ncols, count + 1)
        plt.imshow(X[id].reshape(28, 28), cmap="gray")
        plt.title(f"id: {id} \n label: {y[id]}")
        plt.axis("off")

    plt.tight_layout(h_pad=2.0)

class ClassifierModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.out(X)
        return X

if __name__ == "__main__":
    SEED = 123
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    warnings.filterwarnings("ignore", "Lazy modules are a new feature.*")


    mnist = fetch_openml("mnist_784")  # Fetch the MNIST dataset

    X = mnist.data.astype("float32").to_numpy() # 2D array (images are flattened into 1D)
    X /= 255.0  # Scale the features to the [0, 1] range
    X = X.reshape(len(X), 1, 28, 28)  # reshape into [N, C, H, W] for PyTorch

    y = mnist.target.astype("int64").to_numpy()  # 1D array of labels

    model_skorch = NeuralNetClassifier(ClassifierModule)

    num_crossval_folds = 3  # for efficiency; values like 5 or 10 will generally work better
    pred_probs = cross_val_predict(
        model_skorch,
        X,
        y,
        cv=num_crossval_folds,
        method="predict_proba",
    )

    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(y, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc}")

    ranked_label_issues = find_label_issues(
        y,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")

    plot_examples(ranked_label_issues[range(15)], 3, 5)
    plt.show()
