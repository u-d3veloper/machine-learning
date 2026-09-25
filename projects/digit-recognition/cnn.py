"""PyTorch CNN for handwritten digit recognition on MNIST.

    python cnn.py train [--epochs 10]
    python cnn.py predict path/to/digit.png

The input image must be a white digit on a black background, like MNIST.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

CHECKPOINT = Path(__file__).parent / "checkpoints" / "digitnet.pth"
TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,)),  # MNIST mean and standard deviation
    ]
)


class DigitNet(nn.Module):
    """Two conv layers, max-pooling and dropout, then two fully connected layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Map images (batch, 1, 28, 28) to log-probabilities (batch, 10)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(F.max_pool2d(x, 2))
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return F.log_softmax(self.fc2(self.dropout2(x)), dim=1)


@torch.no_grad()
def accuracy(model, loader, device):
    """Fraction of correctly classified examples in a loader."""
    model.eval()
    correct = sum(
        (model(x.to(device)).argmax(dim=1) == y.to(device)).sum().item()
        for x, y in loader
    )
    return correct / len(loader.dataset)


def train(epochs=10, batch_size=64):
    """Train on 54k images, keep the best epoch on a 6k validation split.

    The test set is only used once, at the end, on the best checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = datasets.MNIST("data", train=True, download=True, transform=TRANSFORM)
    test = datasets.MNIST("data", train=False, download=True, transform=TRANSFORM)
    train_set, val_set = random_split(
        full, [54_000, 6_000], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size)
    test_loader = DataLoader(test, batch_size)

    model = DigitNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            F.nll_loss(model(x.to(device)), y.to(device)).backward()
            optimizer.step()
        val_accuracy = accuracy(model, val_loader, device)
        print(f"epoch {epoch}/{epochs}  validation accuracy {val_accuracy:.4f}")
        if val_accuracy > best:
            best = val_accuracy
            CHECKPOINT.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"test accuracy {accuracy(model, test_loader, device):.4f}")


def predict(image_path, model):
    """Return ``(digit, confidence)`` for a white-on-black digit image."""
    image = Image.open(image_path).convert("L").resize((28, 28))
    with torch.no_grad():
        probabilities = model.eval()(TRANSFORM(image).unsqueeze(0)).exp()[0]
    digit = probabilities.argmax().item()
    return digit, probabilities[digit].item()


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN: train or predict.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train").add_argument("--epochs", type=int, default=10)
    sub.add_parser("predict").add_argument("image")
    args = parser.parse_args()

    if args.command == "train":
        train(args.epochs)
    else:
        model = DigitNet()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        digit, confidence = predict(args.image, model)
        print(f"{digit} ({confidence:.1%})")


if __name__ == "__main__":
    main()
