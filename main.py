import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

script_dir = Path(__file__).resolve().parent
model_path = script_dir / "cifar_10_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_filters=32, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        fc1_neuron_num = 256
        self.fc1 = nn.LazyLinear(fc1_neuron_num)
        self.fc2 = nn.Linear(fc1_neuron_num, 10)
        self.patience = 5
        self.loss_history = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def fit(self, train_loader, val_loader, epochs=10, lr=0.001):
        best_val_loss = float('inf')
        trigger_times = 0
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
            val_loss = self._compute_validation_loss(val_loader, loss_fn)
            self.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                torch.save(self.state_dict(), model_path)
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            epoch_loss /= len(train_loader.dataset)
            self.loss_history.append(epoch_loss)
            print(f"Epoch {epoch}, Train Loss {epoch_loss:.4f}, Val Loss {val_loss:.4f}")
        # Load best state model if early stop happens
        self.load_state_dict(torch.load(model_path))

    def _compute_validation_loss(self, val_loader, loss_fn):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
        return total_loss / len(val_loader.dataset)

    def predict(self, test_loader):
        all_predictions = []
        self.eval()
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = self(batch_X)
                preds = torch.argmax(outputs, dim=1)
                all_predictions.append(preds)
        predictions = torch.cat(all_predictions)
        return predictions


# Standard CIFAR-10 normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])

def _load_cifar_training_data(batch_size=128):
    cifar_train = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    val_size = int(0.1 * (len(cifar_train)))
    train_size = len(cifar_train) - val_size
    train_dataset, val_dataset = random_split(cifar_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def _load_cifar_testing_data(batch_size=128):
    cifar_test = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
    return test_loader

def _visualize(model, test_loader, model_exists, num_samples=16):
    save_dir = Path(__file__).parent

    if not model_exists:
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(model.loss_history)), model.loss_history, marker='o')
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(save_dir / "cifar_loss_curve.png")
        plt.close()

    #sample predictions
    model.eval()
    X_sample, y_sample = next(iter(test_loader))  # get first batch
    X_sample = X_sample[:num_samples].to(device)
    y_sample = y_sample[:num_samples].to(device)
    with torch.no_grad():
        outputs = model(X_sample)
        preds = torch.argmax(outputs, dim=1)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        img = X_sample[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        # Unnormalize the image for better visualization
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = img.clip(0, 1)
        plt.imshow(img)
        plt.title(f"P: {class_names[preds[i]]}\nT: {class_names[y_sample[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / "cifar_samples.png")
    plt.close()


def main():
    print(f"Using device: {device}")
    model = CNN().to(device)
    model_exists = model_path.exists()
    if model_exists:
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        start_time = time.time()
        torch.manual_seed(20)
        train_loader, val_loader = _load_cifar_training_data()
        model.fit(train_loader, val_loader)
        end_time = time.time()
        print(f"Model trained and saved in: {end_time - start_time:.6f} seconds")

    test_loader = _load_cifar_testing_data()
    predictions = model.predict(test_loader)

    y_test = torch.cat([batch_y for _, batch_y in _load_cifar_testing_data()]).to(device)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    _visualize(model, test_loader, model_exists)

if __name__ == '__main__':
    main()
