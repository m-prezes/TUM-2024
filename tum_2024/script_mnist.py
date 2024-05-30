import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Model
from train import train
from utils import save_results, set_seed

RESULTS_DIR = "results"



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


mnist_train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

mnist_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


set_seed(42)
mnist_train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=mnist_train_transform
)
mnist_train_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=64, shuffle=True
)

mnist_test = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=mnist_test_transform
)
mnist_test_loader = torch.utils.data.DataLoader(
    mnist_test, batch_size=64, shuffle=False
)

INPUT_SIZE = mnist_train[0][0].shape[1] * mnist_train[0][0].shape[2]
HIDDEN_SIZES = [1024, 1024, 1024]
OUTPUT_SIZE = len(mnist_train.classes)
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam
LR = 0.0005

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


for dropout_rate in DROPOUT_RATES:
    set_seed(42)
    model = Model(dropout_rate, INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    train_losses, test_losses, test_accuracies = train(
        model,
        mnist_train_loader,
        mnist_test_loader,
        CRITERION,
        optimizer,
        MAX_EPOCHS,
        EARLY_STOPPING_PATIENCE,
        DEVICE,
    )
    save_results(
        model,
        train_losses,
        test_losses,
        test_accuracies,
        f"{RESULTS_DIR}/mnist/dropout_{dropout_rate}",
    )