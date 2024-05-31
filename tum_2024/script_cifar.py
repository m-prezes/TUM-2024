import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Model
from train import train
from utils import save_results, set_seed

RESULTS_DIR = "results"


cifar_train_transform = transforms.Compose(
    [

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

cifar_test_transform = transforms.Compose(
    [

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

set_seed(42)
cifar_train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=cifar_train_transform
)
cifar_train_loader = torch.utils.data.DataLoader(
    cifar_train, batch_size=64, shuffle=True
)

cifar_test = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=cifar_test_transform
)
cifar_test_loader = torch.utils.data.DataLoader(
    cifar_test, batch_size=64, shuffle=False
)

INPUT_SIZE = 128 * 4 * 4
OUTPUT_SIZE = len(cifar_train.classes)
DROPOUT_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]

CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam
LR = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10


for dropout_rate in DROPOUT_RATES:
    set_seed(42)
    model = Model( dropout_rate=dropout_rate, input_channels=3,output_size=OUTPUT_SIZE).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    train_losses, test_losses, test_accuracies = train(
        model,
        cifar_train_loader,
        cifar_test_loader,
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
        f'{RESULTS_DIR}/cifar/dropout_{dropout_rate}',
    )