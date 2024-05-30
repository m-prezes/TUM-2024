import torch
from tqdm import tqdm


def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(test_loader), correct / total


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    max_num_epochs,
    early_stopping_patience,
    device,
):
    train_losses = []
    test_losses = []
    test_accuracies = []

    early_stopping = 0
    best_test_accuracy = 0
    for epoch in range(max_num_epochs):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{max_num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if test_accuracy <= best_test_accuracy:
            early_stopping += 1
            if early_stopping == early_stopping_patience:
                print("Early stopping")
                break
        else:
            best_test_accuracy = test_accuracy
            early_stopping = 0

    return train_losses, test_losses, test_accuracies