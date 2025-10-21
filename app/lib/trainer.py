import torch


def train_model(model, data_loader, criterion, optimizer, device: str, epochs: int):
    """
    :param criterion: Loss function normally.
    :param device: CPU or CUDA
    """
    # TODO: run several iterations of the training loop (based on epochs parameter) and return the model

    correct = 0
    total = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for ind, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ind % 100 == 99:
                print(f"[{epoch + 1}, {ind + 1}] accuracy: {100 * correct / total}%, loss: {running_loss / 100}")
                running_loss = 0.0

    print("Finished Training!")
    return model
