import torch


def evaluate_model(model, data_loader, criterion, device: str) -> tuple[int, float]:
    # TODO: calculate average loss and accuracy on the test dataset
    _correct = 0
    _total = 0
    avg_loss = 0
    with torch.no_grad():
        for _data in data_loader:
            _images, _labels = _data[0].to(device), _data[1].to(device)
            model.to(device)
            _outputs = model(_images)
            _, _predicted = torch.max(_outputs.data, 1)
            _total += _labels.size(0)
            _correct += (_predicted == _labels).sum().item()
            avg_loss += criterion(_outputs, _labels) / _labels.size(0)

    accuracy = 100 * _correct / _total
    print(f"Accuracy of the network on the 10000 test images: {accuracy}%")
    return avg_loss, accuracy
