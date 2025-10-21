import torch
import torch.nn as nn
import torch.optim as optim
from app.lib.data_loader import get_data_loader
from app.lib.evaluator import evaluate_model
from app.lib.model import CNN
from app.lib.trainer import train_model


if __name__ == "__main__":
    train_loader = get_data_loader('data/train', batch_size=64, train=True)
    test_loader = get_data_loader('data/test', batch_size=64, train=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, train_loader, criterion, optimizer, device='cpu', epochs=30)
    evaluate_model(trained_model, test_loader, criterion, device='cpu')

    torch.save(trained_model.state_dict(), "./app/model/CNN.pth")
