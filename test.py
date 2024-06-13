import torch

from data import test_loader
from device import move_device
from model import model


def test(model, test_loader, weight_path):
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    total, correct = 0, 0
    for data in test_loader:
        x, y = move_device(data)
        output = model(x)
        _, predicted = torch.max(output, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print(f'Acurracy : {accuracy}%')


if __name__ == '__main__':
    test(model, test_loader, weight_path='./data/model_best.pth')
