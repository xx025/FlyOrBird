import os

import torch

from data import train_loader, test_loader
from device import move_device
from model import model, optimizer, criterion


def train(model, train_data, test_data, optimizer, criterion, max_epoch=10, save_dir="./data"):
    record = {
        'train': {},
        'test': {}
    }

    for epoch in range(max_epoch):
        print(f'Epoch: {epoch}')
        for data in train_data:
            x, y = move_device(data)  # Move data to device
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        else:
            torch.save(model.state_dict(), f'{save_dir}/model_{epoch}.pth')  # 存储模型
            with torch.no_grad():
                for part_name, part_data in {'train': train_data, 'test': test_data}.items():
                    total = 0
                    correct = 0
                    for data in part_data:
                        x, y = move_device(data)
                        output = model(x)
                        _, predicted = torch.max(output, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                    accuracy = 100 * correct / total
                    record[part_name][epoch] = accuracy
                    print(f'Acurracy of {part_name}: {accuracy}%')
    else:
        # 找到测试集中准确率最高的epoch
        best_epoch = max(record['test'], key=record['test'].get)
        print(f'Best epoch: {best_epoch}')
        for epoch in range(max_epoch):
            if epoch != best_epoch:
                os.remove(f'{save_dir}/model_{epoch}.pth')

        # 如果存在model_best.pth，则删除
        if os.path.exists(f'{save_dir}/model_best.pth'):
            os.remove(f'{save_dir}/model_best.pth')
        os.rename(f'{save_dir}/model_{best_epoch}.pth', f'{save_dir}/model_best.pth')


if __name__ == '__main__':
    train(model, train_loader, test_loader, optimizer, criterion, max_epoch=20)
