import torch
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader


class FlyBird(torchvision.datasets.VisionDataset):
    """
    A dataset for fly and bird.
     使用torch 自带的cifar10数据集
    """

    classes = ['airplane', 'bird']

    def __init__(self, root='./data', train=False, transform=None, target_transform=None):
        super(FlyBird, self).__init__(root, transform=transform, target_transform=target_transform)

        if train:
            self._cifar10 = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        else:
            self._cifar10 = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

        self.classes = FlyBird.classes

        self.class_to_idx = {classname: label for label, classname in enumerate(self.classes)}

        self._label_map = {label: self.class_to_idx[classname] for classname, label in self._cifar10.class_to_idx.items() if classname in self.classes}
        # old label -> new label

        self._need_labels = set(self._label_map.keys())
        # needed labels for fly and bird

        self._data = {}  # index ->  cifar10 index

        # 遍历出来所有的 fly 和 bird 索引
        index = 0
        for cifar10_index, (img, label) in enumerate(self._cifar10):
            if label in self._need_labels:
                self._data[index] = cifar10_index
                index += 1

        print(f'dataset buid success, {len(self._data)} images.')

    def __getitem__(self, index):

        if index >= len(self._data):
            raise IndexError

        real_index = self._data[index]
        img, label = self._cifar10[real_index]
        label = self._label_map[label]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self._data)

    @staticmethod
    def one_hot(label):
        """
        将label 转换成 one-hot
        :param label: label
        :return: one-hot
        """
        one_hot_label = torch.zeros(len(FlyBird.classes))
        one_hot_label[label] = 1
        return one_hot_label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])

train_loader = DataLoader(
    FlyBird(
        train=True,
        transform=transform,  # 对图片进行处理
        # target_transform=FlyBird.one_hot  # 对标签进行处理 # CrossEntropyLoss() 不需要one-hot
    ),
    shuffle=True,
    batch_size=16,
    # num_workers=4, # 多线程读取数据
    pin_memory=True,
)

test_loader = DataLoader(
    FlyBird(
        train=False,
        transform=transform,
        # target_transform=FlyBird.one_hot # CrossEntropyLoss() 不需要one-hot
    ),
    shuffle=False,
    batch_size=16,
    # num_workers=4,
    pin_memory=True,

)

# if __name__ == '__main__':
#     pass
#     # 测试
#     for i in enumerate(train_loader):
#         print(i)
#         break
