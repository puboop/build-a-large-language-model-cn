# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/4/2 10:36
# @Author     : white
# @FileName   : 简易训练.py
# @Software   : PyCharm
# **************************************
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        用于检索单个数据记录及其对应标签的指令
        :param index:
        :return:
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]  # 用于返回数据集总长度的指令


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # 将输入和输出的数量编码为变量很有用，这样可以为具有不同特征和类别数量的数据集重用相同的代码。
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),  # Linear 层将输入和输出节点的数量作为参数。
            torch.nn.ReLU(),  # 非线性激活函数放置在隐藏层之间。

            # 2nd hidden layer
            torch.nn.Linear(30, 20),  # 一个隐藏层的输出节点数必须与下一个隐藏层的输入节点数相匹配。
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits  # 最后一层的输出被称为 logits。


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)  # 上一节的数据集包含 2 个特征和 2 个类别
"""
随机梯度下降（SGD）优化器
学习率（lr）设置为 0.5
"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # 我们让优化器知道需要优化哪些参数

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()  # 将上一轮的梯度设置为零，以防止意外的梯度累积
        loss.backward()  # 计算损失函数相对于模型参数的梯度
        optimizer.step()  # 优化器使用梯度来更新模型参数

        ### LOGGING
        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    model.eval()
# 模型保存
torch.save(model.state_dict(), "model.pth")


def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions  # 这会返回一个由 True/False 值组成的张量，取决于标签是否匹配
        correct += torch.sum(compare)  # sum 操作会计算 True 值的数量
        total_examples += len(compare)

    return (correct / total_examples).item()  # 这是正确预测的比例，一个介于 0 和 1 之间的值。并且 .item() 返回张量的值作为 Python 浮点数。


print(compute_accuracy(model, train_loader))
# 模型加载
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))
