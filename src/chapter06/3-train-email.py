# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/4/4 12:47
# @Author     : white
# @FileName   : 3-trian-email.py
# @Software   : PyCharm
# **************************************
import time
import tiktoken
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.chapter05.gpt import GPTModel, load_weights_into_gpt
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter06.tools import classify_review, SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset("train.csv", tokenizer, None)
val_dataset = SpamDataset("validation.csv", tokenizer, train_dataset.max_length)
test_dataset = SpamDataset("test.csv", tokenizer, train_dataset.max_length)

# 此设置可确保与大多数计算机兼容
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,  # 数据集
    batch_size=batch_size,  # 批次大小
    shuffle=True,  # 打乱数据
    num_workers=num_workers,  # 多进程加载
    drop_last=True  # 丢弃最后不完整批次
)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
# 模型参数名加载
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size"    : 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate"     : 0.0,  # 丢弃率（Dropout 比例）
    "qkv_bias"      : True  # 查询-键-值偏置（是否启用）
}
model_configs = {
    "gpt2-small (124M)" : {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)" : {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)"   : {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
# 确保数据集的最大文本长度，不会超过模型能接受的最大上下文长度
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
# 模型定义
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)  # 将tenerflower的权重信息加载到torch中来
model.eval()  # 在推理模式下禁用dropout
# 为了让模型准备好进行分类微调，我们首先通过将所有层设为不可训练来冻结模型
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)

# 设置分类目标为2两类
num_classes = 2

# 替换掉输出层（model.out_head），该层原本将层输入映射到 50,257 维空间（即词汇表大小）
# 这个新的输出层 model.out_head 的 requires_grad 属性默认为 True，意味着它是模型训练过程中唯一会被更新的层
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # 替换最后一层为分类头

# 为了让最终的 LayerNorm 和最后一个 Transformer 模块参与训练（如图 6.10 所示），我们将它们的 requires_grad 设置为 True
# 这样做的目的在于获取更好的模型性能
for param in model.trf_blocks[-1].parameters():  # 训练最后一个Transformer层
    param.requires_grad = True
for param in model.final_norm.parameters():  # 训练最后一个归一化层
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)  # shape: (batch_size, num_tokens, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CountLoss:
    """
    不同的损失值计算方法
    """

    @classmethod
    def calc_accuracy_loader(cls, data_loader, model, device, num_batches=None):
        """分类准确率评估，直接评估模型答对了多少，答错了多少
        估算多个数据集上的分类准确率，为提高效率，这里基于 10 个批次的结果进行估算
        :param data_loader: 数据加载器（批量喂数据）
        :param model: 训练好的模型
        :param device: CPU / GPU
        :param num_batches: 只计算前N个batch，加速评估
        :return: 准确率 0~1
        """
        # 1. 切换模型为【评估模式】，关闭dropout、batchnorm等训练专用层
        model.eval()
        # 2. 初始化：正确预测数 + 总样本数
        correct_predictions, num_examples = 0, 0
        # 3. 确定要计算多少个batch（默认全部，否则取最小值）
        num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        # 4. 遍历数据
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # 如果超过设定的批次，直接停止（加速用）
            if i > num_batches: break
            # 把数据搬到设备上（GPU/CPU）
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # 5. 关闭梯度计算（推理时不需要，省显存、提速）
            with torch.no_grad():
                # 模型前向传播，取【最后一个token】的输出logits
                # 这是序列分类/语言模型分类的典型写法！
                logits = model(input_batch)[:, -1, :]

            # ====================== 核心：argmax 登场 ======================
            # logits 形状：[batch_size, num_classes]
            # dim=-1 = 最后一维 = 按每个样本取【概率最大的类别】
            predicted_labels = torch.argmax(logits, dim=-1)
            # 6. 统计总样本数
            num_examples += predicted_labels.shape[0]
            # 7. 统计预测正确的数量
            correct_predictions += (predicted_labels == target_batch).sum().item()
        # 8. 准确率 = 正确数 / 总数
        return correct_predictions / num_examples

    @classmethod
    def count_loss_accuracy(cls):
        model.to(device)
        # 分别计算 训练集/验证集/测试集 的准确率
        # 只算前10个batch，快速估算，不用跑完整数据集
        train_accuracy = cls.calc_accuracy_loader(train_loader, model, device, num_batches=10)
        val_accuracy = cls.calc_accuracy_loader(val_loader, model, device, num_batches=10)
        test_accuracy = cls.calc_accuracy_loader(test_loader, model, device, num_batches=10)
        print("==============少量数据集评估模型==============")
        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")
        print()

    @classmethod
    def calc_loss_batch(cls, input_batch, target_batch, model, device):
        """
        交叉损失计算 用于二分类的专用损失函数
        :param input_batch:
        :param target_batch:
        :param model:
        :param device:
        :return:
        """
        model.to(device)
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        # 获取输出的最后一个token，
        logits = model(input_batch)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    @classmethod
    def calc_loss_loader(cls, data_loader, model, device, num_batches=None):
        """交叉熵损失 衡量模型预测分布与真实标签的差异，是训练过程的核心监控指标
        循环算多个批次，求平均损失
        :param data_loader:
        :param model:
        :param device:
        :param num_batches:
        :return:
        """
        total_loss = 0.
        # 空数据判断
        if len(data_loader) == 0: return float("nan")
        # 取要计算的批次数量（默认全部，最多不超过真实批次）
        num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        # 循环计算 N 个批次
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i > num_batches: break
            loss = cls.calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # 把张量转成普通数字累加
        # 返回平均损失
        return total_loss / num_batches

    @classmethod
    def count_loss_cross(cls):
        """
        计算交叉熵损失
        :return:
        """
        # Similar to calculating the training accuracy, we now compute the initial loss for each data set:
        with torch.no_grad():  # 关闭梯度追踪以提高效率，因为当前未进行训练
            train_loss = cls.calc_loss_loader(train_loader, model, device, 5)
            val_loss = cls.calc_loss_loader(val_loader, model, device, 5)
            test_loss = cls.calc_loss_loader(test_loader, model, device, 5)
        print(f"Training loss:    {train_loss:.3f}")
        print(f"Validation loss:  {val_loss:.3f}")
        print(f"Test loss:        {test_loss:.3f}")
        print()


class TrianClassifier:
    @classmethod
    def plot_values(cls, epochs_seen, examples_seen, train_values, val_values, label="loss"):
        # ✅ 核心修复：自动把所有 tensor 转成 Python 列表（零 numpy 依赖）
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()  # 支持 GPU / 计算图张量
            return x

        epochs_seen = to_list(epochs_seen)
        examples_seen = to_list(examples_seen)
        train_values = to_list(train_values)
        val_values = to_list(val_values)

        # 绘图逻辑不变
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.plot(examples_seen, train_values, alpha=0)
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()
        plt.savefig(f"{label}-plot.pdf")
        plt.show()

    @classmethod
    def train_classifier_simple(cls, model, train_loader, val_loader, optimizer,
                                device, num_epochs, eval_freq, eval_iter, tokenizer):
        """
        文本分类模型的完整训练与评估函数（核心训练流程）
        :param model: 待训练的神经网络模型（Transformer/LLM分类模型）
        :param train_loader: 训练集数据加载器，批量加载训练数据
        :param val_loader: 验证集数据加载器，用于监控模型泛化能力
        :param optimizer: 优化器（如AdamW），负责更新模型参数
        :param device: 运行设备（CPU/GPU）
        :param num_epochs: 训练总轮数（完整遍历训练集的次数）
        :param eval_freq: 评估频率，每训练eval_freq步就评估一次损失
        :param eval_iter: 评估时使用的批次数量，只计算前N批加速评估
        :param tokenizer: 文本分词器（本函数未使用，为预留接口）
        :return: 训练过程中的损失、准确率曲线，以及总样本数
        """
        # 初始化列表，存储训练过程中的指标，用于后续绘制曲线
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        # 记录模型总共学习过的样本数量 + 全局训练步数（每处理一个batch+1）
        examples_seen, global_step = 0, -1

        # ==================== 主训练循环：遍历每一轮 epoch ====================
        for epoch in range(num_epochs):
            model.train()  # 将模型设置为【训练模式】，启用Dropout/BatchNorm等训练层
            # 遍历训练集的每一个批次数据
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # 清空上一批次的梯度，防止梯度累积干扰
                # 计算当前批次的损失值（前向传播）
                loss = CountLoss.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # 反向传播：计算模型参数的梯度
                optimizer.step()  # 更新模型参数：根据梯度优化权重
                # 累计统计：模型已经学习过的样本总数
                examples_seen += input_batch.shape[0]
                global_step += 1  # 全局训练步数 +1

                # ==================== 定期评估：每eval_freq步评估一次 ====================
                if global_step % eval_freq == 0:
                    # 计算训练集、验证集的平均损失
                    train_loss, val_loss = cls.evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    # 保存损失值到列表
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    # 打印日志：当前轮数、步数、训练损失、验证损失
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # ==================== 每轮训练结束：计算准确率 ====================
            train_accuracy = CountLoss.calc_accuracy_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_accuracy = CountLoss.calc_accuracy_loader(
                val_loader, model, device, num_batches=eval_iter
            )
            # 打印本轮训练集与验证集的准确率
            print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
            # 保存准确率到列表
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        # 返回所有训练指标，用于绘制曲线、分析训练过程
        return train_losses, val_losses, train_accs, val_accs, examples_seen

    @classmethod
    def evaluate_model(cls, model, train_loader, val_loader, device, eval_iter):
        """
        模型评估函数：计算训练集、验证集的平均损失
        :param model: 训练中的模型
        :param train_loader: 训练集数据加载器
        :param val_loader: 验证集数据加载器
        :param device: 运行设备
        :param eval_iter: 评估使用的批次数量
        :return: 训练集平均损失、验证集平均损失
        """
        model.eval()  # 将模型切换为【评估模式】，关闭训练专用层
        with torch.no_grad():  # 关闭梯度计算，节省显存、加速计算
            # 计算训练集前eval_iter个批次的平均损失
            train_loss = CountLoss.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            # 计算验证集前eval_iter个批次的平均损失
            val_loss = CountLoss.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()  # 评估完成，切回【训练模式】继续训练
        return train_loss, val_loss

    @classmethod
    def train(cls):
        # ==================== 训练启动配置与执行 ====================
        start_time = time.time()  # 记录训练开始时间
        torch.manual_seed(123)  # 设置随机种子，保证训练结果可复现

        # 初始化优化器：AdamW（大模型首选优化器）
        # lr=5e-5：学习率（控制参数更新步长），weight_decay=0.1：权重衰减（防止过拟合）
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        """
         AdamW带权重衰减（正则化）的 Adam 优化器，专门解决 “过拟合” 问题，是现在大模型 / Transformer 训练的默认首选。
         
        1. SGD（最基础、最经典）
            全称：随机梯度下降
            优点：简单、稳定、泛化能力最强
            缺点：收敛极慢、容易卡在局部最优
            适合：图像分类、大数据集、需要极致泛化
        2. Momentum（动量 SGD）
            带惯性，冲过局部最优
            比 SGD 快，但仍不如 Adam 系列
        3. RMSprop
            自适应学习率
            常用于 RNN、序列模型
        4. Adam（最流行，但有缺陷）
            自适应学习率 + 动量
            收敛飞快
            缺点：权重衰减失效，容易过拟合
            现在基本被 AdamW 取代
        5. AdamW（目前最强首选）
            修复 Adam 缺陷
            权重衰减有效防止过拟合
            Transformer/LLM/ 文本分类标配
        6. AdamW8bit / AdamWFP8（显存优化版）
            用更少显存训练大模型
            来自 bitsandbytes 库
            适合显存小但要训大模型的场景
        """

        num_epochs = 5  # 设置训练总轮数

        # 启动训练，接收返回的训练指标
        train_losses, val_losses, train_accs, val_accs, examples_seen = cls.train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50, eval_iter=5,
            tokenizer=tokenizer
        )
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        cls.plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        train_accuracy = CountLoss.calc_accuracy_loader(train_loader, model, device)
        val_accuracy = CountLoss.calc_accuracy_loader(val_loader, model, device)
        test_accuracy = CountLoss.calc_accuracy_loader(test_loader, model, device)

        print(f"Training accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

        text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
        print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))
        text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
        print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))
        torch.save(model.state_dict(), "review_classifier.pth")


TrianClassifier.train()
