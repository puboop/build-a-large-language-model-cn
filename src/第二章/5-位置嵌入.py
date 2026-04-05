# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/3/30 16:37
# @Author     : white
# @FileName   : 位置嵌入.py
# @Software   : PyCharm
# **************************************
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # A

        for i in range(0, len(token_ids) - max_length, stride):  # B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):  # C
        return len(self.input_ids)

    def __getitem__(self, idx):  # D
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # B
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # C
        num_workers=0  # D
    )

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
# 我们现在考虑更现实和有用的嵌入大小，并将输入token编码为256维的向量表示。
# 这比原始的GPT-3模型使用的要小（在GPT-3中，嵌入大小为12,288维），
# 但对于实验仍然是合理的。此外，我们假设token ID 是由我们之前实现的BPE分词器创建的，该分词器的词汇量为50,257
# token嵌入层 包含所有的token，每个token占据一行
# 会自动初始化出vocab_size行output_dim列的矩阵，这个矩阵的初始化方式如何，暂未得知。
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 如果我们从数据加载器中采样数据，我们将每个批次中的每个token嵌入到一个 256 维的向量中。
# 如果我们的批次大小为 8，每个批次有四个token，那么结果将是一个形状为 8 x 4 x 256 的张量
max_length = 4
# 定义数据加载器，数据的批次为8行max_length列的矩阵，表示着当前数据同时有8个样本数据
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
# 根据8个样本的数据，取出每个样本的token所处于token嵌入层的具体行的数据
token_embeddings = token_embedding_layer(inputs)
# 一共有8 x 4个token，需要取出每个token的嵌入向量，每个向量有256列
print(token_embeddings.shape)  # 8 x 4 x 256

# 初始位置嵌入层
context_length = max_length
# 行数为context_length，这个为当前输入的token大小保持一致，列数为output_dim，这个与词嵌入向量保持一致，以便后续的运算
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
