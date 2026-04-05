# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/3/30 16:18
# @Author     : white
# @FileName   : 嵌入层理解.py
# @Software   : PyCharm
# **************************************
import torch

# 假设我们有以下四个输入token，它们的 ID 分别为 2、3、5 和 1：
input_ids = torch.tensor([2, 3, 5, 1])
# 为了简化并起到说明的目的，假设我们有一个只有 6 个单词的小词汇表（而不是 BPE 分词器中的 50,257 个单词），
# 并且我们希望创建大小为 3 的嵌入向量（在 GPT-3 中，嵌入大小为 12,288 维）：
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
# 得到当前嵌入层的信息，vocab_size行，output_dim列的二维矩阵
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
# 获取指定行的权重信息
# print(embedding_layer(torch.tensor([3])))
# 如果我们将token ID 3 的嵌入向量与之前的嵌入矩阵进行比较，会发现它与第四行相同（Python 从零开始索引，因此它对应于索引 3 的行）。
# 换句话说，嵌入层本质上是一个查找功能，通过token ID 从嵌入层的权重矩阵中检索行。

# 现在让我们将其应用于之前定义的所有四个输入 ID
# 实际上这里只是做了检索操作，将embedding_layer.weight中的矩阵数据按照索引从0开始，按照input_ids的输入顺序进行输出了
# 这就是所谓词向量嵌入
print(embedding_layer(input_ids))
