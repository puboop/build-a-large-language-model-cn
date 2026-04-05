# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/4/4 14:10
# @Author     : white
# @FileName   : tools.py
# @Software   : PyCharm
# **************************************
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        # 对文本进行预分词
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 若序列超过最大长度则进行截断
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        # 将序列填充至最长序列长度
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                              for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    # 在推理模式下禁用dropout
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(text)  # 准备模型输入
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_context_length)]  # 截断过长序列

    input_ids += [pad_token_id] * (max_length - len(input_ids))  # 填充序列至最长长度
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 增加批次维度

    with torch.no_grad():  # 关闭梯度跟踪，进行模型推理
        logits = model(input_tensor)[:, -1, :]  # 获取最后一个输出 token 的 logits
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"  # 返回分类结果
