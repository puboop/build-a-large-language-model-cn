# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/3/30 14:23
# @Author     : white
# @FileName   : a.py
# @Software   : PyCharm
# **************************************
# Listing 2.1 Reading in a short story as text sample into Python
import re

from typing import Dict, Iterable

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# Listing 2.3 Implementing a simple text tokenizer
class SimpleTokenizerV1:
    """
    将词汇表作为类属性存储，以方便在 encode 和 decode 方法中访问
    """

    def __init__(self, vocab: Dict[str, int]):
        """
        :param vocab:词向量表
        """
        self.str_to_int = vocab
        # 创建一个反向词汇表，将token ID 映射回原始的文本token
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str):
        """
        将输入文本转换为token ID
        :param text:
        :return:
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: Iterable):
        """
        将token ID 还原为文本
        :param ids:
        :return:
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # 在指定的标点符号前去掉空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # E
        return text

    @classmethod
    def main(cls):
        tokenizer = cls(vocab)
        text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
        ids = tokenizer.encode(text)
        print(ids)
        print(tokenizer.decode(ids))


# Listing 2.4 A simple text tokenizer that handles unknown words
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 用 <|unk|> tokens替换未知词汇
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 在指定标点符号前替换空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

    @classmethod
    def main(cls):
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|endoftext|> ".join((text1, text2))
        print(text)


SimpleTokenizerV2.main()
