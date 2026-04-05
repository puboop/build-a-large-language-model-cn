# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/3/30 15:10
# @Author     : white
# @FileName   : BPE分词器.py
# @Software   : PyCharm
# **************************************
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
#
# a = [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
# print(tokenizer.decode(a))

a = "Akwirw ier"
tokenId = tokenizer.encode(a)
print(tokenId)

print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
