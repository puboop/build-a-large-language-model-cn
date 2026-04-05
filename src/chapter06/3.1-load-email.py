# **************************************
# --*-- coding: utf-8 --*--
# @Project    : Build-A-Large-Language-Model-CN
# @Time       : 2026/4/4 14:09
# @Author     : white
# @FileName   : 3.1-load-email.py
# @Software   : PyCharm
# **************************************
import tiktoken
import torch

from src.chapter05.gpt import GPTModel, load_weights_into_gpt
from src.chapter05.gpt_download import download_and_load_gpt2
from src.chapter06.tools import classify_review, SpamDataset

# 设置分类目标为2两类
num_classes = 2
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size"    : 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate"     : 0.0,  # Dropout rate
    "qkv_bias"      : True  # Query-key-value bias
}
model_configs = {
    "gpt2-small (124M)" : {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)" : {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)"   : {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
# 输出层替换
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # 替换最后一层为分类头
# 模型权重加载
model_state_dict = torch.load("review_classifier.pth")
model.load_state_dict(model_state_dict)
# 分词器加载
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))
while True:
    text = input("用户输入：")
    print(classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length))
