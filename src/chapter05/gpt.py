import numpy as np
import torch
from torch import nn

from torch.nn import GELU


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        定义注意力层需要的所有参数和层
        :param d_in: 输入特征维度（GPT2中是768）
        :param d_out: 输出特征维度（通常和d_in相等，保证残差连接）
        :param context_length: 最大上下文长度（模型能处理的最大词数）
        :param dropout: dropout概率，防止过拟合
        :param num_heads: 注意力头的数量（比如12头）
        :param qkv_bias: Q/K/V投影层是否使用偏置项
        """
        super().__init__()
        # 断言：输出维度必须能被头数整除（保证每个头维度相同）
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out  # 保存输出维度
        self.num_heads = num_heads  # 保存注意力头数量
        # 在定义多头时，需要考虑到最终每头输出的维度大小，最后会将这些头的输出大小拼接为一个完成的矩阵，也就是输出的大小
        self.head_dim = d_out // num_heads  # 每个注意力头的维度 = 总维度 / 头数

        # 定义Q/K/V三个线性投影层：输入d_in → 输出d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询向量投影层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键向量投影层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值向量投影层

        # 将多头的结果整合多头结果
        # 这个矩阵的大小为：d_out x d_out是因为输入已经是拼接好的 d_out，希望输出维度不变（为了残差连接）
        # 输入的数据是一个(1,4,768)而当前是一个d_out的方形矩阵，(1,4,768)可以看成(4,768)，所以(4,768)x(768,768)得到的结果还是(4,768)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层：合并所有头的结果
        self.dropout = nn.Dropout(dropout)  # Dropout层：随机失活，防过拟合

        # 注册因果掩码（上三角矩阵）：确保只能看前面的词，不能看未来的词
        # 会生成一个context_length x context_length的全是1的上三角矩阵
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播：真正计算注意力的逻辑
        :param x: 输入张量，形状 (batch_size, 词数量, 输入维度)
        :return:
        """
        # 解包输入形状：b=批次大小，num_tokens=句子词数，d_in=输入维度
        b, num_tokens, d_in = x.shape

        # 1. 线性投影得到Q/K/V向量
        keys = self.W_key(x)  # 键：形状 (b, num_tokens, d_out)
        queries = self.W_query(x)  # 查询：形状 (b, num_tokens, d_out)
        values = self.W_value(x)  # 值：形状 (b, num_tokens, d_out)

        # 2.形状变换类似于reshape 拆分多头：把d_out拆成 [num_heads, head_dim]
        # 形状变化：(b, num_tokens, d_out) → (b, num_tokens, num_heads, head_dim)
        """
        执行前：
        keys.shape = (b, num_tokens, d_out)
            b：批次大小（一次喂多少句话）
            num_tokens：每句话有多少个词
            d_out：每个词的向量长度（比如 768）
        执行后：
            keys.shape = (b, num_tokens, num_heads, head_dim)
            num_heads：注意力头数（比如 12）
            head_dim：每个头的维度 = d_out //num_heads（768/12=64）
        原本是b批次，num_tokens个token数，d_in列（嵌入数据量）
        现在是b批次，num_tokens个token数，self.num_heads列（当前注意力头数），每个注意力头分到：self.head_dim
        (批次, 词数, 头数, 头维度)
        作用：把最后一维的总特征维度，拆分成 “多头 + 单头维度”，让模型可以并行计算多个注意力头。
        可以理解为将d_in拆分为：self.num_heads x self.head_dim
        """
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. 转置维度：把注意力头放到前面，方便并行计算
        # 形状变化：(b, num_tokens, num_heads, head_dim) → (b, num_heads, num_tokens, head_dim)
        # (批次, 词数, 头数, 头维度) -> (批次, 头数, 词数, 头维度)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. 计算注意力分数：Q × K^T （点积相似度）
        # 形状：(b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 5. 生成因果掩码：截断到当前句子的词数，转成布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 6. 掩码填充：把未来位置的分数设为负无穷（softmax后变为0）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 7. 计算注意力权重：缩放+softmax归一化
        # 除以√head_dim 防止数值过大，dim=-1按行归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 对权重做dropout

        # 8. 加权求和得到上下文向量：权重 × V
        # 形状：(b, num_heads, num_tokens, head_dim)
        # 转置后：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 9. 合并多头：把所有头的结果拼接回d_out维度
        # 形状：(b, num_tokens, d_out)
        # 这里的合并多头指的是合并qkv的计算结果
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 10. 最终投影：线性层整合结果（可选）
        context_vec = self.out_proj(context_vec)

        return context_vec  # 返回最终的上下文向量


class FeedForward(nn.Module):
    """
    前馈神经网络对每个位置的特征单独做 “深加工 + 抽象判断”
    逐词独立处理（不看其他词，只看当前词的特征）
    非线性变换 + 特征提纯（把有用信息留下，没用的丢掉）
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 输入，将当前维度信息放大4倍进行更细致的查看 放大
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 一个非线性激活函数，它的核心任务：给模型加入 “非线性能力”，让神经网络能学会复杂的规律，而不是只能算简单的线性问题。
            # 如果没有激活函数，神经网络再深也只是线性变换，跟一层没区别，学不会复杂的语义（比如垃圾短信识别、文本分类）
            # ReLU：硬截断，负数直接变 0，容易造成神经元 “死亡”
            # GELU：平滑过渡，负数慢慢变小，训练更稳定、收敛更快
            GELU(),
            # 输出，将放大4倍的神经元缩小至原有的大小 压缩
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        定义归一化需要的参数
        作用 1:
            LayerNorm 把每一个词的特征，强行变成均值≈0，方差≈1 的标准分布，让训练超级稳定、不震荡、不爆炸。
            LayerNorm 强行把值拉回标准范围→ 训练不震荡、不爆炸、不消失
        作用 2：让每一层输入都保持标准分布
            让注意力层、前馈层学得更快、更稳
        作用 3：Transformer 必须用它
            没有 LayerNorm→ 你的模型根本训不起来→ 注意力计算会数值爆炸→ 分类完全不准
        :param emb_dim: 模型的特征维度（比如768）
        """
        super().__init__()
        # 极小值，防止分母为0
        self.eps = 1e-5
        # 可学习的缩放参数 → 形状：[emb_dim]
        # 初始全是 1
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习的偏移参数 → 形状：[emb_dim]
        # 初始全是 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        前向传播：真正做归一化的地方
        :param x: 输入形状：(批次, 词数, 特征维度)
        :return:
        """
        # 1. 对 最后一维（特征维度）求均值
        mean = x.mean(dim=-1, keepdim=True)
        # 2. 对 最后一维求方差（不开根号，不使用无偏估计）
        #  unbiased=False = 用简单方差，不是统计无偏方差（深度学习都这么用）
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 3. 核心归一化公式
        # 标准归一化公式: (x - 均值) / 标准差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 4. 让网络自己决定最优分布形态
        # 归一化后 × 可学习scale + 可学习shift
        return self.scale * norm_x + self.shift


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x  # 注意力模块中的快捷连接
        x = self.norm1(x)  # 归一化
        x = self.att(x)  # 多头注意力（找词与词之间的关系）
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut  # 残差相加
        shortcut = x  # 前馈网络模块中的快捷链接
        x = self.norm2(x)  # 归一化
        x = self.ff(x)  # 经过多头注意力的计算后，将其信息进行更为细致的查看，然后将查看修改的信息又压缩回原来的大小
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回到输出中
        return x


class GPTModel(nn.Module):
    """
    为什么 Transformer 用 LayerNorm 而不是 BatchNorm？
        这是深度学习里的高频考点，核心原因就一句话：两者归一化的维度完全相反，LayerNorm 更适配 Transformer 处理文本的特性
        我们先从两个归一化的本质区别讲起，再结合你的文本分类模型分析。
    一、先搞懂：LayerNorm 和 BatchNorm 归一化的维度不一样
        假设你的输入张量形状是：(batch_size, num_tokens, emb_dim)
        batch_size：一次训练的句子数（比如 32）
        num_tokens：每句话的词数（比如 20）
        emb_dim：每个词的特征维度（比如 768）
        1. BatchNorm：跨样本、同特征 归一化
            BatchNorm 的计算维度是 batch_size，它的逻辑是：
            对同一个特征位置，计算所有样本的均值和方差
            对应到你的文本张量，它会这么算：
            固定第 2 个词、第 100 维特征 → 计算 32 个样本在这个位置的均值 / 方差
            固定第 5 个词、第 300 维特征 → 再计算 32 个样本在这个位置的均值 / 方差
            致命缺点（对文本不友好）：
            依赖 batch_size 大小：如果 batch 太小，均值 / 方差统计不准，模型训崩
            文本长度不固定：不同句子的 num_tokens 不一样，BatchNorm 没法统一计算
            不适合 NLP 任务：文本的特征是每个词的独立向量，不是图像那种 “空间共享特征”
        2. LayerNorm：单样本、跨特征 归一化
            LayerNorm 的计算维度是 emb_dim（最后一维特征），它的逻辑是：
            对同一个样本的同一个词，计算所有特征维度的均值和方差
            对应到你的文本张量，它会这么算：
            固定第 1 个样本、第 3 个词 → 计算这个词 768 维特征的均值 / 方差
            固定第 2 个样本、第 5 个词 → 计算这个词 768 维特征的均值 / 方差
            核心优点（适配文本）：
            不依赖 batch_size：哪怕 batch=1 也能正常计算，训练稳定
            不关心文本长度：每个词独立归一化，长短句子都能处理
            适合 NLP 任务：聚焦单个词的特征分布，刚好匹配 Transformer 对词向量的处理逻辑
    二、Transformer 选 LayerNorm 的 3 个核心原因
        1. 文本的 batch_size 不稳定
            NLP 训练时，为了提高效率会做动态 padding（把长短不一的句子补到同一长度），但 batch_size 经常波动（比如最后一个 batch 样本数很少）。
            BatchNorm：小 batch 下统计的均值 / 方差偏差大，直接影响模型效果
            LayerNorm：只看单个样本的特征，不受 batch 大小影响，稳如老狗
        2. Transformer 的 残差连接要求维度不变
            你的 TransformerBlock 里有残差连接：x = x + shortcut这要求归一化后的张量形状必须和输入完全一致。
            LayerNorm：只归一化最后一维特征，形状完全不变，完美适配残差连接
            BatchNorm：会引入额外的 “移动均值 / 移动方差” 参数，文本场景下容易和残差连接冲突
        3. 注意力机制需要 词向量的独立分布
            多头注意力是对每个词的向量计算相似度，这要求每个词的特征分布是稳定的。
            LayerNorm：给每个词的向量做标准化，让注意力分数计算更稳定，不会出现数值爆炸
            BatchNorm：跨样本归一化会破坏单个词向量的独立性，导致注意力机制失效
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 多头注意力层
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后一个激活层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 设备设置将根据输入数据所在的位置选择在 CPU 或 GPU 上训练模型
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # return torch.nn.Parameter(torch.tensor(right))
    return torch.nn.Parameter(
        torch.tensor(right, dtype=torch.float32)  # 指定dtype，彻底解决类型问题
    )


def load_weights_into_gpt(gpt, params):
    # OpenAI 的原始 GPT-2 模型在输出层中复用了 token 嵌入的权重，以减少参数总量，这一概念称为权重共享
    # 将模型的位置嵌入和token 嵌入的权重设置为 params 中指定的值
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    # 遍历模型中的每个 Transformer 模块
    for b in range(len(params["blocks"])):
        # 使用 np.split 函数将注意力和偏置权重分为三等份，分别用于查询、键和值组件
        # 1. 把 c_attn（三合一投影层）的权重 w 在最后一维切成 3 等份 → Q、K、V 权重
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 2. 把 Q 权重转置后赋值给模型的 W_query 层
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        # 3. 把 K 权重转置后赋值给模型的 W_key 层
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        # 4. 把 V 权重转置后赋值给模型的 W_value 层
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                     params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                       params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                     params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                       params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                     params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])


def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):  # For循环与之前相同：获取logits，仅关注最后的时间步
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:  # 在新步骤中，通过top-k采样过滤logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:  # 在新步骤中应用temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # 在未使用temperature scaling时，执行贪婪的下一个token选择
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:  # 如果遇到序列结束token且指定了eos_id，则提前停止生成
            break
        # idx_next = idx_next.unsqueeze(1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前上下文中索引的数组，形状为 (batch, n_tokens)
    for _ in range(max_new_tokens):
        # 若上下文长度超出支持范围，则进行裁剪。例如，若模型仅支持 5 个 token，而上下文长度为 10，仅使用最后 5 个 token 作为上下文
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # 仅关注最后一个时间步，将形状从 (batch, n_token, vocab_size) 转换为 (batch, vocab_size)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)  # probas 的形状为 (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # idx_next 的形状为 (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # 将采样的索引追加到当前序列中，此时 idx 的形状为 (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
