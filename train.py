# my-nanogpt/train.py

from model import MiniGPT

import os
import pickle
import numpy as np
import torch

# 1. 确定数据所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data", "shakespeare_char")

# 2. 加载 meta.pkl，获取词表大小
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
itos = meta["itos"]
stoi = meta["stoi"]
print(f"词表大小: {vocab_size}")

# 3. 使用 np.memmap 加载数据
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
print(f"训练集数据量: {len(train_data):,} 个token")
print(f"验证集数据量: {len(val_data):,} 个token")

# 4. 设置超参数
block_size = 64
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 5. 写一个 get_batch 函数
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 6. 测试抓一批数据
xb, yb = get_batch("train")
print("\n测试 get_batch 成功！")
print("输入 x 的形状 (batch_size, block_size):", xb.shape)
print("目标 y 的形状 (batch_size, block_size):", yb.shape)
print("\n看看第一条数据 x 的前20个字符 (数字形式):")
print(xb[0][:20].tolist())
print("\n对应的目标 y 的前20个字符 (数字形式):")
print(yb[0][:20].tolist())

# ==========================================
# 接下来是模型部分
# ==========================================

# 1. 定义模型超参数
n_embd = 64   # 每个字符变成 64 维的向量
n_head = 4    # 多头注意力的头数（暂时不用，后面会用到）
head_size = n_embd // n_head # 每个头的大小 (64 // 4 = 16)
n_layer = 4

# ==========================================
# 第七部分：测试 MiniGPT
# ==========================================
print("\n测试完整的 MiniGPT...")
model = MiniGPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, block_size=block_size, n_layer=n_layer).to(device)

# 1. 测试前向传播和 Loss 计算
xb, yb = get_batch("train")
logits, loss = model(xb, yb)
print("输入 x 的形状:", xb.shape)
print("输出 logits 的形状 (B, T, vocab_size):", logits.shape)
print(f"初始 loss: {loss.item():.4f}") # 初始 loss 应该在 ln(65) ≈ 4.17 左右

# 2. 测试文本生成（此时模型还没训练，输出会是一堆乱码，这是正常的）
print("\n测试生成（未训练，输出为随机乱码）：")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 起始字符设为 0
generated = model.generate(context, max_new_tokens=50)
print(generated[0].tolist())

# ==========================================
# 第八部分：定义解码函数与优化器
# ==========================================
def decode(l):
    """ 把数字列表翻译回英文字符 """
    return ''.join([itos[i] for i in l])

# 定义优化器：AdamW 是大模型预训练的标准优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ==========================================
# 第九部分：真正的训练循环！
# ==========================================
print("\n开始训练...")
max_iters = 2000      # 总训练步数
eval_interval = 500   # 每 500 步评估一次
eval_iters = 50       # 评估时跑 50 个 batch 取平均

for iter in range(max_iters):
    
    # 1. 定期评估当前的 Loss (分别看训练集和验证集)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval() # 切换到评估模式
        with torch.no_grad(): # 评估时不需要算梯度
            losses = {}
            for split in ['train', 'val']:
                split_losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = model(X, Y)
                    split_losses[k] = loss.item()
                losses[split] = split_losses.mean()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        model.train() # 切回训练模式
    
    # 2. 抓取一批训练数据
    xb, yb = get_batch('train')
    
    # 3. 前向传播（算出预测和 loss）
    logits, loss = model(xb, yb)
    
    # 4. 清空上一轮的梯度
    optimizer.zero_grad(set_to_none=True)
    
    # 5. 反向传播（计算梯度）
    loss.backward()
    
    # 6. 更新参数（核心！模型在这里学习）
    optimizer.step()

# 保存模型
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter_num': max_iters,
    'loss': losses['train'],
}
torch.save(checkpoint, os.path.join(current_dir, "ckpt.pt"))
print("\n模型已保存到 ckpt.pt！")

# ==========================================
# 第十部分：看看训练后的成果
# ==========================================
print("\n训练结束！测试生成莎士比亚风格文本：")
# 用换行符 '\n' 作为起始字符，让模型开始续写
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 如果你还记得字符表，可以试试用 'ROMEO: ' 的开头，这里先用随机起始
generated_ids = model.generate(context, max_new_tokens=200)
print(decode(generated_ids[0].tolist()))

print("\n测试指定开头 'ROMEO: ' 生成：")
# 把 'ROMEO: ' 变成数字
start_text = "ROMEO: "
start_ids = torch.tensor([stoi[s] for s in start_text], dtype=torch.long, device=device).unsqueeze(0) # 变成 (1, T)
generated_ids = model.generate(start_ids, max_new_tokens=200)
print(decode(generated_ids[0].tolist()))

