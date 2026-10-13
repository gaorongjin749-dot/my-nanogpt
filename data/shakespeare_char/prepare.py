# my-nanogpt/data/prepare.py
import os
import pickle
import numpy as np

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, "input.txt")

# 1. 读取文本
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"文本长度: {len(text)}")

# 2. 构建字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"词表大小: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 3. 编码全文
data = np.array([stoi[c] for c in text], dtype=np.uint16)

# 4. 切分训练集和验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 5. 保存为 .bin 文件 (直接保存在当前 data 目录下)
train_data.tofile(os.path.join(current_dir, "train.bin"))
val_data.tofile(os.path.join(current_dir, "val.bin"))

# 6. 保存 meta.pkl
with open(os.path.join(current_dir, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

print("预处理完成！已生成 train.bin, val.bin, meta.pkl")