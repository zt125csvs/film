import os
import re
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

SEED = 52
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


### ====================== 数据预处理部分 ====================== ###
#### 1. 处理GloVe词向量，保存为npy文件
def load_glove_vector():
    """
    加载GloVe词向量文件，转换为单词列表和向量数组，并保存为.npy文件
    """
    word_list = []
    vocabulary_vectors = []
    glove_path = "data/glove.6B.50d.txt"  # GloVe文件路径
    assert os.path.isfile(glove_path), f"错误：GloVe文件未找到 - {glove_path}"

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="加载GloVe词向量"):
            parts = line.strip().split()
            word = parts[0].lower()
            vector = list(map(float, parts[1:]))

            word_list.append(word)
            vocabulary_vectors.append(vector)

    # 转换为numpy数组并保存
    word_list = np.array(word_list, dtype=np.object_)
    vocabulary_vectors = np.array(vocabulary_vectors, dtype=np.float32)

    npys_dir = "data/npys/"
    os.makedirs(npys_dir, exist_ok=True)  # 自动创建目录
    np.save(os.path.join(npys_dir, "word_list.npy"), word_list)
    np.save(os.path.join(npys_dir, "vocabulary_vectors.npy"), vocabulary_vectors)

    print(f"GloVe处理完成：单词数={len(word_list)}, 向量维度={vocabulary_vectors.shape[1]}")
    return word_list, vocabulary_vectors


#### 2. 加载IMDb数据集，预处理文本（去标点、转小写）
def load_imdb_data(path="data/Imdb", flag="train"):
    """
    加载IMDb数据集（train/test），返回预处理后的单词列表和标签
    :param path: IMDb数据集根路径
    :param flag: train或test
    :return: 列表，每个元素为[单词列表, 标签（1=pos, 0=neg）]
    """
    data = []
    for label in ["pos", "neg"]:
        # 构建路径（如data/Imdb/train/pos）
        label_path = os.path.join(path, flag, label)
        assert os.path.isdir(label_path), f"错误：目录未找到 - {label_path}"

        files = os.listdir(label_path)
        for file in tqdm(files, desc=f"处理{flag}-{label}数据", leave=False):
            file_path = os.path.join(label_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

                # 预处理步骤
                text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号（保留字母数字和空格）
                text = text.lower()  # 转小写
                words = [word for word in text.split() if word]  # 分割为单词列表，过滤空字符串

                data.append([words, 1 if label == "pos" else 0])

    print(f"加载{flag}数据完成：样本数={len(data)}")
    return data


#### 3. 将文本转换为索引序列（截断/填充至固定长度）
def text_to_indices(dataset, word_list, max_len=250, oov_idx=399999):
    """
    将单词列表转换为GloVe索引序列
    :param dataset: 数据集（load_imdb_data返回的格式）
    :param word_list: GloVe单词列表（从npy加载）
    :param max_len: 最大序列长度（截断/填充目标）
    :param oov_idx: 未登录词（OOV）的索引
    :return: 索引数组（形状为[样本数, max_len]）
    """
    word_set = set(word_list)
    indices_list = []

    for words, _ in tqdm(dataset, desc="转换为索引序列"):
        indices = []
        for word in words:
            indices.append(word_list.tolist().index(word) if word in word_set else oov_idx)

        # 截断或填充
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))  # 填充0（注意：0可能对应实际单词，需确保GloVe中0为填充符）
        else:
            indices = indices[:max_len]

        indices_list.append(indices)

    return np.array(indices_list, dtype=np.int32)


#### 4. 数据预处理主流程
def preprocess_data(load_glove=True):
    """
    数据预处理主函数：加载GloVe、处理IMDb数据、生成索引和词向量
    :param load_glove: 是否重新加载GloVe（首次运行设为True，后续设为False以加快速度）
    """
    npys_dir = "data/npys/"

    # 加载或生成GloVe数据
    if load_glove:
        word_list, vocab_vectors = load_glove_vector()
    else:
        word_list = np.load(os.path.join(npys_dir, "word_list.npy"), allow_pickle=True)
        vocab_vectors = np.load(os.path.join(npys_dir, "vocabulary_vectors.npy"))

    # 加载IMDb数据
    train_data = load_imdb_data(flag="train")
    test_data = load_imdb_data(flag="test")
    all_data = train_data + test_data  # 合并用于后续划分

    # 转换为索引序列
    train_indices = text_to_indices(train_data, word_list)
    test_indices = text_to_indices(test_data, word_list)

    # 保存索引数据
    np.save(os.path.join(npys_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(npys_dir, "test_indices.npy"), test_indices)
    np.save(os.path.join(npys_dir, "train_labels.npy"), np.array([d[1] for d in train_data]))
    np.save(os.path.join(npys_dir, "test_labels.npy"), np.array([d[1] for d in test_data]))

    print("数据预处理完成，已保存索引和标签到npys目录")

    ### ====================== 模型定义部分 ====================== ###


class LSTMClassifier(nn.Module):
    """
    基于LSTM的文本分类模型
    """

    def __init__(self, input_size=50, hidden_size=64, num_layers=1, dropout=0.5, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输入维度（词向量维度，GloVe为50）
            hidden_size=hidden_size,  # 隐藏层维度
            num_layers=num_layers,  # LSTM层数
            batch_first=True,  # 输入形状为(batch_size, seq_len, input_size)
            bidirectional=False  # 单向LSTM
        )

        # 全连接层（分类头）
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为(batch_size, seq_len, input_size)
        :return: 输出张量，形状为(batch_size, num_classes)
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM输出：out (batch_size, seq_len, hidden_size), (hn, cn)
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        last_out = out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        x = self.dropout(last_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    ### ====================== 模型训练部分 ====================== ###


def train_model(batch_size=100, epochs=2, lr=5e-5):
    """
    训练模型并保存参数
    :param batch_size: 批次大小
    :param epochs: 训练轮数
    :param lr: 学习率
    :return: 训练损失历史
    """
    npys_dir = "data/npys/"

    # 加载预处理数据
    train_indices = np.load(os.path.join(npys_dir, "train_indices.npy"))
    train_labels = np.load(os.path.join(npys_dir, "train_labels.npy"))
    vocab_vectors = np.load(os.path.join(npys_dir, "vocabulary_vectors.npy"))

    # 转换为词向量（形状：[样本数, seq_len, input_size]）
    train_vectors = np.array([vocab_vectors[idx] for idx in train_indices])

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_vectors).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = LSTMClassifier(input_size=50, hidden_size=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    train_losses = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录并打印训练损失
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} | 平均损失: {avg_loss:.4f}")

    # 保存模型
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_classifier.pth"))
    print("模型已保存到models/lstm_classifier.pth")

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses


### ====================== 模型评估部分 ====================== ###
def evaluate_model(batch_size=100, test_type="full"):
    """
    评估模型性能
    :param batch_size: 批次大小
    :param test_type: full（完整测试集）或custom（自定义100条样本）
    """
    npys_dir = "data/npys/"
    model_dir = "models/"

    # 加载模型
    model = LSTMClassifier(input_size=50, hidden_size=64).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_classifier.pth")))
    model.eval()

    if test_type == "full":
        # 完整测试集评估
        test_indices = np.load(os.path.join(npys_dir, "test_indices.npy"))
        test_labels = np.load(os.path.join(npys_dir, "test_labels.npy"))
        vocab_vectors = np.load(os.path.join(npys_dir, "vocabulary_vectors.npy"))
        test_vectors = np.array([vocab_vectors[idx] for idx in test_indices])

        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_vectors).float(),
            torch.from_numpy(test_labels).long()
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # 计算准确率
        accuracy = correct / total * 100
        print(f"完整测试集准确率: {accuracy:.2f}%")



        return accuracy

    elif test_type == "custom":
        # 自定义测试数据（100条：50正+50负）
        custom_path = "data/test_review/test"
        labels = ["pos", "neg"]
        custom_data = []
        true_labels = []
        review_texts = []  # 保存原始文本用于展示

        for label in labels:
            label_path = os.path.join(custom_path, label)
            assert os.path.isdir(label_path), f"错误：自定义测试路径未找到 - {label_path}"

            files = os.listdir(label_path)[:50]  # 各取50条
            for file in files:
                file_path = os.path.join(label_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    review_texts.append(text)  # 保存原始文本

                    # 预处理步骤与训练时一致
                    text = re.sub(r'[^\w\s]', '', text).lower()
                    words = [word for word in text.split() if word]

                    # 转换为索引
                    word_list = np.load(os.path.join(npys_dir, "word_list.npy"), allow_pickle=True)
                    word_set = set(word_list)
                    indices = [word_list.tolist().index(word) if word in word_set else 399999 for word in words]
                    indices = indices[:250] if len(indices) > 250 else indices + [0] * (250 - len(indices))

                    custom_data.append(indices)
                    true_labels.append(1 if label == "pos" else 0)

        # 转换为词向量
        vocab_vectors = np.load(os.path.join(npys_dir, "vocabulary_vectors.npy"))
        custom_vectors = np.array([vocab_vectors[idx] for idx in custom_data])
        inputs = torch.from_numpy(custom_vectors).float().to(device)
        true_labels = torch.tensor(true_labels, dtype=torch.long).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

        # 输出前10条结果
        print("自定义测试结果（前10条）:")
        for i in range(min(10, len(review_texts))):
            text = review_texts[i][:300] + "..." if len(review_texts[i]) > 300 else review_texts[i]
            print(f"文本: {text}")
            print(f"预测: {'正面' if predicted[i] == 1 else '负面'}, 真实: {'正面' if true_labels[i] == 1 else '负面'}")
            print("-" * 80)

        accuracy = (predicted == true_labels).sum().item() / len(true_labels)
        print(f"自定义测试准确率: {accuracy * 100:.2f}%")

        return accuracy





### ====================== 主函数 ====================== ###
if __name__ == "__main__":
    # 步骤1：数据预处理（首次运行需设为True，后续设为False）
    preprocess_data(load_glove=True)  # 仅首次运行时加载GloVe

    # 步骤2：训练模型
    train_losses = train_model(batch_size=100, epochs=10)

    # 步骤3：评估模型（可选full或custom）
    test_accuracy = evaluate_model(test_type="full")
    # custom_accuracy = evaluate_model(test_type="custom")