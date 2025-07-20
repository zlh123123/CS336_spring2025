import os
from typing import BinaryIO


def BPETokenizerTraining(
    file: BinaryIO, vocab_size: int, special_tokens: list[str], num_process: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE（Byte Pair Encoding）分词器

    Args:
        file (BinaryIO): 输入的二进制文件，包含训练数据
        vocab_size (int): 最终词汇表的最大大小
        special_tokens (list[str]): 特殊token列表，在这里主要是<|endoftext|>
        num_process (int): 并行处理的进程数

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 返回一个元组，包含：
            - dict[int, bytes]: token ID到字节序列的映射（词汇表）
            - list[tuple[bytes, bytes]]: BPE合并规则列表，每个元素是一对要合并的字节序列
    """
    # 构建基础词汇表和合并表
    vocab = {}
    merge=[]

    # 创建包含0-255所有字节的词汇表
    for i in range(256):
        vocab[i] = bytes([i])

    # 为特殊token分配位置
    new_token_place = 256
    for sp_token in special_tokens:
        vocab[new_token_place] = sp_token
        new_token_place += 1
        
    # 读取文件并转换为token
    

    print(vocab)
    return None


if __name__ == "__main__":
    filename = r"C:\Users\HP\Desktop\CS336_spring2025\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    with open(filename, "rb") as f:
        BPETokenizerTraining(f, 100, ["<|endoftext|>"], 5)
