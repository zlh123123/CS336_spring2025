import os
from typing import BinaryIO
import multiprocessing as mp
from functools import partial


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def chunk2token(
    chunk: bytes, special_tokens: list[str], vocab: dict[int, bytes]
) -> list[int]:
    """
    将chunk转换为token ID序列

    Args:
        chunk: 原始字节数据
        special_tokens: 特殊token列表
        vocab: 当前词汇表

    Returns:
        list[int]: token ID序列
    """
    # 字节找token ID
    bytes_to_id = {v: k for k, v in vocab.items()}
    # 先处理特殊token
    i = 0
    result = []

    while i < len(chunk):
        is_special = False
        for special_token in special_tokens:
            special_byte = special_token.encode("utf-8")
            # chunk中第i项往后的，如果开头是special_byte
            if chunk[i:].startswith(special_byte):
                # 在字典中能找到
                if special_byte in bytes_to_id:
                    result.append(bytes_to_id[special_byte])
                    i += len(special_byte)
                    is_special = True
                    break

        if not is_special:
            byte = bytes([chunk[i]])
            if byte in bytes_to_id:
                result.append(bytes_to_id[byte])
            i += 1

    return result


def count_pairs(tokens: list[int]) -> dict[tuple[int, int], int]:
    """
    统计每个字节对的出现次数

    Args:
        tokens (list[int]): token列表

    Returns:
        dict[tuple[int,int],int]: 每对相邻token的值为键，出现次数为值
    """
    pairs_num = {}

    for i in range(len(tokens) - 1):
        twotoken = (tokens[i], tokens[i + 1])
        pairs_num[twotoken] = pairs_num.get(twotoken, 0) + 1
    return pairs_num


def merge_pair_in_tokens(
    tokens: list[int], pairs: tuple[int, int], new_token_id: int
) -> list[int]:
    """
    把指定的token对合并后，根据其新的id，生成新的token list
    这个的作用是整个token list

    Args:
        tokens (list[int]): 原token序列
        pairs (tuple[int,int]): 待合并token对
        new_token_id (int): token对对应的新token id

    Returns:
        list[int]: 新token序列
    """
    new_token = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pairs[0] and tokens[i + 1] == pairs[1]:
            new_token.append(new_token_id)
            i += 2
        else:
            new_token.append(tokens[i])
            i += 1

    return new_token


def BPETokenizerTraining(
    file: str, vocab_size: int, special_tokens: list[str], num_process: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE（Byte Pair Encoding）分词器

    Args:
        file (str): 文件路径
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
    merge = []
    all_tokens = []

    # 创建包含0-255所有字节的词汇表
    for i in range(256):
        vocab[i] = bytes([i])

    # 为特殊token分配位置
    new_token_place = 256
    for sp_token in special_tokens:
        vocab[new_token_place] = sp_token.encode("utf-8")
        new_token_place += 1

    # 读取文件并转换为token
    with open(file, "rb") as f:

        # 分块处理，根据特殊token将文件分为num_process个chunk，每个进程单独处理一个chunk
        if num_process > 1:

            # print(special_tokens[0].encode("utf-8"))
            boundaries = find_chunk_boundaries(
                f,
                num_process,
                special_tokens[0].encode("utf-8"),  # 默认就选第一个特殊token
            )
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start)
                chunks.append(chunk)

            with mp.Pool(num_process) as pool:
                process_func = partial(
                    chunk2token, special_tokens=special_tokens, vocab=vocab
                )
                chunk_results = pool.map(process_func, chunks)
                for chunk_tokens in chunk_results:
                    all_tokens.extend(chunk_tokens)

        # 不需要分块就这样子
        else:
            f.seek(0)
            data = f.read()
            all_tokens = chunk2token(data, special_tokens, vocab)

    ### 下面开始BPE的训练循环
    now_vocab_size = len(vocab)

    while now_vocab_size < vocab_size:
        pair_counts = count_pairs(all_tokens)
        # 没有token列表
        if not pair_counts:
            break

        # 选出现次数最大的token对
        max_count_token_pair = max(pair_counts, key=pair_counts.get)

        # 组合上面的token对后生成的新token id即now_vocab_size的值
        new_token_id = now_vocab_size

        # 把词汇表中新添加合成后的token对
        # 先需要知道原来两个token对应的字节都是啥
        byte1 = vocab[max_count_token_pair[0]]
        byte2 = vocab[max_count_token_pair[1]]

        new_byte = byte1 + byte2
        vocab[new_token_id] = new_byte

        # 合并记录到merge里
        merge.append((byte1, byte2))

        # 把合并作用到整个token列表
        all_tokens = merge_pair_in_tokens(
            all_tokens, max_count_token_pair, new_token_id
        )

        now_vocab_size += 1

    # print(vocab)
    return vocab, merge


# if __name__ == "__main__":
#     filename = r"C:\Users\HP\Desktop\CS336_spring2025\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"

#     BPETokenizerTraining(filename, 100, ["<|endoftext|>"], 10)
