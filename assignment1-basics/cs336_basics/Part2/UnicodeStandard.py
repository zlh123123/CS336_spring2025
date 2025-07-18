# 验证 chr(0) 确实返回了一个字符
null_char = chr(0)
print(f"长度: {len(null_char)}")
print(f"Unicode 码点: {ord(null_char)}")
print(f"repr 表示: {repr(null_char)}")

# 在字符串中间使用空字符
print(f"Hello{chr(0)}World")
print(f"上面的字符串长度: {len('Hello' + chr(0) + 'World')}")

print(chr(0).__repr__())