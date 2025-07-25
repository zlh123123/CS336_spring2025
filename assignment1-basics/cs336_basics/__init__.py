import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # 如果包没有安装，使用默认版本
    __version__ = "0.1.0"