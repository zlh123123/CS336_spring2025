def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


print(decode_utf8_bytes_to_str_wrong("！！！".encode("utf-8")))