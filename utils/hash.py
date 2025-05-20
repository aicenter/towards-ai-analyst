import base64

import msgpack


def generate_id(data: dict) -> str:
    binary_data = msgpack.packb(data)
    encoded = base64.urlsafe_b64encode(binary_data).decode("utf-8")
    return encoded


def decode_id(encoded: str) -> dict:
    binary_data = base64.urlsafe_b64decode(encoded.encode("utf-8"))
    return msgpack.unpackb(binary_data)
