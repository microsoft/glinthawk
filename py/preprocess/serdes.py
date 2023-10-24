import os
import struct
import base58
import hashlib
import logging

def sha256_base58(data):
    """Compute the sha256 hash of the data and encode it in base58."""
    return base58.b58encode(hashlib.sha256(data).digest())


def serialize(tokens, output_dir):
    """Serialize a list of tokens into a byte string."""
    """Each token is represented as a 4-byte LE integer."""

    serialized_data = b""
    serialized_data += struct.pack("<I", len(tokens))

    for token in tokens:
        serialized_data += struct.pack("<I", token)

    output_file = sha256_base58(serialized_data).decode()
    output_path = os.path.join(output_dir, f'{output_file}.ghp')

    with open(output_path, "wb") as f:
        f.write(serialized_data)

    logging.info("Serialized %d tokens to %s", len(tokens), output_file)

def deserialize(input_path):
    """Deserialize a list of tokens from a byte string."""
    """Each token is represented as a 4-byte LE integer."""

    with open(input_path, "rb") as f:
        serialized_data = f.read()

    num_tokens = struct.unpack("<I", serialized_data[:4])[0]
    tokens = struct.unpack("<{}I".format(num_tokens), serialized_data[4:])

    logging.info("Deserialized %d tokens from %s", len(tokens), input_path)
    assert len(tokens) == num_tokens, "Number of extracted tokens does not match."
    return tokens
