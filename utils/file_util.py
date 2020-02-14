import json
import logging
import os
import pickle
import sys


def pickle_dump_large_file(obj, filepath):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load_large_file(filepath):
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj


def save(filepath, obj, message=None):
    if message is not None:
        logging.info("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def load(filepath):
    return pickle_load_large_file(filepath)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'wb') as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False).
                encode('utf-8'))
