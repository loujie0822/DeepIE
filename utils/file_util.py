import json
import logging
import os
import pickle
import sys

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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


def _read_conll(path, encoding='utf-8', indexes=2, dropna=True):
    """
    Construct a generator to read conll items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in range(indexes)]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        if start != '':
            sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            logger.warning('Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            continue
                        raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
            elif line.startswith('#'):
                continue
            else:
                sample.append(line.split('\t'))
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                logger.error('invalid instance ends at line: {}'.format(line_idx))
                raise e
