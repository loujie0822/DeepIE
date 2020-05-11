
from fastNLP.io import Pipe, ConllLoader
from fastNLP.io import DataBundle
from fastNLP.io.pipe.utils import _add_words_field, _indexize
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP.io.pipe.utils import _add_chars_field
from fastNLP.io.utils import check_loader_paths

from fastNLP.io import Conll2003NERLoader
from fastNLP import Const

def word_shape(words):
    shapes = []
    for word in words:
        caps = []
        for char in word:
            caps.append(char.isupper())
        if all(caps):
            shapes.append(0)
        elif any(caps) is False:
            shapes.append(1)
        elif caps[0]:
            shapes.append(2)
        elif any(caps):
            shapes.append(3)
        else:
            shapes.append(4)
    return shapes


class Conll2003NERPipe(Pipe):
    """
    Conll2003的NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。
    经过该Pipe过后，DataSet中的内容如下所示

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "target", "words", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4,...]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |   True  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """

    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        """

        :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
        :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
        :param boll word_shape: 是否新增一列word shape，5维
        """
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"

        :param ~fastNLP.DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]在传入DataBundle基础上原位修改。
        :return DataBundle:
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        # 将所有digit转为0
        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        # index
        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = Conll2003NERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


from fastNLP.io import OntoNotesNERLoader

class OntoNotesNERPipe(Pipe):
    """
    处理OntoNotes的NER数据，处理之后DataSet中的field情况为

    .. csv-table::
       :header: "raw_words", "target", "words", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |   True  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """
    def __init__(self, encoding_type: str = 'bio', lower: bool = False, word_shape: bool=False):
        """

        :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
        :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
        :param boll word_shape: 是否新增一列word shape，5维
        """
        if encoding_type == 'bio':
            self.convert_tag = iob2
        elif encoding_type == 'bioes':
            self.convert_tag = lambda words: iob2bioes(iob2(words))
        else:
            raise ValueError("encoding_type only supports `bio` and `bioes`.")
        self.lower = lower
        self.word_shape = word_shape

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"

        :param ~fastNLP.DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]在传入DataBundle基础上原位修改。
        :return DataBundle:
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        if self.word_shape:
            data_bundle.apply_field(word_shape, field_name='raw_words', new_field_name='word_shapes')
            data_bundle.set_input('word_shapes')

        # 将所有digit转为0
        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
                field_name=Const.INPUT, new_field_name=Const.INPUT)

        # index
        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        data_bundle = OntoNotesNERLoader().load(paths)
        return self.process(data_bundle)


def bmeso2bio(tags):
    new_tags = []
    for tag in tags:
        tag = tag.lower()
        if tag.startswith('m') or tag.startswith('e'):
            tag = 'i' + tag[1:]
        if tag.startswith('s'):
            tag = 'b' + tag[1:]
        new_tags.append(tag)
    return new_tags


def bmeso2bioes(tags):
    new_tags = []
    for tag in tags:
        lowered_tag = tag.lower()
        if lowered_tag.startswith('m'):
            tag = 'i' + tag[1:]
        new_tags.append(tag)
    return new_tags


class CNNERPipe(Pipe):
    def __init__(self, bigrams=False, encoding_type='bmeso'):
        super().__init__()
        self.bigrams = bigrams
        if encoding_type=='bmeso':
            self.encoding_func = lambda x:x
        elif encoding_type=='bio':
            self.encoding_func = bmeso2bio
        elif encoding_type == 'bioes':
            self.encoding_func = bmeso2bioes
        else:
            raise RuntimeError("Only support bio, bmeso, bioes")

    def process(self, data_bundle: DataBundle):
        _add_chars_field(data_bundle, lower=False)

        data_bundle.apply_field(self.encoding_func, field_name=Const.TARGET, new_field_name=Const.TARGET)

        # 将所有digit转为0
        data_bundle.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
            field_name=Const.CHAR_INPUT, new_field_name=Const.CHAR_INPUT)

        #
        input_field_names = [Const.CHAR_INPUT]
        if self.bigrams:
            data_bundle.apply_field(lambda chars:[c1+c2 for c1,c2 in zip(chars, chars[1:]+['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')

        # index
        _indexize(data_bundle, input_field_names=input_field_names, target_field_names=Const.TARGET)

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        paths = check_loader_paths(paths)
        loader = ConllLoader(headers=['raw_chars', 'target'])
        data_bundle = loader.load(paths)
        return self.process(data_bundle)