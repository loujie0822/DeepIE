

from fastNLP.embeddings import TokenEmbedding
import torch
from fastNLP import Vocabulary
import torch.nn.functional as F
from fastNLP import logger
from fastNLP.embeddings.utils import _construct_char_vocab_from_vocab, get_embeddings
from torch import nn
from .transformer import TransformerEncoder


class TransformerCharEmbed(TokenEmbedding):
    def __init__(self, vocab: Vocabulary, embed_size: int = 30, char_emb_size: int = 30, word_dropout: float = 0,
                 dropout: float = 0, pool_method: str = 'max', activation='relu',
                 min_char_freq: int = 2, requires_grad=True, include_word_start_end=True,
                 char_attn_type='adatrans', char_n_head=3, char_dim_ffn=60, char_scale=False, char_pos_embed=None,
                 char_dropout=0.15, char_after_norm=False):
        """
        :param vocab: 词表
        :param embed_size: TransformerCharEmbed的输出维度。默认值为50.
        :param char_emb_size: character的embedding的维度。默认值为50. 同时也是Transformer的d_model大小
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param dropout: 以多大概率drop character embedding的输出以及最终的word的输出。
        :param pool_method: 支持'max', 'avg'。
        :param activation: 激活函数，支持'relu', 'sigmoid', 'tanh', 或者自定义函数.
        :param min_char_freq: character的最小出现次数。默认值为2.
        :param requires_grad:
        :param include_word_start_end: 是否使用特殊的tag标记word的开始与结束
        :param char_attn_type: adatrans or naive.
        :param char_n_head: 多少个head
        :param char_dim_ffn: transformer中ffn中间层的大小
        :param char_scale: 是否使用scale
        :param char_pos_embed: None, 'fix', 'sin'. What kind of position embedding. When char_attn_type=relative, None is
            ok
        :param char_dropout: Dropout in Transformer encoder
        :param char_after_norm: the normalization place.
        """
        super(TransformerCharEmbed, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        assert char_emb_size%char_n_head == 0, "d_model should divide n_head."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")

        logger.info("Start constructing character vocabulary.")
        # 建立char的词表
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        logger.info(f"In total, there are {len(self.char_vocab)} distinct characters.")
        # 对vocab进行index
        max_word_len = max(map(lambda x: len(x[0]), vocab))
        if include_word_start_end:
            max_word_len += 2
        self.register_buffer('words_to_chars_embedding', torch.full((len(vocab), max_word_len),
                                                                    fill_value=self.char_pad_index, dtype=torch.long))
        self.register_buffer('word_lengths', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了. 修改为不区分pad与否
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = \
                torch.LongTensor([self.char_vocab.to_index(c) for c in word])
            self.word_lengths[index] = len(word)

        self.char_embedding = get_embeddings((len(self.char_vocab), char_emb_size))
        self.transformer = TransformerEncoder(1, char_emb_size, char_n_head, char_dim_ffn, dropout=char_dropout, after_norm=char_after_norm,
                                              attn_type=char_attn_type, pos_embed=char_pos_embed, scale=char_scale)
        self.fc = nn.Linear(char_emb_size, embed_size)

        self._embed_size = embed_size

        self.requires_grad = requires_grad

    def forward(self, words):
        """
        输入words的index后，生成对应的words的表示。

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars = self.words_to_chars_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        # 为mask的地方为1
        chars_masks = chars.eq(self.char_pad_index)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        char_embeds = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        char_embeds = self.dropout(char_embeds)
        reshaped_chars = char_embeds.reshape(batch_size * max_len, max_word_len, -1)

        trans_chars = self.transformer(reshaped_chars, chars_masks.eq(0).reshape(-1, max_word_len))
        trans_chars = trans_chars.reshape(batch_size, max_len, max_word_len, -1)
        trans_chars = self.activation(trans_chars)
        if self.pool_method == 'max':
            trans_chars = trans_chars.masked_fill(chars_masks.unsqueeze(-1), float('-inf'))
            chars, _ = torch.max(trans_chars, dim=-2)  # batch_size x max_len x H
        else:
            trans_chars = trans_chars.masked_fill(chars_masks.unsqueeze(-1), 0)
            chars = torch.sum(trans_chars, dim=-2) / chars_masks.eq(0).sum(dim=-1, keepdim=True).float()

        chars = self.fc(chars)

        return self.dropout(chars)