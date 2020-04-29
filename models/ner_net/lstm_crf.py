# _*_ coding:utf-8 _*_
import copy
import warnings

import torch
from torch import nn

from layers.decoders.crf import CRF
from layers.encoders.rnns.stacked_rnn import StackedBRNN

warnings.filterwarnings("ignore")


class SentenceEncoder(nn.Module):
    def __init__(self, args, embed_size):
        super(SentenceEncoder, self).__init__()
        rnn_type = nn.LSTM if args.rnn_encoder == 'lstm' else nn.GRU
        self.encoder = StackedBRNN(
            input_size=embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            dropout_output=True,
            concat_layers=False,
            rnn_type=rnn_type,
            padding=True
        )

    def forward(self, input, mask):
        return self.encoder(input, mask)


class NERNet(nn.Module):
    """
        NERNet : Lstm+CRF
    """

    def __init__(self, args, model_conf):
        super(NERNet, self).__init__()
        char_emb = model_conf['char_emb']
        bichar_emb = model_conf['bichar_emb']
        embed_size = args.char_emb_dim
        if char_emb is not None:
            self.char_emb = nn.Embedding.from_pretrained(char_emb, freeze=False, padding_idx=0)
            embed_size = char_emb.size()[1]
        else:
            vocab_size = len(model_conf['char_vocab'])
            self.char_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=args.char_emb_dim,
                                               padding_idx=0)
        self.bichar_emb = None
        if bichar_emb is not None:
            self.bichar_emb = nn.Embedding.from_pretrained(bichar_emb, freeze=False, padding_idx=0)
            embed_size += bichar_emb.size()[1]

        self.sentence_encoder = SentenceEncoder(args, embed_size)
        self.emission = nn.Linear(args.hidden_size * 2, len(model_conf['entity_type']))
        self.crf = CRF(len(model_conf['entity_type']), batch_first=True)

    def forward(self, char_id, bichar_id, label_ids=None, is_eval=False):
        # use anti-mask for answers-locator
        mask = char_id.eq(0)
        chars = self.char_emb(char_id)
        if self.bichar_emb is not None:
            bichars = self.bichar_emb(bichar_id)
            chars = torch.cat([chars, bichars], dim=-1)
        sen_encoded = self.sentence_encoder(chars, mask)

        bio_mask = char_id != 0
        emission = self.emission(sen_encoded)

        if not is_eval:
            crf_loss = -self.crf(emission, label_ids, mask=bio_mask, reduction='mean')
            return crf_loss
        else:
            pred = self.crf.decode(emissions=emission, mask=bio_mask)

            # TODO:check
            max_len = char_id.size(1)
            temp_tag = copy.deepcopy(pred)
            for line in temp_tag:
                line.extend([0] * (max_len - len(line)))
            ent_pre = torch.tensor(temp_tag).to(emission.device)
            return ent_pre