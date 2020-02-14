# _*_ coding:utf-8 _*_
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


class VanillaAttention(nn.Module):
    """
        VanillaAttention
    """

    def __init__(self, p):
        super(VanillaAttention, self).__init__()

        self.dropout = nn.Dropout(p)
        self.mask = None

    def forward(self, q, k, v, mask=None):
        dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())

        assert (b_k == b_v)  # batch size should be equal
        assert (t_k == t_v)  # times should be equal

        qk = torch.matmul(k, q.unsqueeze(1)).squeeze(2)
        # qk.div_(dim_k ** 0.5)
        qk.masked_fill_(mask, -1e30)
        qk = F.softmax(qk, 1)

        return torch.bmm(qk.unsqueeze(1), v).squeeze(1)  # b,n


def get_attn_padding_mask(seq_q, seq_k):
    """
        Indicate the padding-related part to mask

    :param seq_q:
    :param seq_k:
    :return:
    """

    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)  # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


class BiAttention(nn.Module):
    """
    BI-DIRECTIONAL ATTENTION FLOW in BiDAF model

    """

    def __init__(self, dim):
        super(BiAttention, self).__init__()
        self.linear = nn.Linear(3 * dim, 1)

    def forward(self, x1, x1_mask, x2, x2_mask):
        """
        :param x1: b x n x d
        :param x2: b x m x d
        :param x1_mask: b x n
        :param x2_mask: b x m

        ####### caculation to similarity matrix #######
        这里计算的是 二维层面的相似度 是 字 和 字 之间的相似度
        S(n,m) = alpha(x1,x2) = W [x1, x2, x1*x2]

        ####### context to query attention  #######
        # (b,n,m) = (b,1,m) --> (b,n,m)
        # x2_mask (n,m)
        #    我 爱 你 空 空
        # 我 1  1  1  0  0
        # 爱 1  1  1  0  0
        # 北 1  1  1  0  0
        # 京 1  1  1  0  0
        # 人 1  1  1  0  0
        # 空 1  1  1  0  0
        # 空 1  1  1  0  0

        ####### query to context attention  #######

        ******
        这里q2c是现将context对齐到query，然后再重复一次c2q，将query对齐到context
        ******

        # (b,n,m) = (b,n,1) --> (b,n,m)
        #    我 爱 你 空 空
        # 我 1  1  1  1  1
        # 爱 1  1  1  1  1
        # 北 1  1  1  1  1
        # 京 1  1  1  1  1
        # 人 1  1  1  1  1
        # 空 0  0  0  0  0
        # 空 0  0  0  0  0
        """
        # (b,n,m,d) = (b,n,d) --> (b,n,1,d) --> (b,n,m,d)
        x1_aug = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), x2.size(1), x1.size(2))

        # (b,n,m,d) = (b,m,d) --> (b,1,m,d) --> (b,n,m,d)
        x2_aug = x2.unsqueeze(1).expand(x1.size(0), x1.size(1), x2.size(1), x2.size(2))

        # (b,n,m,3d)
        x_input = torch.cat([x1_aug, x2_aug, x1_aug * x2_aug], dim=3)

        # (b,n,m) = (b,n,m,3d) --> (b,n,m,1) --> (b,n,m)
        similarity = self.linear(x_input).squeeze(3)

        ####### context to query attention  #######
        # (b,n,m) = (b,1,m) --> (b,n,m)
        # x2_mask (n,m)
        #    我 爱 你 空 空
        # 我 1  1  1  0  0
        # 爱 1  1  1  0  0
        # 北 1  1  1  0  0
        # 京 1  1  1  0  0
        # 人 1  1  1  0  0
        # 空 1  1  1  0  0
        # 空 1  1  1  0  0
        # x2_mask = x2_mask[:, :x2.size(1)]
        # x1_mask = x1_mask[:, :x1.size(1)]

        # a_non_pad = x1_mask.ne(1).type(torch.float).unsqueeze(-1)
        # q2c_non_pad = x2_mask.ne(1).type(torch.float).unsqueeze(-1)

        x2_mask = x2_mask.unsqueeze(1).expand_as(similarity)
        # (b,n,m)
        similarity.data.masked_fill_(x2_mask.data, -2e20)
        # (b,n,m)
        sim_row = F.softmax(similarity, dim=2)
        # (b,n,d) = (b,n,m) *  (b,m,d)
        c2q_att = sim_row.bmm(x2)

        # attn_a = a_non_pad * attn_a

        ####### query to context attention  #######
        # (b,n,m) = (b,n,1) --> (b,n,m)
        #    我 爱 你 空 空
        # 我 1  1  1  1  1
        # 爱 1  1  1  1  1
        # 北 1  1  1  1  1
        # 京 1  1  1  1  1
        # 人 1  1  1  1  1
        # 空 0  0  0  0  0
        # 空 0  0  0  0  0
        x1_mask = x1_mask.unsqueeze(2).expand_as(similarity)
        # (b,n,m) = (b,n,1) --> (b,n,m)
        # TODO: 检查此时的similarity（其实已经不重要了）
        similarity.data.masked_fill_(x1_mask.data, -2e20)
        sim_col = F.softmax(similarity, dim=1)
        # (b,m,d) = (b,m,n) *  (b,n,d)
        q2c = sim_col.transpose(1, 2).bmm(x1)
        # q2c = q2c_non_pad * q2c

        # (b,n,d) = (b,n,m) *  (b,m,d)
        # 这里q2c是现将context对齐到query，然后再重复一次c2q，将query对齐到context

        q2c_att = sim_row.bmm(q2c)
        # attn_b = a_non_pad * attn_b

        return torch.cat([x1, c2q_att, x1 * c2q_att, x1 * q2c_att], dim=-1)


class SDPAttention(nn.Module):
    """
        Scaled Dot-Product Attention from TransFormer
        ####### self attention  #######
        这里给出的self attention 最简单的实现，在输入维度q=k,没有通过linear
        #    我 爱 你 北  京 人 空  空
        # 我 1  1  1  1  1  1  0  0
        # 爱 1  1  1  1  1  1  0  0
        # 北 1  1  1  1  1  1  0  0
        # 京 1  1  1  1  1  1  0  0
        # 人 1  1  1  1  1  1  0  0
        # 空 0  0  0  0  0  0  0  0
        # 空 0  0  0  0  0  0  0  0
    """

    def __init__(self, p=0.1):
        super(SDPAttention, self).__init__()
        self.dropout = nn.Dropout(p)
        self.mask = None

    def set_mask(self, masked):
        # applies a mask of b x tq length
        self.mask = masked

    def forward(self, q, k, v):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, dim_v = list(v.size())

        assert (b_q == b_k and b_k == b_v)  # batch size should be equal
        assert (dim_q == dim_k)  # dims should be equal
        assert (t_k == t_v)  # times should be equal

        """
        similrity matrix caculation        
        sm_qk = {QK.t()/sqrt(512/8)}
        """
        qk = torch.bmm(q, k.transpose(1, 2))  # b x t_q x t_k
        qk.div_(dim_k ** 0.5)
        qk.masked_fill_(self.mask, -1e9)
        sm_qk = self.dropout(F.softmax(qk, 2))

        return torch.bmm(sm_qk, v), sm_qk  # b x t_q x dim_v


class MultiHeadAttention(nn.Module):
    """
        Scaled Dot-Product Attention
    """

    def __init__(self, embed_dim=None, num_heads=None, dropout=0.1):

        super(MultiHeadAttention, self).__init__()

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.input_size = self.head_dim * num_heads
        self.output_size = self.head_dim * num_heads
        self.linear_q = nn.Linear(self.input_size, self.input_size)
        self.linear_k = nn.Linear(self.input_size, self.input_size)
        self.linear_v = nn.Linear(self.input_size, self.input_size)
        self.linear_out = nn.Linear(self.input_size, self.output_size)
        self.sdp_attention = SDPAttention(p=dropout)
        self.relu = nn.ReLU()
        self.dp_out = nn.Dropout()

    def set_mask_sdp(self, masked):
        self.sdp_attention.set_mask(masked)

    def forward(self, q, k, v, attn_mask=None):

        if attn_mask is not None:
            self.set_mask_sdp(attn_mask)

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        # use extra dim_size and chunk to simplify the repeat process
        qw = qw.chunk(self.num_heads, 2)
        kw = kw.chunk(self.num_heads, 2)
        vw = vw.chunk(self.num_heads, 2)

        output = []
        attention_scores = []
        for i in range(self.num_heads):
            out_h, score = self.sdp_attention(qw[i], kw[i], vw[i])
            output.append(out_h)
            attention_scores.append(score)

        output = torch.cat(output, 2)

        return self.linear_out(self.dp_out(self.relu(output)))


class AttentionLayer(nn.Module):
    """
    Params:
      num_units: Number of units used in the attention layer
    """

    def __init__(self, query_size, key_size, value_size=None, mode='bahdanau',
                 normalize=False, dropout=0, batch_first=False,
                 output_transform=True, output_nonlinearity='tanh', output_size=None):
        super(AttentionLayer, self).__init__()
        assert mode == 'bahdanau' or mode == 'dot_prod'
        value_size = value_size or key_size  # Usually key and values are the same
        self.mode = mode
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.normalize = normalize
        if mode == 'bahdanau':
            self.linear_att = nn.Linear(key_size, 1)
            if normalize:
                self.linear_att = nn.utils.weight_norm(self.linear_att)
        if output_transform:
            output_size = output_size or query_size
            self.linear_out = nn.Linear(query_size + key_size, output_size)
            self.output_size = output_size
        else:
            self.output_size = value_size
        self.linear_q = nn.Linear(query_size, key_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.output_nonlinearity = output_nonlinearity
        self.mask = None

    def set_mask(self, mask):
        # applies a mask of b x t length
        self.mask = mask
        if mask is not None and not self.batch_first:
            self.mask = self.mask.t()

    def calc_score(self, att_query, att_keys):
        """
            att_query is: b x t_q x n
            att_keys is b x t_k x n
            return b x t_q x t_k scores
        """

        b, t_k, n = list(att_keys.size())
        t_q = att_query.size(1)
        if self.mode == 'bahdanau':
            att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
            att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
            sum_qk = att_query + att_keys
            sum_qk = sum_qk.view(b * t_k * t_q, n)
            out = self.linear_att(F.tanh(sum_qk)).view(b, t_q, t_k)
        elif self.mode == 'dot_prod':
            out = torch.bmm(att_query, att_keys.transpose(1, 2))
            if self.normalize:
                out.div_(n ** 0.5)
        return out

    def forward(self, query, keys, values=None):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if values is not None:
                values = values.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False
        values = keys if values is None else values

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)

        # Fully connected layers to transform query
        att_query = self.linear_q(query)

        scores = self.calc_score(att_query, keys)  # size b x t_q x t_k
        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            scores.masked_fill_(mask, -1e12)

        # Normalize the scores
        scores_normalized = F.softmax(scores, dim=2)

        # Calculate the weighted average of the attention inputs
        # according to the scores
        scores_normalized = self.dropout(scores_normalized)
        context = torch.bmm(scores_normalized, values)  # b x t_q x n

        if hasattr(self, 'linear_out'):
            context = self.linear_out(torch.cat([query, context], 2))
            if self.output_nonlinearity == 'tanh':
                context = F.tanh(context)
            elif self.output_nonlinearity == 'relu':
                context = F.relu(context, inplace=True)
        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)

        return context, scores_normalized


if __name__ == '__main__':
    feats = autograd.Variable(torch.LongTensor([range(10)] * 45))
    vocab_s = 8888
    embedding_d = 100
    repeat_t = 1
    interm_c = 10
    tagset_s = 18
    batch_s = 10
    seq_l = 45
    dilation_r = [1, 2, 4, 8, 16, 32, 64, 128]
    word_embeds = nn.Embedding(vocab_s, embedding_d)

    # emb = word_embeds(feats)
    # print emb.size()
    #
    # emb0 = emb[:, :5, :]
    # print emb0.size()
    #
    # ap = AttentionPooling(100, [100], 75, True)
    #
    # o = ap(emb, emb0, emb0)
    # print o
