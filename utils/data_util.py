import torch


def padding(seqs, is_float=False, batch_first=False):
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)

    seq_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(float(0)) if is_float \
        else torch.LongTensor(batch_length, len(seqs)).fill_(0)

    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])

    if batch_first:
        seq_tensor = seq_tensor.t()

    return seq_tensor, lengths


def mpn_padding(seqs, label, class_num, is_float=False, use_bert=False):
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)

    o1_tensor = torch.FloatTensor(len(seqs), batch_length, class_num).fill_(float(0)) if is_float \
        else torch.LongTensor(len(seqs), batch_length, class_num).fill_(0)
    o2_tensor = torch.FloatTensor(len(seqs), batch_length, class_num).fill_(float(0)) if is_float \
        else torch.LongTensor(len(seqs), batch_length, class_num).fill_(0)
    for i, label_ in enumerate(label):
        for attr in label_:
            if use_bert:
                o1_tensor[i, attr.value_pos_start + 1, attr.attr_type_id] = 1
                o2_tensor[i, attr.value_pos_end, attr.attr_type_id] = 1
            else:
                o1_tensor[i, attr.value_pos_start, attr.attr_type_id] = 1
                o2_tensor[i, attr.value_pos_end - 1, attr.attr_type_id] = 1

    return o1_tensor, o2_tensor


def _handle_pos_limit(pos, limit=30):
    for i, p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]
