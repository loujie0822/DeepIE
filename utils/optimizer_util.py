from torch import optim

from layers.encoders.transformers.bert.bert_optimization import BertAdam


def set_optimizer(args, model, train_steps=None):
    if args.warm_up:
        print('using BertAdam')
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=train_steps)
        return optimizer
    else:
        print('using optim Adam')
        parameters_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam(parameters_trainable, lr=args.learning_rate)
    return optimizer
