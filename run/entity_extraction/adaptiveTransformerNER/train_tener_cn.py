import argparse

from torch import optim

from fastNLP import BucketSampler, SpanFPreRecMetric
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from fastNLP import cache_results
from fastNLP.embeddings import StaticEmbedding, BertEmbedding
from models.ner_net.tener import TENER
from models.ner_net.bert_tener import BERT_TENER

from run.entity_extraction.adaptiveTransformerNER.modules.callbacks import EvaluateCallback
from run.entity_extraction.adaptiveTransformerNER.modules.pipe import CNNERPipe

device = 0
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='medical', choices=['weibo', 'resume', 'ontonotes', 'msra'])
parser.add_argument('--encoder', type=str, default='transformer', choices=['lstm', 'transformer'])

args = parser.parse_args()

dataset = args.dataset
if dataset == 'resume':
    n_heads = 4
    head_dims = 64
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 50
elif dataset == 'weibo':
    n_heads = 4
    head_dims = 32
    num_layers = 1
    lr = 0.001
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'ontonotes':
    n_heads = 4
    head_dims = 48
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'msra':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'medical':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'test':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100

pos_embed = None

batch_size = 32
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True

dropout = 0.15
fc_dropout = 0.4

encoding_type = 'bmeso'
name = 'caches/{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)


@cache_results(name, _refresh=False)
def load_data():
    # 替换路径
    if dataset == 'ontonotes':
        paths = {'train': '../data/OntoNote4NER/train.char.bmes',
                 "dev": '../data/OntoNote4NER/dev.char.bmes',
                 "test": '../data/OntoNote4NER/test.char.bmes'}
        min_freq = 2
    elif dataset == 'weibo':
        paths = {'train': '../data/WeiboNER/train.all.bmes',
                 'dev': '../data/WeiboNER/dev.all.bmes',
                 'test': '../data/WeiboNER/test.all.bmes'}
        min_freq = 1
    elif dataset == 'resume':
        paths = {'train': '../data/ResumeNER/train.char.bmes',
                 'dev': '../data/ResumeNER/dev.char.bmes',
                 'test': '../data/ResumeNER/test.char.bmes'}
        min_freq = 1
    elif dataset == 'msra':
        paths = {'train': 'data/MSRANER/train.txt',
                 'dev': 'data/MSRANER/dev.txt',
                 'test': 'data/MSRANER/dev.txt'}
        min_freq = 2
    elif dataset == 'medical':
        paths = {'train': 'data/mediacal_data/train.txt',
                 'dev': 'data/mediacal_data/dev.txt',
                 'test': 'data/mediacal_data/dev.txt'}
        min_freq = 2
    elif dataset == 'test':
        paths = {'train': 'data/test_data/dev.txt',
                 'dev': 'data/test_data/dev.txt',
                 'test': 'data/test_data/dev.txt'}
        min_freq = 2
    data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)
    embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name='cpt/gigaword/uni.ite50.vec',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    bi_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'),
                               model_dir_or_name='cpt/gigaword/bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=min_freq,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    return data_bundle, embed, bi_embed


data_bundle, embed, bi_embed = load_data()

bert_embed = BertEmbedding(data_bundle.get_vocab('chars'), model_dir_or_name='transformer_cpt/bert',
                           requires_grad=False)

print(data_bundle)
model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
              d_model=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=dropout,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=bi_embed,bert_embed=bert_embed,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=attn_type == 'transformer')
# model = BERT_TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
#               d_model=d_model, n_head=n_heads,
#               feedforward_dim=dim_feedforward, dropout=dropout,
#               after_norm=after_norm, attn_type=attn_type,
#               bi_embed=bi_embed, bert_embed=bert_embed,
#               fc_dropout=fc_dropout,
#               pos_embed=pos_embed,
#               scale=attn_type == 'transformer')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=n_epochs, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=None)
trainer.train(load_best_model=False)
