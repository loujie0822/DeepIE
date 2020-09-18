# /usr/bin/env python
# encoding=utf8
# for chip2020 entity extractor

"""
1. 对brat标注的ann格式结果转换为chip2020 text 描述
2. 对chip2020描述的样本数据，生成ann待标注数据；
uuid = md5(text)
"""


import json
import os
import sys
import hashlib
import click
from tqdm import tqdm


@click.command(name='trans format between ann and json')
@click.option('--type', type=click.Choice(['ann2json', 'json2ann']), default='ann2json', help='ann2json or json2ann')
@click.argument('ann_path', type=click.Path())
@click.argument('jfile', type=click.Path())
def main(type, ann_path, jfile):
    if type == 'ann2json':
        trans_ann2json(ann_path, jfile)

    elif type == 'json2ann':
        trans_json2ann(ann_path, jfile)


def trans_ann2json(ann_path, jfile):
    """
    按新的格式重新标注数据 output_format
    """
    assert os.path.exists(ann_path)

    md = hashlib.md5()
    result = []
    #for fname in tqdm(filter(lambda k: k.endswith('ann'), os.listdir(ann_path)),
    #                  total=len(os.listdir(ann_path))/2, desc='sample labeled: '):
    total = int(len(os.listdir(ann_path)) / 2)
    for fid in tqdm(range(total), total=total, desc='sample labeled: '):
        fid += 1
        fname = os.path.join(ann_path, '{}'.format(fid))
        elem = dict()
        text = open(fname+'.txt').read()
        elem['text'] = text
        md.update(text.encode('utf8'))
        elem['uuid'] = md.hexdigest()
        elem['filename'] = fname + '.txt'
        weizhi = ann_labels(fname+'.ann', text)
        elem['weizhi'] = weizhi

        result.append(elem)

    #s = json.dumps(result, ensure_ascii=False, indent=2)
    #open(jfile, mode='w').write(s)

    txt = json2chip2020(result)
    open(jfile, mode='w').write(txt)


def json2chip2020(jdata):
    lines = []
    for elem in jdata:
        line = elem['text']

        for wz in elem['weizhi']:
            pos = wz['pos']
            label = wz['label']
            s, e = int(pos[0]), int(pos[1])-1
            line = '{}|||{}    {}    {}'.format(line, s, e, label)

        if line.find('|||') != -1:
            line += '|||'
        lines.append(line)
    s = '\n'.join(lines)
    return s + '\n'


def ann_labels(annfile, text):
    """
    ann format:
    Tid \t NER_name start end;start end \t data \n
    Rid \t REL_name Arg1:Tid Arg2:Tid \n
    1. 只做实体类转换 T
    2. 再做关系类转换 R todo
    chip2020: text 每行一个样本 ||| 分割标注
    术前禁食4～6小时，以防术中、术后呕吐窒息。|||17    20    sym|||19    20    sym|||
    """
    assert os.path.exists(annfile)
    weizhi = []
    for line in open(annfile):
        elem = dict()
        line = line.strip()

        # 当前只处理Entity Tag, Relation Tag todo...
        if not line.startswith('T'):
            print('[Ann not Entity Tag]: {}'.format(line))
            continue
        terms = line.split('\t')
        if len(terms) != 3:
            print('[Ann Entity Tag Error]: {}'.format(line))
            continue

        tid = terms[0]
        # 这里如果跨行的话\n在data中会被空格替换掉 注意!
        label_pos = terms[1].split(' ')
        elem['tid'] = tid
        elem['label'] = label_pos[0]
        elem['pos'] = [label_pos[1], label_pos[-1]]
        #elem['data'] = terms[2]
        # 这里把ann标注中的换行-空格又转换为了原始换行符
        elem['data'] = text[int(label_pos[1]):int(label_pos[-1])]
        weizhi.append(elem)
    return weizhi


def trans_json2ann(ann_path, jfile):
    """
    ann format: 注意如果实体中间有\n的话，在ann标注中data用空格替换，\n位置前后用;分割为多段
    Tid \t NER_name start end;start end \t data \n
    Rid \t REL_name Arg1:Tid Arg2:Tid \n
    1. 只做实体类转换 T
    2. 再做关系类转换 R todo

    json中的每个样本对应到ann中是一个文件, 写入到ann_path
    chip2020: text 每行一个样本 ||| 分割标注
    术前禁食4～6小时，以防术中、术后呕吐窒息。|||17    20    sym|||19    20    sym|||
    """
    assert os.path.exists(jfile)
    if not os.path.isdir(ann_path):
        os.mkdir(ann_path)

    tid = 0
    for line in tqdm(open(jfile, mode='r'), desc='sample anns: '):
        tid += 1
        terms = line.strip().split('|||')
        text = terms[0]
        weizhi = []
        for elem in terms[1:]:
            if len(elem) == 0:
                continue
            s, e, tag = elem.split(' ' * 4)
            data = text[int(s):int(e)+1]
            wz = {'label': tag, 'data': data, 'pos':[int(s), int(e)+1]}
            weizhi.append(wz)

        tfname = os.path.join(ann_path, '{}.txt'.format(tid))
        open(tfname, 'w').write(text)

        anns = labels_ann(weizhi)
        afname = os.path.join(ann_path, '{}.ann'.format(tid))
        open(afname, 'w').write(anns)


def labels_ann(weizhi):
    anns = []
    tid = 1
    for elem in weizhi:
        label = elem['label']
        tag_pos = '{} {}'.format(elem['pos'][0], elem['pos'][1])
        data = elem['data']
        # 这里假定最多只存在一个\n， 实际上是不严谨的 ...
        pos_split = data.find('\n')
        if pos_split != -1:
            tag_pos = '{};{}'.format(elem['pos'][0], pos_split, pos_split+1, elem['pos'][1])
            data.replace('\n', ' ')

        tag = 'T{}\t{} {}\t{}'.format(tid, label, tag_pos, data)
        anns.append(tag)
        tid += 1

    return '\n'.join(anns)


if __name__ == '__main__':
    """ using case
    # 从chip2020 text导入到ann目录下待标
    python tools/brat_ann_chip2020.py --type=json2ann ann_path test1.txt
      
    # 从标注好的数据导出为text
    python tools/brat_ann_chip2020.py --type=ann2json ann_path test1.ann  

    """
    main()

