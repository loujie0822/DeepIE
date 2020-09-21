# /usr/bin/env python
# encoding=utf8
# for cmkeg relations crawling
# data source: ~/data/CMeKG


import json
import os
import sys
import hashlib
import click
from tqdm import tqdm
import requests
import time


def config():
    conf = {
        'jibing': '/Users/liuhui/data/CMeKG/entity/jibing.json',
        'yaowu': '/Users/liuhui/data/CMeKG/entity/yaowu.json',
        'zhengzhuang': '/Users/liuhui/data/CMeKG/entity/zhengzhuang.json',
        'zhenliao': '/Users/liuhui/data/CMeKG/entity/zhenliao.json',
    }
    knowledge_url = 'https://zstp.pcl.ac.cn:8002'
    return conf, knowledge_url


def request_url(url):
    try:
        data = None
        respose = requests.get(url=url)
        if respose.status_code == 200:
            data = respose.text
            if not json.loads(data).get('link', []):
                data = None
        return data
    except requests.RequestException:
        return None


def main(output):
    conf, kg_url = config()
    #crawl_type = conf.keys()
    crawl_type = ['zhengzhuang', 'zhenliao']
    for type in crawl_type:
        output_new = '{}_{}'.format(output, type)
        outfd = open(output_new, mode='w')
        fpath = conf[type]
        assert os.path.exists(fpath)
        nodes = json.load(open(fpath))['nodes']
        cnt = 0
        for node in tqdm(nodes, total=len(nodes), desc='Process: '):
            name = node['name']
            url = '{}/knowledge?tree_type={}&name={}'.format(kg_url, type, name)
            key = '{}_{}'.format(type, name)
            res_data = request_url(url)
            if res_data is not None:
                line = json.dumps((key, res_data))
                outfd.write(line+'\n')
            else:
                print('{}: {} no relation'.format(type, name))

            if cnt % 50 == 0:
                time.sleep(2)

            cnt += 1


if __name__ == '__main__':
    output = '/Users/liuhui/data/CMeKG/relations/kg.txt'
    main(output)

