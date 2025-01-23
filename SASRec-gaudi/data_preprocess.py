import os
import os.path
import gzip
import json
from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset


def preprocess_raw(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{fname}", trust_remote_code=True)

    dataset = dataset['full']
    # counting interactions for each user and item

    for l in tqdm(dataset):
        line += 1
        asin = l['asin']
        rev = l['user_id']
        time = l['timestamp']
        countU[rev] += 1
        countP[asin] += 1

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()


    for l in tqdm(dataset):
        line += 1
        asin = l['asin']
        rev = l['user_id']
        time = l['timestamp']

        threshold = 5

        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

    f = open(f'./data/processed/{fname}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()

    del dataset

def preprocess(fname):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"5core_last_out_{fname}", trust_remote_code=True)

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()

    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        f = open(f'./data/processed/{fname}_{t}.txt', 'w')
        for l in tqdm(d):
            user_id = l['user_id']
            asin = l['parent_asin']

            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid

            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            f.write('%d %d\n' % (usermap[user_id], itemmap[asin]))
        f.close()

    del dataset