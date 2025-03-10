#!/usr/bin/env python3
# -*- coding: utf8 -*-

import json
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('stopwords')
stopwords = set(sw.words('norwegian')) | set(sw.words('english'))


def main(i, m, o):
    print('Reading meta data file', m)
    meta_df = pd.read_csv(m, index_col='id')
    print('Merging event files')
    events = []
    t = tqdm(os.listdir(i))
    for f in t:
        t.set_description(f)
        with(open(os.path.join(i, f))) as f:
            for line in f:
                e = json.loads(line)
                event_id = e['documentId']
                if event_id in meta_df.index:
                    keywords, title, createtime = meta_df.loc[event_id]
                    e['document'] = {
                        'keywords': keywords,
                        'title': title,
                        'createtime': createtime
                    }
                    tokens = [word.lower() for word in word_tokenize(title)]
                    kws = (keywords
                           if type(keywords) == str
                           else '').lower().split(',')
                    e['words'] = [word
                                  for word in tokens + kws
                                  if len(re.sub(r"[\.'\-_:0-9]", '', word)) > 0 and word not in stopwords]
                    events.append(e)
    count = len(events)
    print('Sorting', count, 'events by timestamp')
    events.sort(key=lambda o: o['time'])
    print('Events are from', events[0]['time'], 'to', events[-1]['time'])
    if o is None:
        print('Dry-run, not writing events to disk')
    else:
        path = os.path.join(o, 'events.csv')
        print('Writing to', path)
        os.makedirs(o, exist_ok=True)
        with open(path, 'w') as f:
            for event in tqdm(events):
                f.write(json.dumps(event))
                f.write('\n')
    split = int(0.8 * count)
    train, test = events[:split], events[split:]
    print(len(train), 'training events and', len(test), 'test events')
    extract_users(train, os.path.join(o, 'train') if o is not None else None)
    extract_users(test, os.path.join(o, 'test') if o is not None else None)
    print('Success!', o)


def extract_users(data, directory):
    if directory is not None:
        print('Extracting users into', directory)
    print('Grouping by users')
    d = {}
    for event in tqdm(data):
        u = event['userId']
        if u not in d:
            d[u] = []
        d[u].append(event)
    if directory is None:
        print('Dry-run, not writing user-specific events to disk')
    else:
        print('Creating', len(d), 'user-specific files')
        os.makedirs(directory, exist_ok=True)
        for user in tqdm(d):
            path = os.path.join(directory, user.replace(':', '_'))
            with open(path, 'w') as f:
                for event in d[user]:
                    f.write(json.dumps(event))
                    f.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('events',
                        help='Input directory, should contain 90 files')
    parser.add_argument('meta',
                        help='Titles and keywords, output file generated by `to_csv.py`')
    parser.add_argument('-o', '--output',
                        help='Output directory, omit to perform dry-run')

    args = parser.parse_args()

    in_dir = args.events
    meta_file = args.meta
    out_file = args.output

    main(in_dir, meta_file, out_file)
