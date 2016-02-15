from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from redis import Redis
from tube.corpus import Vocab
from tube.util import parse_redis_addr
from brightside.postproc.util import argpartition
import numpy as np


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_addr', type=parse_redis_addr)
    parser.add_argument('vocab_key', type=str)
    parser.add_argument('topics_path', type=str)
    ns = parser.parse_args()
    input_db = Redis(*ns.input_addr)
    vocab = Vocab.load_redis(input_db, ns.vocab_key)
    with open(ns.topics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                (prefix, ids_str) = line.split(':')
                ids = map(lambda x: int(x)-1, ids_str.split())
                print u'%s: %s' % (prefix, u' '.join(vocab[i] for i in ids))


if __name__ == '__main__':
    main()
