#!/usr/bin/env python3


import csv
from heapq import nlargest


def load_vocab(vocab_path):
    with open(vocab_path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_tree(tree_csv_path):
    csv.field_size_limit(int(2**31 - 1))
    with open(tree_csv_path) as f:
        nodes = dict(
            (
                tuple(int(x) for x in row['me'].strip().split()),
                dict(parent_loc=tuple(int(x) for x in row['parent'].strip().split()),
                     cnt=float(row['cnt'].strip()),
                     beta_cnt=[float(x) for x in row['beta_cnt'].strip().split()],
                     children={})
            )
            for row in csv.DictReader(f)
        )

    root_loc = (1,)
    if root_loc not in nodes:
        nodes[root_loc] = dict(parent_loc=(), children={})

    for (node_loc, node) in nodes.items():
        parent_loc = node['parent_loc']
        if parent_loc:
            child_idx = node_loc[-1]
            nodes[parent_loc]['children'][child_idx] = node

    for (node_loc, node) in nodes.items():
        node['me'] = node_loc
        node['children'] = [child for (_, child) in sorted(node['children'].items())]

    return nodes[root_loc]


def print_tree(tree_csv_path, vocab_path, num_words=10):
    tree = load_tree(tree_csv_path)
    vocab = load_vocab(vocab_path)

    node_stack = [tree]
    while node_stack:
        node = node_stack.pop()

        top_word_pairs = nlargest(num_words, zip(node.get('beta_cnt', []), vocab))
        print(node['me'], 10 * ' ', ' '.join(word for (_, word) in top_word_pairs))

        for child in reversed(node['children']):
            node_stack.append(child)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Load tree from CSV file and print topics.')
    parser.add_argument('tree_csv_path')
    parser.add_argument('vocab_path')
    parser.add_argument('--num-words', type=int, default=10)
    args = parser.parse_args()
    print_tree(args.tree_csv_path, args.vocab_path, args.num_words)


if __name__ == '__main__':
    main()
