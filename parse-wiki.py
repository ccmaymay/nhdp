#!/usr/bin/env python3


from collections import Counter

from tqdm import tqdm


UNK = '<unk>'


def parse_line(line, vocab, doc):
    tokens = line.strip().lower().split()
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
        word_idx = vocab[token]
        doc[word_idx] += 1


def parse_wikitext(input_path, output_path, vocab_path):
    vocab = {UNK: 0}
    doc = None
    docs = []

    num_lines = 0
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            num_lines += 1

    pb = tqdm(desc='Reading lines', total=num_lines)
    with open(input_path, encoding='utf-8') as f:
        prev_prev_line = None
        prev_line = None
        for line in f:
            line = line.rstrip()

            if prev_line is not None and prev_prev_line is not None:
                if prev_line.startswith(' = ') and not prev_prev_line and not line:
                    # prev_line was a section header like:
                    #
                    #  = = = Section = = = 
                    #

                    if not prev_line.startswith(' = = '):
                        # prev_line was a top-level section header (document title) like:
                        #
                        #  = Title = 
                        #

                        # Start a new document
                        doc = Counter()
                        docs.append(doc)

                parse_line(prev_line, vocab, doc)

            prev_prev_line = prev_line
            prev_line = line

            pb.update(1)

        if prev_line is not None and prev_prev_line is not None:
            parse_line(prev_line, vocab, doc)

        pb.close()

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for (word, _) in tqdm(sorted(vocab.items(), key=lambda p: p[1]), desc='Writing vocab'):
            print(word, file=f)

    pb = tqdm(desc='Writing docs', total=len(docs))
    with open(output_path, 'w') as f:
        for (doc_idx, doc) in enumerate(docs):
            for (word_idx, count) in sorted(doc.items()):
                print('{} {} {}'.format(doc_idx, word_idx, count), file=f)

            pb.update(1)

        pb.close()


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Parse wikitext token data into nHDP format.')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('vocab_path')
    args = parser.parse_args()
    parse_wikitext(args.input_path, args.output_path, args.vocab_path)


if __name__ == '__main__':
    main()
