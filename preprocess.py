import argparse
from transformers import AutoTokenizer
from common.utils import *
from os.path import join
from tqdm import tqdm
import sys
from dataset import *
import itertools

# dump all the features
def _parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    return args

def preprocess_document(ex, tokenzier, supporting_titles=None, answer=None, answer_type=None):
    title = ex[0]
    sentences = ex[1]
    # for s in paragraph:
    sent_input_ids = [tokenzier.convert_tokens_to_ids(tokenzier.tokenize(s)) for s in sentences]
    doc_input_ids = list(itertools.chain.from_iterable(sent_input_ids))
    if supporting_titles is not None:
        if title in supporting_titles:
            label = 1
            if answer_type == 'span':
                if any([(answer in x) for x in sentences]):
                    label = 2
        else:
            label = 0
    else:
        label = -1
    doc_input_ids = doc_input_ids[:tokenzier.model_max_length]
    return HotpotDocument(title, doc_input_ids, label)
        

def preprocess_train_example(raw_data, tokenzier, args):
    qid = raw_data['_id']
    question = raw_data['question']
    question_input_ids = tokenzier.convert_tokens_to_ids(tokenzier.tokenize(question))

    supporting_facts = raw_data['supporting_facts']
    supporting_titles = [x[0] for x in supporting_facts]
    documents = [preprocess_document(x, tokenzier, supporting_titles) for x in raw_data['context']]
    answer = raw_data['answer']
    answer_type = 'choice' if answer in ['yes', 'no'] else 'span'
    type = raw_data['type']
    level = raw_data['level']

    # sanity check
    # print(supporting_titles)
    assert sum([x.label > 0 for x in documents]) == 2

    return HotpotExample(qid, question, question_input_ids, documents, answer=answer, answer_type=answer_type, type=type, level=level)

def preprocess_eval_example(raw_data, tokenzier, args):
    pass

def preprocess_split(raw_fname, tokenzier, args, is_train=False):
    # read json
    raw_dataset = read_json(raw_fname)
    if args.do_mini:
        raw_dataset = raw_dataset[:128]

    
    proc_dataset = []
    for raw_data in tqdm(raw_dataset, desc='Preprocessing', file=sys.stdout, total=len(raw_dataset)):
        if is_train:
            proc_data = preprocess_train_example(raw_data, tokenzier, args)
        else:
            proc_data = preprocess_eval_example()
        proc_dataset.append(proc_data)
    return proc_dataset

    
def main():
    args = _parse_args()
    tokenzier = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    dataset_prefix = 'datasets/hotpot'
    train_fname = join(dataset_prefix, 'hotpot_train_v1.1.json')
    dev_fname = join(dataset_prefix, 'hotpot_dev_distractor_v1.json')
    # test_fname = join(dataset_prefix, 'hotpot_train_v1.1.json')
    
    train_set = preprocess_split(train_fname, tokenzier, args, True)
    dev_set = preprocess_split(dev_fname, tokenzier, args, True)
    
    output_prefix = 'outputs/'

    train_outfile = 'train_dataset.bin'
    dev_outfile = 'dev_dataset.bin'
    if args.do_mini:
        train_outfile = 'mini_' + train_outfile
        dev_outfile = 'mini_' + dev_outfile
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)

    dump_to_bin(train_set, train_outfile)
    dump_to_bin(dev_set, dev_outfile)

if __name__ == "__main__":
    main()