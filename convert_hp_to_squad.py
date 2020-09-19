    
import argparse
from transformers import AutoTokenizer
from common.utils import *
from os.path import join
from tqdm import tqdm
import sys
from dataset import *
import itertools
import re

def _parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    return args

def construct_context_and_title(documents, supporting_titles, answer_text):
    supporting_docs = [x for x in documents if x[0] in supporting_titles]
    supporting_docs = [(x[0], ''.join(x[1])) for x in supporting_docs]
    if not any([answer_text in x[1] for x in supporting_docs]):
        return None, None
    supporting_docs.sort(key=lambda x: answer_text in x[1], reverse=True)
    
    title = f'{supporting_docs[0][0]}, {supporting_docs[1][0]}'
    context = f'{supporting_docs[0][1]} {supporting_docs[1][1]}'
    return context, title

    

        

def construct_answers(context, answer):
    start_positions = [i.start() for i in re.finditer(answer, context)]
    return [{'answer_start': i, 'text': answer} for i in start_positions]
    

def preprocess_train_example(raw_data, tokenzier, args):
    if raw_data['answer'] in ['yes', 'no']:
        return None

    # we'll a have a single paragraph

    answer_text = raw_data['answer']

    supporting_facts = raw_data['supporting_facts']
    supporting_titles = [x[0] for x in supporting_facts]
    context, title = construct_context_and_title(raw_data['context'], supporting_titles, answer_text)
    if context is None:
        return None

    qa0 = {}
    qa0['id'] = raw_data['_id']
    qa0['question'] = raw_data['question']
    # sanity check
    # print(supporting_titles)
    answers = construct_answers(context, answer_text)
    if answers is None:
        return None
    qa0['answers'] = answers

    pargraph0 = {'context': context, 'qas': [qa0]}    
    data = {'title': title, 'paragraphs': [pargraph0]}

    return data


# top  level [data, version]
def preprocess_split(raw_fname, tokenzier, args, is_train=False):
    # read json
    raw_dataset = read_json(raw_fname)
    if args.do_mini:
        raw_dataset = raw_dataset[:32]

    data = {}
    data['version'] = '1.1'
    # data['data'] = 

    proc_dataset = []
    for raw_data in tqdm(raw_dataset, desc='Preprocessing', file=sys.stdout, total=len(raw_dataset)):
        if is_train:
            proc_data = preprocess_train_example(raw_data, tokenzier, args)
        else:
            proc_data = preprocess_eval_example()

        if proc_data is not None:
            proc_dataset.append(proc_data)

    data['data'] = proc_dataset
    return data

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

    train_outfile = 'train_hp_squad.json'
    dev_outfile = 'dev_hp_squad.json'
    if args.do_mini:
        train_outfile = 'mini_' + train_outfile
        dev_outfile = 'mini_' + dev_outfile
    train_outfile = join(output_prefix, train_outfile)
    dev_outfile = join(output_prefix, dev_outfile)

    dump_json(train_set, train_outfile)
    dump_json(dev_set, dev_outfile)

if __name__ == "__main__":
    main()
