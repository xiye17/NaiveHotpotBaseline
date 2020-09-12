import json
import argparse
from transformers import AutoTokenizer
from common.utils import add_common_args
from os.path import join

# dump all the features
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_mini', action='store_true', default=False, help='Test with mini dataset for debugging')    
    parser.add_argument('--model_name', type=str, default='roberta-large', help='Model choosen')
    add_common_args(parser)
    args = parser.parse_args()
    return args

def preprocess_split(raw_fname, tokenzier, args, is_train=False):
    # read json

def main():
    args = _parse_args()
    tokenzier = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    dataset_prefix = 'datasets/hotpot'
    train_fname = join(dataset_prefix, 'hotpot_train_v1.1.json')
    dev_fname = join(dataset_prefix, 'hotpot_dev_distractor.json')
    test_fname = join(dataset_prefix, 'hotpot_train_v1.1.json')


    
    

if __name__ == "__main__":
    main()