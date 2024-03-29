import dataclasses
import logging
import os
from os.path import join
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from common.utils import dump_to_bin

import numpy as np

from transformers import AutoConfig, AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from models.ranker_trainer import doc_level_acc
from transformers import Trainer
from dataset import *

from sklearn.metrics import f1_score
from models.modeling_auto import AutoModelForPragraphRanking

from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class RankerModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default='roberta-base', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_dir: Optional[str] = field(
        default='outputs', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    do_mini: bool = field(default=False, metadata={"help": "Whether to use mini dataset for quick debugging"})

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default='hf_cache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def predict_supporting_facts(dataset, pred_outputs):
    (sp_preds, ct_predicts), (sp_labels, ct_labels, doc_masks), metrics = pred_outputs
    
    # sanity check
    for example, sp_label, ct_label, doc_mask in zip(dataset.examples, sp_labels, ct_labels, doc_masks):
        # sp_label = sp_label
        n_doc = np.sum(doc_mask)
        sp_label = sp_label[doc_mask]
        ct_label = ct_label[doc_mask]
        assert n_doc == len(example.documents)
        label_ = sp_label + ct_label
        label_ = label_.tolist()
        origin_label = [x.label for x in example.documents]
        assert label_ == origin_label
    
    num_hit = 0
    sp_dict = {}
    for example, sp_pred, doc_mask in zip(dataset.examples, sp_preds, doc_masks):
        # print(sp_pred.shape)
        sp_pred = sp_pred[:,1] - sp_pred[:,0]
        sp_pred = sp_pred[doc_mask]
        # print(sp_pred.shape)
        sorted_idx = np.argsort(sp_pred)
        picked = sorted_idx[-2:].tolist()
        picked.sort()

        gt_picked = [i for (i,x) in enumerate(example.documents) if x.label > 0]
        if picked == gt_picked:
            num_hit += 1
        sp_dict[example.id] = picked
        
    print('Hit Gold Support Facts:', num_hit / len(dataset))        
    return sp_dict
        
        
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((RankerModelArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForPragraphRanking.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # Get datasets
    train_file = join(model_args.data_dir, 'mini_train_dataset.bin' if model_args.do_mini else 'train_dataset.bin')
    dev_file = join(model_args.data_dir, 'mini_dev_dataset.bin' if model_args.do_mini else 'dev_dataset.bin')
    train_dataset = HotpotDataset.from_bin_file(train_file)
    dev_dataset = HotpotDataset.from_bin_file(dev_file)
    
    # doc_cls_collator = HotpotCollator()
    collate_fn = partial(collate_fn_for_doc_cls, tokenizer, do_eval=True)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=doc_level_acc,
        data_collator=collate_fn
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_dataset = dev_dataset
        trainer.compute_metrics = doc_level_acc
        trainer.data_collator=partial(collate_fn_for_doc_cls, tokenizer, do_eval=True)

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Predict ***")

        test_dataset = dev_dataset
        predictions = trainer.predict(test_dataset=test_dataset)
        trainer.compute_metrics = doc_level_acc
        trainer.data_collator=partial(collate_fn_for_doc_cls, tokenizer, do_eval=True)

        sp_dict = predict_supporting_facts(test_dataset, predictions)
        predict_file = join(model_args.data_dir, 'dev_ranker_preds.bin')
        dump_to_bin(sp_dict, predict_file)
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()