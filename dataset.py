from collections import namedtuple
from torch.utils.data import Dataset
# from transformers import BatchEncoding
from common.utils import load_bin, dump_to_bin
import pickle
import torch

HotpotDocument = namedtuple('HotpotDocument', ['title', 'doc_input_ids', 'label'])
NUM_DOCUMENT_PER_EXAMPLE = 10

class HotpotDocument:
    def __init__(self, title, doc_input_ids, label):
        self.title = title
        self.doc_input_ids = doc_input_ids
        self.label = label

class HotpotExample:
    def __init__(self, id, question, question_input_ids, documents, answer=None, answer_type=None, supporting_facts=None, type=None, level=None):
        # id
        # question
        # question_input_ids
        # answer
        # answer_type
        # type
        # level
        # documents
        self.id = id
        self.question = question
        self.question_input_ids = question_input_ids
        self.documents = documents
        self.answer = answer
        self.answer_type = answer_type
        self.supporting_facts = supporting_facts
        self.type = type
        self.level = level

class HotpotDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]
    
    @classmethod
    def from_bin_file(cls, fname):
        return cls(load_bin(fname))

def collate_fn_for_doc_cls(tokenizer, data, do_eval=False):
    # initial input
    # input_ids/attention_mask/token_type_ids: B * NUM_DOC * max_seq_len
    # labels: B * NUM_DOC
    batch_size = len(data)
    # print(batch_size)
    # assert batch_size == 1
    batch_encoding = []
    # sp label, containing supporting fact
    # ct label, containing answer
    batch_sp_labels = []
    batch_ct_labels = []
    batch_doc_masks = []
    for ex in data:

        q_ids = ex.question_input_ids
        n_doc = len(ex.documents)
        doc_masks = [(1 if i < n_doc else 0) for i in range(NUM_DOCUMENT_PER_EXAMPLE)]
        sp_labels = [(1 if d.label > 0 else 0) for d in ex.documents] + [0] * (NUM_DOCUMENT_PER_EXAMPLE - n_doc)
        ct_labels = [(1 if d.label == 2 else 0) for d in ex.documents] + [0] * (NUM_DOCUMENT_PER_EXAMPLE - n_doc)
        # batch_labels.append(ex_labels)
        
        batch_doc_masks.append(doc_masks)
        batch_sp_labels.append(sp_labels)
        batch_ct_labels.append(ct_labels)

        for d in ex.documents:
            # truncate here
            pair_encoding = tokenizer.prepare_for_model(q_ids, d.doc_input_ids, truncation=True)
            batch_encoding.append(pair_encoding)

        for _ in range(NUM_DOCUMENT_PER_EXAMPLE - n_doc):
            pair_encoding = tokenizer.prepare_for_model(q_ids, [], truncation=True)
            batch_encoding.append(pair_encoding)
            

    # if do_eval:
        # print([len(x['input_ids']) for x in batch_encoding])
    # truncted comes before padding
    padded_encodings = tokenizer.pad(batch_encoding, padding=True)
    
    padded_encodings['input_ids'] = torch.LongTensor( padded_encodings['input_ids']).view([batch_size, NUM_DOCUMENT_PER_EXAMPLE, -1])
    padded_encodings['attention_mask'] = torch.LongTensor(padded_encodings['attention_mask']).view([batch_size, NUM_DOCUMENT_PER_EXAMPLE, -1])
    if 'token_type_ids' in padded_encodings:
        padded_encodings['token_type_ids'] = torch.LongTensor(padded_encodings['token_type_ids']).view([batch_size, NUM_DOCUMENT_PER_EXAMPLE, -1])
    if do_eval:
        padded_encodings['doc_masks'] = torch.BoolTensor(batch_doc_masks)
        padded_encodings['sp_labels'] = torch.LongTensor(batch_sp_labels)
        padded_encodings['ct_labels'] = torch.LongTensor(batch_ct_labels)
    return padded_encodings
