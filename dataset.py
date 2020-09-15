from collections import namedtuple
from torch.utils.data import Dataset
# from transformers import BatchEncoding
from common.utils import load_bin, dump_to_bin
import pickle
import torch

HotpotDocument = namedtuple('HotpotDocument', ['title', 'doc_input_ids', 'label'])
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
    assert batch_size == 1
    NUM_DOCUMENT_PER_EXAMPLE = len(data[0].documents)
    batch_encoding = []
    batch_labels = []
    for ex in data:
        q_ids = ex.question_input_ids
        ex_labels = [(1 if d.label > 0 else 0) for d in ex.documents]
        batch_labels.append(ex_labels)
        for d in ex.documents:

            # truncate here
            pair_encoding = tokenizer.prepare_for_model(q_ids, d.doc_input_ids, truncation=True)
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
        padded_encodings['labels'] = torch.LongTensor(batch_labels)
    return padded_encodings


# references
# def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
#     """
#     Very simple data collator that:
#     - simply collates batches of dict-like objects
#     - Performs special handling for potential keys named:
#         - ``label``: handles a single value (int or float) per object
#         - ``label_ids``: handles a list of values per object
#     - does not do any additional preprocessing

#     i.e., Property names of the input object will be used as corresponding inputs to the model.
#     See glue and ner for example of how it's useful.
#     """

#     # In this function we'll make the assumption that all `features` in the batch
#     # have the same attributes.
#     # So we will look at the first element as a proxy for what attributes exist
#     # on the whole batch.
#     if not isinstance(features[0], (dict, BatchEncoding)):
#         features = [vars(f) for f in features]

#     first = features[0]
#     batch = {}

#     # Special handling for labels.
#     # Ensure that tensor is created with the correct type
#     # (it should be automatically the case, but let's make sure of it.)
#     if "label" in first and first["label"] is not None:
#         label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
#         dtype = torch.long if isinstance(label, int) else torch.float
#         batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
#     elif "label_ids" in first and first["label_ids"] is not None:
#         if isinstance(first["label_ids"], torch.Tensor):
#             batch["labels"] = torch.stack([f["label_ids"] for f in features])
#         else:
#             dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
#             batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

#     # Handling of all other possible keys.
#     # Again, we will use the first element to figure out which key/values are not None for this model.
#     for k, v in first.items():
#         if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
#             if isinstance(v, torch.Tensor):
#                 batch[k] = torch.stack([f[k] for f in features])
#             else:
#                 batch[k] = torch.tensor([f[k] for f in features])

#     return batch