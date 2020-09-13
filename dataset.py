from collections import namedtuple
from torch.utils.data import Dataset

import pickle

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
    
    
