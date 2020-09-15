from transformers import BertPreTrainedModel, BertLayer, BertModel
from transformers import LongformerConfig, RobertaConfig, BertConfig, RobertaModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

_BERT_CONFIG_FOR_DOC = "BertConfig"
_BERT_TOKENIZER_FOR_DOC = "BertTokenizer"

_ROBERTA_CONFIG_FOR_DOC = "RobertaConfig"
_ROBERTA_TOKENIZER_FOR_DOC = "RobertaTokenizer"

# """Bert Model transformer for paragraph ranking via classification/regression head on top (a linear layer on top of
class BertForParagraphRankingCls(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.inter_document_layer = BertLayer(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # unwrap, for batching
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        
        # reshape
        # apply self attention

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class MLPClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForParagraphRankingCls(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.num_labels = config.num_labels
        
        # it can be a bert layer, or simly a self-attention-layer
        # let's do bert layer for now
        self.inter_document_layer = BertLayer(config)
        self.classifier = MLPClassificationHead(config)
        self.init_weights()

    # initial input
    # input_ids/attention_mask/token_type_ids: B * NUM_DOC * 512
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        NUM_DOCUMENT_PER_EXAMPLE = input_ids.size(1)
        # my gpu sucks. have to do this to fit my gpu size.
        debugging = False
        if debugging:
            NUM_DOCUMENT_PER_EXAMPLE = min(4, NUM_DOCUMENT_PER_EXAMPLE)
            input_ids = input_ids[:,:NUM_DOCUMENT_PER_EXAMPLE,:]
            attention_mask = attention_mask[:,:NUM_DOCUMENT_PER_EXAMPLE,:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:,:NUM_DOCUMENT_PER_EXAMPLE,:]
            labels = labels[:,:NUM_DOCUMENT_PER_EXAMPLE]
        
        # reshape, input_ids, attention_masks, token_type_ids
        input_ids = input_ids.view([batch_size * NUM_DOCUMENT_PER_EXAMPLE, -1])
        attention_mask = attention_mask.view([batch_size * NUM_DOCUMENT_PER_EXAMPLE, -1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view([batch_size * NUM_DOCUMENT_PER_EXAMPLE, -1])

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(outputs)
        _, pooled_output = outputs
        # B * NUM * H
        pooled_output = pooled_output.view([batch_size, NUM_DOCUMENT_PER_EXAMPLE, -1])
        pooled_output= self.inter_document_layer(pooled_output, output_attentions=output_attentions)
        pooled_output = pooled_output[0]

        # B * N * H
        doc_vectors = pooled_output.view([batch_size * NUM_DOCUMENT_PER_EXAMPLE, -1])
        logits = self.classifier(doc_vectors)
        logits = logits.view([batch_size, NUM_DOCUMENT_PER_EXAMPLE, -1])        
        loss = None
        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        raise RuntimeError('No valid return type')