from collections import namedtuple
from transformers import Trainer, PreTrainedModel, TrainingArguments

# class HotpotRankerPredictionOutput(NamedTuple):
#     predictions
#     label_ids: Optional[np.ndarray]
#     metrics: Optional[Dict[str, float]]
from transformers.utils import logging
from tqdm.auto import tqdm, trange
from transformers.trainer import *

from common.utils import *

logger = logging.get_logger(__name__)

HotpotRankerPredictionOutput = namedtuple('HotpotRankerPredictionOutput', ['predictions', 'label_ids', 'metrics'])
HotpotRankerEvalPrediction = namedtuple('HotpotRankerEvalPrediction', ['predictions', 'label_ids'])

# evaluation using sklearn
def doc_level_acc(eva_preds):
    preds, labels = eva_preds
    
    results = []
    for pred, label in zip(preds, labels):
        # select top 2
        pred = pred.squeeze(0)
        label = label.squeeze(0)
        # label = label[:pred.shape[0]]

        pred = pred[:,1] > pred[:,0]
        label = label > 0
        
        results.append(all(pred == label))
    acc = sum(results) * 1.0 / len(results)
    return {
        "acc": acc,
    }


class HotpotRankerTrainer(Trainer):

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    
    def __init__(
        self,
        model = None,
        args = None,
        data_collator= None,
        train_dataset= None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        tb_writer= None,
        optimizers = (None, None),
        **kwargs,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, None, tb_writer, optimizers, **kwargs)
        self.compute_metrics = compute_metrics

    def predict(self, test_dataset):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`nlp.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed.

        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self.prediction_loop(test_dataloader, description="Prediction")

    def prediction_loop(self, dataloader, description, prediction_loss_only = None):
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        eval_losses: List[float] = []
        preds = []
        label_ids = []
        model.eval()


        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        samples_count = 0
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            samples_count += batch_size
            if loss is not None:
                eval_losses.append(loss * batch_size)
            if logits is not None:
                # preds = logits if preds is None else torch.cat((preds, logits), dim=0)
                preds.append(logits)
            if labels is not None:
                # label_ids = labels if label_ids is None else torch.cat((label_ids, labels), dim=0)
                label_ids.append(labels)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            pass
            # Do nothing

            # raise RuntimeError('Distributed evaluating not supported')
            # In distributed mode, concatenate all results from all nodes:
            # if preds is not None:
            #     preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            # if label_ids is not None:
            #     label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            # preds = preds.cpu().numpy()
            preds = [p.cpu().numpy() for p in preds]
        if label_ids is not None:
            # label_ids = label_ids.cpu().numpy()
            label_ids = [p.cpu().numpy() for p in label_ids]

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(HotpotRankerEvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.sum(eval_losses) / samples_count

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        # logger.info(" [epoch %d step %d] metrics: %s", self.epoch, self.global_step, metrics)
        return HotpotRankerPredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(self, model, inputs, prediction_loss_only):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().item()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels)
