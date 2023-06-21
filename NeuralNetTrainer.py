import torch
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from typing_extensions import Literal


class NeuralNetTrainer(pl.LightningModule):
    '''
    A class for training and evaluating a PyTorch neural network.
    You need to prepare the dataset and backbone in advance.

    :param backbone: pytorch neural net to fine-tune
    :param func_loss: loss function
    :param optimizer: optimizer
    :param train_torch_dataset: train torch iterable dataset
    :param val_torch_dataset: validation torch iterable dataset
    :param task: ['multiclass', 'multilabel', 'binary']
    :param batch_size: batch size
    :type batch_size: int
    :param num_classes: if task is 'multiclass'
    :type num_classes: int
    :param num_labels: if task is 'multilabel'
    :type num_labels: int
    :param pred_torch_dataset: prediction torch iterable dataset
    :param specify_forward_step: function that specify forward step (commonly used with pre-trained bert models,
    check the default function in the source code)
    :param specify_get_loss: function that specify get loss (commonly used with pre-trained bert models,
    check the default function in the source code)
    :param  specify_get_accuracy: function that specify get accuracy (commonly used with pre-trained bert models,
    check the default function in the source code)

    :raises ValueError:
    If you want to train model , but optimizer or val_torch_dataset or task is not defined
    if task is multiclass then num_classes must be initialized,
    if task is multilabel then num_labels must be initialized,
    if val_dataloader has been called, but val_torch_dataset is None
    if train_dataloader has been called, but train_torch_dataset is None
    if predict_dataloader has been called, but pred_torch_dataset is None
    if get_loss_and_logits has been called, but self.func_loss is None

    :return: tensor
    '''

    def __init__(self,
                 backbone,
                 func_loss=None,
                 optimizer=None,
                 train_torch_dataset=None,
                 val_torch_dataset=None,
                 task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
                 batch_size: int = 2,
                 num_classes: Optional[int] = None,
                 num_labels: Optional[int] = None,
                 pred_torch_dataset=None,
                 specify_forward_step=None,
                 specify_get_loss=None,
                 specify_get_accuracy=None
                 ):
        super().__init__()

        self.model = backbone
        self.func_loss = func_loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset
        self.pred_torch_dataset = pred_torch_dataset
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels

        if specify_forward_step is not None:
            self.forward_step = specify_forward_step
            print("if you specify forward step then you should generally specify 'specify_get_loss', and 'specify_get_accuracy'")
        else:
            self.forward_step = NeuralNetTrainer.default_forward_step

        if specify_get_loss is not None:
            self.get_loss = specify_get_loss
        else:
            self.get_loss = NeuralNetTrainer.default_get_loss

        if specify_get_accuracy is not None:
            self.get_accuracy = specify_get_accuracy
        else:
            self.get_accuracy = NeuralNetTrainer.default_get_acc


        if train_torch_dataset is not None and (self.optimizer is None or
                                                self.val_torch_dataset is None or self.task is None):
            raise ValueError("If you want to train model then func_loss, optimizer, train_torch_dataset, "
                             "val_torch_dataset and task must be defined")

        if self.task == 'multiclass' and self.num_classes is None:
            raise ValueError("if task is multiclass then num_classes must be initialized")
        elif self.task == 'multilabel' and self.num_labels is None:
            raise ValueError("if task is multilabel then num_labels must be initialized")


        self.vl_loss = torch.Tensor([])
        self.vl_acc = torch.Tensor([])

    def forward(self, *args, **kwargs):
        x = self.forward_step(self.model, *args, **kwargs)
        return x

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.get_loss(self, preds, batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.get_loss(self, preds, batch)
        acc = self.get_accuracy(self, preds, batch)
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        self.vl_loss = torch.cat([self.vl_loss, loss.unsqueeze(0).to('cpu')])
        self.vl_acc = torch.cat([self.vl_acc, acc.unsqueeze(0).to('cpu')])

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self(batch)
        return preds

    def train_dataloader(self):
        if self.train_torch_dataset is None:
            raise ValueError("train_dataloader has been called, but train_torch_dataset is None")
        train_dataloader = DataLoader(
            self.train_torch_dataset,
            batch_size=self.batch_size
        )
        return train_dataloader

    def val_dataloader(self):
        if self.val_torch_dataset is None:
            raise ValueError("val_dataloader has been called, but val_torch_dataset is None")
        validation_dataloader = DataLoader(
            self.val_torch_dataset,
            batch_size=self.batch_size
        )
        return validation_dataloader

    def predict_dataloader(self):
        if self.pred_torch_dataset is None:
            raise ValueError("predict_dataloader has been called, but pred_torch_dataset is None")
        pred_dataloader = DataLoader(
            self.pred_torch_dataset,
            batch_size=1
        )
        return pred_dataloader

    def on_validation_epoch_end(self):
        val_loss_mean = torch.mean(self.vl_loss)
        val_acc_mean = torch.mean(self.vl_acc)

        print(f"\n=======\nMean validation loss: {round(val_loss_mean.item(), 6)}")
        print(f"Mean validation accuracy: {round(val_acc_mean.item(), 6)}\n=======\n")

        self.vl_loss = torch.Tensor([])
        self.vl_acc = torch.Tensor([])

        return {'val_loss': val_loss_mean, "val_acc": val_acc_mean}

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    @staticmethod
    def default_forward_step(model, batch):
        input_ids = batch[0]
        x = model(input_ids)
        return x

    @staticmethod
    def default_get_loss(segmenter_object, preds, batch):
        target = batch[1]
        loss = segmenter_object.func_loss(preds, target)
        return loss

    @staticmethod
    def default_get_acc(segmenter_object, preds, batch):
        target = batch[1]
        acc = accuracy(preds.squeeze(),
                       target.squeeze().to(torch.int),
                       task=segmenter_object.task,
                       num_labels=segmenter_object.num_labels,
                       num_classes=segmenter_object.num_classes
                       )
        return acc
