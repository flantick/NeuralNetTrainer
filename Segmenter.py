import torch
from torch.optim import Adam
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Segmenter (pl.LightningModule):
    '''
    Segmenter fine-tune neural net to segments images.
    You need to prepare the dataset and backbone in advance.

    :param backbone: neural net to fine-tune
    :param func_loss: loss function
    :param learning_rate: learning rate
    :type learning_rate: float
    :param train_torch_dataset: train torch iterable dataset
    :param val_torch_dataset: validation torch iterable dataset
    :param task: ['multiclass', 'multilabel', 'binary']
    :param batch_size: batch size
    :type batch_size: int
    :param num_classes: if task is 'multiclass'
    :type num_classes: int
    :param num_labels: if task is 'multilabel'
    :type num_labels: int
    :param train_torch_dataset: prediction torch iterable dataset

    :raises ValueError: if task is multiclass then num_classes must be initialized,
    if task is multilabel then num_labels must be initialized,
    if you use predcit step when  pred_torch_dataset is None.

    :return: tensor
    '''

    def __init__(self, backbone, func_loss, learning_rate, train_torch_dataset, val_torch_dataset, task, batch_size,
                 num_classes=None, num_labels=None, pred_torch_dataset=None):
        super().__init__()

        self.model = backbone
        self.func_loss = func_loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset
        self.pred_torch_dataset = pred_torch_dataset
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels

        if self.task == 'multiclass' and self.num_classes is None:
            raise ValueError("if task is multiclass then num_classes must be initialized")
        elif self.task == 'multilabel' and self.num_labels is None:
            raise ValueError("if task is multilabel then num_labels must be initialized")



        self.vl_loss = torch.Tensor([])
        self.vl_acc = torch.Tensor([])

    def forward(self, input_ids):
        x = self.model(input_ids)
        return x

    def training_step(self, batch, batch_idx):
        input_ids = batch[0]
        labels = batch[1]

        logits = self(input_ids)
        loss = self.func_loss(logits, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        with torch.no_grad():
            logits = self(input_ids)
            loss = self.func_loss(logits, labels)

        acc = accuracy(logits.squeeze(), labels.squeeze().to(torch.int), task=self.task, num_labels=self.num_labels,
                       num_classes=self.num_classes)

        metrics = {"val_loss": loss, "val_acc": acc}
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        self.vl_loss = torch.cat([self.vl_loss, loss.unsqueeze(0).to('cpu')])
        self.vl_acc = torch.cat([self.vl_acc, acc.unsqueeze(0).to('cpu')])

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch[0]

        preds = self(input_ids)

        return preds

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_torch_dataset,
            batch_size=self.batch_size
        )
        return train_dataloader

    def val_dataloader(self):
        validation_dataloader = DataLoader(
            self.val_torch_dataset,
            batch_size=self.batch_size
        )
        return validation_dataloader

    def predict_dataloader(self):
        if self.pred_torch_dataset is None:
            raise ValueError("pred_torch_dataset is None")
        pred_dataloader = DataLoader(
            self.pred_torch_dataset,
            batch_size=self.batch_size
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
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate,
                         )

        return {"optimizer": optimizer}
