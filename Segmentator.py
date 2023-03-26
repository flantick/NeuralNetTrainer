import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split

from apps import UrDataset


class Segmentator (pl.LightningModule):

    def __init__(self, backbone, learning_rate, train_torch_dataset, val_torch_dataset, batch_size=10, pred_torch_dataset=None):
        super().__init__()

        self.model = backbone
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_torch_dataset = train_torch_dataset
        self.val_torch_dataset = val_torch_dataset
        self.pred_torch_dataset = pred_torch_dataset

        self.vl_loss = []
        self.vl_ac = []

    def forward(self, input_ids):
        x = self.model(input_ids)
        return x

    def training_step(self, batch, batch_idx):
        input_ids = batch[0]
        labels = batch[1]

        logits = self(input_ids)
        loss = F.binary_cross_entropy(logits, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]
        labels = batch[1]

        logits = self(input_ids)
        loss = F.binary_cross_entropy(logits, labels)

        acc = accuracy(logits.squeeze() ,labels.squeeze().to(torch.int), task="binary") #<--check this

        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics)
        self.vl_loss.append(loss)
        self.vl_ac.append(acc)

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch[0]

        preds = self(input_ids)

        return preds

    # def prepare_data(self):
    #     global data_t, data_v
    #     data = pd.read_csv('dataset/train/segmentation.csv')
    #     data_t, data_v = train_test_split(data, train_size=0.8)
    #
    # def prepare_for_pred(self, CSVPath):
    #     global pred_data
    #     pred_data = pd.read_csv(CSVPath)

    def train_dataloader(self):

        # train_dataset = UrDataset(data_t)  # problem mb here
        # Data loader
        train_dataloader = DataLoader(
            self.train_torch_dataset,  # The training samples.
            # sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = self.batch_size # Trains with this batch size.
        )
        return train_dataloader

    def val_dataloader(self):

        # val_dataset = UrDataset(data_v)
        # Data loader
        validation_dataloader = DataLoader(
            self.val_torch_dataset, # The validation samples.
            # sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size=self.batch_size # Evaluate with this batch size.
        )
        return validation_dataloader

    def predict_dataloader(self):
        if self.pred_torch_dataset is None:
            raise ValueError("pred_torch_dataset is None")
        # pred_dataset = UrDataset(pred_data)
        # Data loader
        pred_dataloader = DataLoader(
            self.pred_torch_dataset,  # The validation samples.
            # sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size=self.batch_size  # Evaluate with this batch size.
        )
        return pred_dataloader

    def on_validation_epoch_end(self):
        # outputs = list of dictionaries
        # avg = outputs[0]
        # for x in outputs:
        #  avg = torch.cat((avg,x), 0)

        # avg_loss = avg.mean()
        avg_loss = torch.stack([x for x in self.vl_loss]).mean()
        avg_acc = torch.stack([x for x in self.vl_ac]).mean()
        # tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        print(avg_loss)
        print(avg_acc)
        self.vl_loss = []
        self.vl_ac = []
        return {'val_loss': avg_loss, "val_acc": avg_acc  }  # , 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5 (1e-3)
                         )

        return {"optimizer": optimizer}


