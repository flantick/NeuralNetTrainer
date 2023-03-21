import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split

from cfg import backbone
from apps import pleaseDataset


class Segmentator (pl.LightningModule):

    def __init__(self, batch_size=10):
        super().__init__()

        self.model = backbone
        self.batch_size = batch_size

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

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch[0]

        preds = self(input_ids)

        return preds

    def prepare_data(self):
        global data_t, data_v
        data = pd.read_csv('dataset/train/segmentation.csv')
        data_t, data_v = train_test_split(data, train_size=0.8)

    def prepare_for_pred(self, CSVPath):
        global pred_data
        pred_data = pd.read_csv(CSVPath)

    def train_dataloader(self):

        train_dataset = pleaseDataset(data_t)  # problem mb here
        # Data loader
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            # sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = self.batch_size # Trains with this batch size.
        )
        return train_dataloader

    def val_dataloader(self):

        val_dataset = pleaseDataset(data_v)
        # Data loader
        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            # sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = self.batch_size # Evaluate with this batch size.
        )
        return validation_dataloader

    def predict_dataloader(self):

        pred_dataset = pleaseDataset(pred_data)
        # Data loader
        pred_dataloader = DataLoader(
            pred_dataset, # The validation samples.
            # sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = self.batch_size # Evaluate with this batch size.
        )
        return pred_dataloader

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        # avg = outputs[0]
        # for x in outputs:
        #  avg = torch.cat((avg,x), 0)

        # avg_loss = avg.mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        # tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        print(avg_loss)
        print(avg_acc)
        return {'val_loss': avg_loss, "val_acc": avg_acc  }  # , 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=1e-3,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                         )

        return {"optimizer": optimizer}


