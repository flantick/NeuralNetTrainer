# NeuralNetTrainer
 A class for training and evaluating PyTorch neural networks. It provides a convenient interface for training and 
 fine-tuning neural networks using custom loss functions, optimizers, and 
 datasets. This class is suitable for various tasks such as classification. 
 It handles key functionalities like training, validation, and prediction. 
 It simplifies the process of training neural networks and allows for easy 
 customization.

## Usage

```python
from NeuralNetTrainer import NeuralNetTrainer

...

backbone = ...
optimizer = ...
loss_func = ...

model = NeuralNetTrainer(backbone, loss_func, optimizer, train_dataset, val_dataset, task_type, num_labels=num_labels,
                         batch_size=batch_size, pred_torch_dataset=pred_dataset)

trainer = pl.Trainer(
    accelerator='gpu',
    min_epochs=min_epochs,
    max_epochs=max_epochs
)

trainer.fit(model)

```
## Examples
check [examples_notebooks](examples_notebooks/)
