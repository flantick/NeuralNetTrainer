# Segmenter
The Segmenter class is a PyTorch Lightning module for fine-tuning a neural network to segment images. To use this module, you need to initialize an instance of the Segmenter class with the required arguments. Then, you can train the model using the fit method provided by PyTorch Lightning.

## Usage
```python
from Segmenter import Segmenter

...

backbone = ...
optimizer = ...

model = Segmenter(backbone, optimizer, train_dataset, val_dataset, task_type, num_labels=num_labels,
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
