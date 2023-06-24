#This is docker interpreter for run NeuralNetTrainer and examples
###Usage
```commandline
docker run -p 8888:8888 -v path/to_your/data:/usr/src/app -it --gpus all flantick/neuralnettrainer
```
###or
```commandline
docker compose up
```