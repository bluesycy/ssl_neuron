import json
from ssl_neuron.datasets import build_dataloader
from ssl_neuron.graphdino import create_model
from ssl_neuron.train import Trainer


config = json.load(open('../configs/config.json'))


model = create_model(config)
model.train()
model.cuda();


dataloaders = build_dataloader(config)

trainer = Trainer(config, model, dataloaders)

trainer.train()