#!/usr/bin/env python
# coding: utf-8

# ## Load Stuff

import torch
import sys
mypath = "/Users/james/playground/mario-gpt/"
sys.path.insert(0, mypath)

from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize
from mario_gpt.lm.llama import MarioLlama

mario_lm = MarioLM()


mps_device = torch.device("mps")
mario_lm = mario_lm.to(mps_device)


# ### Load Dataset (Optional)
dataset = MarioDataset(mario_lm.tokenizer)


# ### Setup training

config = TrainingConfig(save_iteration=100)
trainer = MarioGPTTrainer(mario_lm, dataset, config=config)
trainer.train(1000, batch_size=1)
