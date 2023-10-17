#!/usr/bin/env python
# coding: utf-8

# ## Load Stuff

# In[2]:


import torch
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt.utils import view_level, convert_level_to_png, join_list_of_list, characterize


# In[3]:


print(torch.backends.mps.is_available())


# In[4]:


BASE = "distilgpt2"


# In[5]:


from transformers import AutoConfig, AutoModelWithLMHead


# In[6]:


mario_lm = MarioLM(lm_path=BASE, tokenizer_path=BASE)


# In[9]:


#mps_device = torch.device("mps")
#mario_lm = mario_lm.to(mps_device)


# ### Load Dataset (Optional)

# In[7]:


dataset = MarioDataset(mario_lm.tokenizer)


# In[8]:


view_level(dataset.input_ids[:700], dataset.tokenizer)


# In[9]:


img = convert_level_to_png(dataset.input_ids[:700],  dataset.tokenizer)[0]
img


# ### Setup training

# In[10]:


config = TrainingConfig(save_iteration=10)


# In[11]:


trainer = MarioGPTTrainer(mario_lm, dataset, config=config)


# In[ ]:


trainer.train(100, batch_size=1)


# In[ ]:




