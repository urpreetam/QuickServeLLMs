from logger import logging
import time
import torch
from simple_model import SimpleModel
from transformers import AutoModelForCausalLM, AutoTokenizer


#TODO: Implement the superclass simple model here.
class Batching(SimpleModel):

    def __init__(self, model_name):
        super().__init__(model_name)
        #Addition of the padding token, trucations etc.
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model.config.pad_token_id = model.config.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    def promptConfig(self, prompts):
        inputs = self.tokenizer(prompts, padding = True, return_tensors = "pt")
        #TODO: Complete this function
