# from logger import logging
import time
import torch
from src.generate.simple_model import SimpleModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Batching(SimpleModel):

    def __init__(self, model_name):
        super().__init__(model_name)
        #Addition of the padding token, trucations etc.
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"


    def generateToken(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        last_logits = logits[:, -1, :]
        next_token_ids = last_logits.argmax(dim=1)
        return next_token_ids, outputs.past_key_values
    
    def promptConfig(self, prompts):
        inputs = self.tokenizer(prompts, padding = True, return_tensors = "pt")
        return inputs

    def inference(self, inputs, max_length=32):
        generated_tokens = [
            [] for _ in range(inputs["input_ids"].shape[0])
        ]

        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        next_inputs = {
            "position_ids": position_ids,
            **inputs
        }

        for _ in range(max_length):
            next_token_ids, past_key_values = \
                self.generateToken(next_inputs)

            next_inputs = {
                "input_ids": next_token_ids.reshape((-1, 1)),
                "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
                "attention_mask": torch.cat([
                    next_inputs["attention_mask"],
                    torch.ones((next_token_ids.shape[0], 1)),  
                ], dim=1),
                "past_key_values": past_key_values,
            }

            next_tokens = self.tokenizer.batch_decode(next_token_ids)
            for i, token in enumerate(next_tokens):
                generated_tokens[i].append(token)
                
        return ["".join(tokens) for tokens in generated_tokens]

