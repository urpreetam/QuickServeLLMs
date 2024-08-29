import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.generate.simple_model import SimpleModel

class ModelWithKVCache(SimpleModel):
    def __init__(self, model_name):
        super().__init__(model_name)


    def generateToken(self, inputs):
        #Caching technique is used to store the past key values of the model
        with torch.no_grad():
            output = self.model(**inputs)
            last_logits = output.logits[0, -1, :]
            next_token = torch.argmax(last_logits).unsqueeze(0)
            return next_token, output.past_key_values

    def inference(self, inputs, max_length=32):
        generated_tokens = []
        next_inputs = inputs
        durations_s = []
        for _ in range(10):
            t0 = time.time()
            next_token_id, past_key_values = self.generateToken(next_inputs)
            durations_s += [time.time() - t0]
            
            next_inputs = {
                "input_ids": next_token_id.reshape((1, 1)),
                "attention_mask": torch.cat(
                    [next_inputs["attention_mask"], torch.tensor([[1]])],
                    dim=1),
                "past_key_values": past_key_values,
            }
            
            next_token = self.tokenizer.decode(next_token_id)
            generated_tokens.append(next_token)

        plot_data = self.plotData(durations_s)
        return "".join(generated_tokens), plot_data