import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class KVCACHE:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model.to(self.device)
        self.model.eval()


    def generate_token_cache(self, input_ids):
        with torch.no_grad():
            output = self.model(**input_ids)
            last_logits = output.logits[0, -1, :]
            next_token = torch.argmax(last_logits).unsqueeze(0)
            return next_token, output.past_key_values

    def generate(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output_list = []
        t0 = time.time()
        for _ in range(max_length):
            next_token, past_key_values = self.generate_token_cache(input_ids)
            input_ids = {
                'input_ids': next_token.rehsape(1, 1),
                'attention_mask': torch.cat([input_ids['attention_mask'], torch.ones(1, 1).to(self.device)], dim=1),
                'past_key_values': past_key_values
            }
            output_list.append(self.tokenizer.decode(next_token))

        t0 = time.time() - t0
        print(f'Generated the sequence in {t0:.3f} seconds')
        return "".join(output_list)

    def __call__(self, prompt, max_length=50):
        return self.generate(prompt, max_length)


if __name__ == '__main__': 
    model_name = 'gpt2'
    model = KVCACHE(model_name)
    prompt = 'Once upon a time'
    print(model(prompt))