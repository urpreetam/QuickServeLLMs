import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        t0 = time.time()
        output = self.model.generate(input_ids, max_length=100, pad_token_id=self.tokenizer.eos_token_id)
        t0 = time.time() - t0

        print(f'Generated the sequence in {t0:.3f} seconds')
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def __call__(self, prompt, max_length=50):
        return self.generate(prompt, max_length)


if __name__ == '__main__':
    model_name = 'gpt2'
    model = SimpleModel(model_name)
    prompt = 'Once upon a time'
    print(model(prompt))