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

    def generateToken(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        last_logits = logits[0, -1, :]
        next_token_id = last_logits.argmax()
        return next_token_id
    
    def plotData(self, durations_s):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64

        # Generate the plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(durations_s) + 1), durations_s, marker='o')
        plt.title(f'Total Duration: {sum(durations_s):.4f} seconds')
        plt.xlabel('Token Generation Step')
        plt.ylabel('Time (seconds)')
        plt.grid(True)

        # Save the plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()

        # Close the plot to free up memory
        plt.close()
        return plot_data


    def inference(self, prompt, max_length=8):
        generated_tokens = []
        next_inputs = prompt
        durations_s = []
        for _ in range(max_length):
            t0 = time.time()
            next_token_id = self.generateToken(next_inputs)
            durations_s.append(time.time() - t0)
            
            next_inputs = {
                "input_ids": torch.cat(
                    [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
                    dim=1),
                "attention_mask": torch.cat(
                    [next_inputs["attention_mask"], torch.tensor([[1]])],
                    dim=1),
            }
    
            next_token = self.tokenizer.decode(next_token_id)
            generated_tokens.append(next_token)
            plot_data = self.plotData(durations_s)


        return ''.join(generated_tokens), plot_data
    

    def __call__(self, prompt, max_length=8):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return self.inference(inputs, max_length)
