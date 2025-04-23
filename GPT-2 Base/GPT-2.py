import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Responder(torch.nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set pad token to eos token if not set (gpt2 has no pad token by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, input_text, max_new_tokens=100):
        # If you don't format it like this, it goes even farther off the rails
        prompt = f"User: {input_text}\nAI:"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part (after 'AI:')
        return response.split("AI:")[-1].strip()


if __name__ == "__main__":
    bot = GPT2Responder('gpt2')
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            break
        response = bot(user_input)
        print(f"AI: {response}\n")
