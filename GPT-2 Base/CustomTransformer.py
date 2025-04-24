from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2Tokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tqdm import tqdm
import os


# === Load Tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# === Streaming Dataset Class ===
class StreamingTextDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset = load_dataset(
            "Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True
        )

    def __iter__(self):
        for sample in self.dataset:
            text = sample["text"]
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.seq_len,
                return_tensors="pt"
            )
            yield {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0)
            }


# === Custom Transformer Definition ===
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, n_heads=4, n_layers=4, max_length=128, pad_token_id=None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.size()

        # embeddings + positions
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.position_embeddings(positions)

        # causal mask (float type, as required by nn.Transformer)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
            diagonal=1
        )

        # Fix: ensure src_key_padding_mask is explicitly bool
        key_padding_mask = ~attention_mask.to(torch.bool) if attention_mask is not None else None

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(loss=loss, logits=logits)


class CustomConfig(PretrainedConfig):
    def __init__(self, vocab_size=50257, hidden_size=256, n_heads=4, n_layers=4, max_length=128,
                 pad_token_id=50256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_length = max_length
        self.pad_token_id = pad_token_id


class CustomGPTModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_length=config.max_length,
            pad_token_id=config.pad_token_id
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask, labels)


# === Main Training Function ===
def main():
    # === Set up model, tokenizer, and dataset ===
    model = CustomGPTModel(CustomConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = StreamingTextDataset(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir="./custom-transformer",
        per_device_train_batch_size=8,
        max_steps=1000,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        save_steps=250,  # Save checkpoint every 500 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        logging_steps=10  # Log progress every 50 steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # === Train Model ===
    trainer.train()

    # === Save Model & Tokenizer ===
    model.save_pretrained("./custom-gpt")
    # tokenizer.save_pretrained("./custom-gpt")


if __name__ == "__main__":
    main()
