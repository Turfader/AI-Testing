import copy
import random
import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import logging
from CustomTransformer import StreamingTextDataset, CustomConfig, CustomGPTModel

logging.basicConfig(
    filename='training_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class PopulationManager:
    def __init__(
        self,
        pop_size: int = 100,
        mutation_rate: float = 0.02,
        eval_steps: int = 250,
        device: torch.device = None,
        seq_len: int = 128,
        batch_size: int = 8,
        lr: float = 5e-4,
        output_dir: str = "./evolved_models"
    ):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.eval_steps = eval_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Tokenizer & data collator
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Dataset and dataloader
        self.dataset = StreamingTextDataset(tokenizer=self.tokenizer, seq_len=self.seq_len)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)

        # Initialize population of models and optimizers
        self.population = []
        self.optimizers = []
        self._initialize_population()

    def _initialize_population(self):
        for idx in range(self.pop_size):
            cfg = CustomConfig()
            model = CustomGPTModel(cfg)
            model.to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
            self.population.append(model)
            self.optimizers.append(opt)

    def _mutate(self, config: CustomConfig) -> CustomConfig:
        new_cfg = copy.deepcopy(config)
        if random.random() < self.mutation_rate:
            delta = random.choice([-1, 1])
            new_cfg.n_heads = max(1, min(8, new_cfg.n_heads + delta))
        if random.random() < self.mutation_rate:
            delta = random.choice([-1, 1])
            new_cfg.n_layers = max(1, min(8, new_cfg.n_layers + delta))
        if random.random() < self.mutation_rate:
            delta = random.choice([-1, 1]) * 32
            new_cfg.hidden_size = max(128, min(512, new_cfg.hidden_size + delta))
        return new_cfg

    def _reproduce(self, parents: list) -> list:
        children = []
        while len(children) + len(parents) < self.pop_size:
            parent = parents[len(children) % len(parents)]
            parent_cfg = copy.deepcopy(parent.config)
            child_cfg = self._mutate(parent_cfg)
            child = CustomGPTModel(child_cfg)
            child.to(self.device)
            children.append(child)
        return children

    def _evaluate(self, model, optimizer) -> float:
        model.train()
        total_loss = 0.0
        it = iter(self.loader)
        for step in range(self.eval_steps):
            batch = next(it)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step + 1}/{self.eval_steps} - Loss: {loss.item():.4f}")
                logging.info(f"Step {step + 1}/{self.eval_steps} - Loss: {loss.item():.4f}")

        return total_loss / self.eval_steps

    def _save_best(self, parents: list, generation: int):
        gen_dir = os.path.join(self.output_dir, f"generation_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        for idx, model in enumerate(parents):
            save_path = os.path.join(gen_dir, f"model_{idx}")
            model.save_pretrained(save_path)
        # Save tokenizer once per generation
        # self.tokenizer.save_pretrained(gen_dir)
        print(f"Saved top {len(parents)} models in {gen_dir}")
        logging.info(f"Saved top {len(parents)} models in {gen_dir}")

    def evolve(self, generations: int = 4, top_k: int = 10):
        for gen in range(1, generations + 1):
            print(f"\n=== Generation {gen} ===")
            logging.info(f"\n=== Generation {gen} ===")
            # Evaluate
            losses = []
            for idx, (m, opt) in enumerate(zip(self.population, self.optimizers)):
                print(f"\nEvaluating model {idx}...")
                logging.info(f"\nEvaluating model {idx}...")
                avg_loss = self._evaluate(m, opt)
                losses.append((avg_loss, m))
                print(f"Model {idx}: Average Loss = {avg_loss:.4f}")
                logging.info(f"Model {idx}: Average Loss = {avg_loss:.4f}")

            # Select
            losses.sort(key=lambda x: x[0])
            parents = [m for _, m in losses[:top_k]]

            # Save best
            self._save_best(parents, gen)

            # Reproduce + mutate + reset optimizers
            children = self._reproduce(parents)
            self.population = parents + children
            self.optimizers = [torch.optim.AdamW(m.parameters(), lr=self.lr) for m in self.population]

    def run(self, generations: int = 4, top_k: int = 10):
        self.evolve(generations=generations, top_k=top_k)


if __name__ == "__main__":
    pm = PopulationManager(pop_size=100, mutation_rate=0.02, eval_steps=250)
    pm.run(generations=4, top_k=10)
