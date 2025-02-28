import torch
from datasets import load_dataset
from transformer_lens.hook_points import HookedRootModule


class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.dataset = load_dataset(cfg["dataset_path"], split="train", 
                                   streaming=True, trust_remote_code=True)
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        # Match the full implementation parameter names
        self.store_batch_size_prompts = cfg.get("store_batch_size_prompts", 8)
        self.train_batch_size_tokens = cfg.get("train_batch_size_tokens", 4096)
        self.device = cfg["device"]
        self.n_batches_in_buffer = cfg.get("n_batches_in_buffer", 8)
        self.half_buffer_size = self.n_batches_in_buffer // 2
        self.tokens_column = self._get_tokens_column()
        self.config = cfg
        self.tokenizer = model.tokenizer
        self.iterable_sequences = self._iterate_tokenized_sequences()
        self._storage_buffer = None
        
    def _get_tokens_column(self):
        sample = next(iter(self.dataset))
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")
            
    def _iterate_raw_dataset(self):
        for row in self.dataset:
            yield row[self.tokens_column]
            
    def _iterate_raw_dataset_tokens(self):
        for row in self._iterate_raw_dataset():
            if self.tokens_column == "text":
                tokens = (
                    self.model.to_tokens(
                        row,
                        truncate=False,
                        move_to_device=False,
                        prepend_bos=self.config.get("prepend_bos", True),
                    )
                    .squeeze(0)
                    .to(self.device)
                )
                if len(tokens.shape) != 1:
                    raise ValueError(f"tokens.shape should be 1D but was {tokens.shape}")
                yield tokens
            else:
                yield torch.tensor(row, dtype=torch.long, device=self.device)
                
    def _iterate_tokenized_sequences(self):
        if self.tokens_column != "text":
            # If pretokenized
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row[:self.context_size],
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            # For untokenized data, implement a proper concat_and_batch
            from itertools import islice

            from tqdm.autonotebook import tqdm
            
            buffer = []
            tokens_iter = self._iterate_raw_dataset_tokens()
            
            while True:
                try:
                    # Get next sequence
                    tokens = next(tokens_iter)
                    buffer.extend(tokens.tolist())
                    
                    # When buffer has enough tokens, yield context-sized chunks
                    while len(buffer) >= self.context_size:
                        yield torch.tensor(
                            buffer[:self.context_size], 
                            dtype=torch.long, 
                            device=self.device
                        )
                        buffer = buffer[self.context_size:]
                except StopIteration:
                    # Reset iterator
                    tokens_iter = self._iterate_raw_dataset_tokens()
                    if len(buffer) >= self.context_size:
                        yield torch.tensor(
                            buffer[:self.context_size], 
                            dtype=torch.long, 
                            device=self.device
                        )
                        buffer = buffer[self.context_size:]
    
    def get_batch_tokens(self, batch_size=None):
        if not batch_size:
            batch_size = self.store_batch_size_prompts
            
        sequences = []
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                sequences.append(next(self.iterable_sequences))
                
        return torch.stack(sequences, dim=0)
    
    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.config["layer"] + 1,
                prepend_bos=False,  # We already handle BOS in tokenization
            )
        activations = cache[self.hook_point]
        
        # Handle reshape similar to full implementation
        n_batches, n_context = activations.shape[:2]
        d_in = self.config["act_size"]
        
        if activations.ndim > 3:  # if we have a head dimension
            return activations.reshape(n_batches, n_context, -1)
        else:
            return activations
            
    @property
    def storage_buffer(self):
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.half_buffer_size)
        return self._storage_buffer
    
    def get_buffer(self, n_batches_in_buffer, shuffle=True):
        batch_size = self.store_batch_size_prompts
        total_size = batch_size * n_batches_in_buffer
        d_in = self.config["act_size"]
        
        # Initialize buffer
        from tqdm.autonotebook import tqdm
        buffer = torch.zeros(
            (total_size, self.context_size, d_in),
            dtype=self.config.get("dtype", torch.float32),
            device=self.device,
        )
        
        for i in tqdm(range(0, total_size, batch_size), desc="Filling buffer", leave=False):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens)
            buffer[i:i+batch_size] = activations
            
        buffer = buffer.reshape(-1, d_in)
        
        if shuffle:
            idx = torch.randperm(buffer.shape[0])
            buffer = buffer[idx]
            
        return buffer
        
    def get_data_loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        
        # Get new samples for half the buffer
        new_samples = self.get_buffer(self.half_buffer_size)
        
        # Mix with storage buffer (similar to full implementation)
        mixing_buffer = torch.cat([new_samples, self.storage_buffer], dim=0)
        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
        
        # Update storage buffer with half
        self._storage_buffer = mixing_buffer[:mixing_buffer.shape[0] // 2]
        
        # Return dataloader with other half
        return iter(DataLoader(
            TensorDataset(mixing_buffer[mixing_buffer.shape[0] // 2:]),
            batch_size=self.train_batch_size_tokens,
            shuffle=True
        ))
        
    def next_batch(self):
        try:
            if not hasattr(self, 'dataloader'):
                self.dataloader = self.get_data_loader()
            return next(self.dataloader)[0]
        except StopIteration:
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)[0]