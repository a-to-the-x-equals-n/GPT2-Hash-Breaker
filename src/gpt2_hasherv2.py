import yaml
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils.similarity import char_similarity, levenshtein, jaccard
# --- personal debug lib ---
from debuggernaut import heimdahl, laufeyspawn
torch.cuda.empty_cache()

class GPT2HashTuner:
    _ROOT = Path.cwd().parent
    # --- train ---
    _DEV_SHARD = _ROOT / 'data/shards/dev_shards.jsonl'
    _GPT2_MODEL = _ROOT / 'gpt2-hashmodel'
    # --- test ---
    _EVAL_SRC = _ROOT / 'data/eval/dev_eval.yaml'
    _EVAL_OUT = _ROOT / 'results/gpt/inferences.yaml'

    _EVAL_STRATEGY = 'no'

    @laufeyspawn(summoned = True)
    def __init__(
        self,
        data: str | Path = None,
        /,
        *,
        limit: int = None,
        batch: int = 8,
        lr: float = 5e-5,
        save_steps: int = 10_000,
        checkpoints: int = 2,
        weight_decay: float = 0.01,
        fp16: bool = True,
        load_best: bool = False,
        save_strategy: str = 'steps',
        gradient_checkpointing: bool = False,
        vb: bool = False,
        iters: int = None,
        **kwargs
    ):
        '''
        Initializes model components and training configuration (streaming .jsonl dataset).

        Parameters:
        -----------
        limit : int
            maximum number of dataset entries to load (default: None, no cap).
        batch : int
            batch size per device (default: 8).
        lr : float
            initial learning rate (default: 5e-5).
        save_steps : int
            steps between checkpoint saves (default: 1000).
        checkpoints : int
            maximum checkpoints to retain (default: 2).
        weight_decay : float
            L2 regularization strength (default: 0.01).
        fp16 : bool
            enable mixed precision training (default: True).
        load_best : bool
            load best model after training (default: False).
        save_strategy : str
            checkpoint saving strategy (default: 'steps').
        gradient_checkpointing : bool
            enable memory-efficient gradients (default: True).
        vb : bool
            verbose debug output (default: False).
        kwargs : dict
            extra keyword arguments passed to Trainer/TrainingArguments.
        '''
        self.vb = vb
        self.iters = iters
        self.limit = limit
        self.data = self.__class__._ROOT / data or self.__class__._DEV_SHARD

        # --- gpt-2 components ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.__class__._GPT2_MODEL)      # load pretrained tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token                             # set pad token for gpt-2 compatibility
        self.model = AutoModelForCausalLM.from_pretrained(self.__class__._GPT2_MODEL)   # load pretrained model

        # --- dataset ---
        self.dataset = self._load_dataset(self.data)                        # load streaming dataset
        self.tokenized = self.dataset.map(self._tokenize, batched = True)   # tokenize all text prompts

        # --- training configuration ---
        self.batch = batch                                      # samples per device
        self.lr = lr                                            # optimizer learning rate
        self.save_steps = save_steps                            # checkpoint frequency
        self.checkpoints = checkpoints                          # max saved checkpoints
        self.weight_decay = weight_decay                        # L2 regularization
        self.fp16 = fp16                                        # mixed precision flag
        self.load_best = load_best                              # best model retention
        self.save_strategy = save_strategy                      # checkpoint saving strategy
        self.gradient_checkpointing = gradient_checkpointing    # memory optimization flag
        self.args_extra = kwargs                                # catch anything extra safely

        heimdahl('[INITIALIZED]', unveil = self.vb)

    # --- streaming jsonl loader ---
    @laufeyspawn(summoned = False)
    def _load_dataset(self, source: str | Path = None):
        '''
        Loads a dataset from pre-sharded .jsonl files using streaming.

        Returns:
        --------
        Dataset
            a streaming HuggingFace dataset with prompts formatted as text fields
        '''
        pattern = str(source)
        ds = load_dataset('json', data_files = pattern, streaming = True)['train']
        ds = ds.shuffle(buffer_size = 100_000, seed = 42)

        if self.limit is not None:
            ds = ds.take(self.limit)

        def to_prompt(ex):
            return {'text': f'Hash: {ex["input"]}\nPassword: {ex["output"]}'}

        return ds.map(to_prompt)

    # --- tokeniser with masking ---
    @laufeyspawn(summoned = False)
    def _tokenize(self, batch):
        '''
        Tokenizes a batch of text prompts for model input,
        and masks the labels so that loss is only calculated on the password portion.

        Parameters:
        -----------
        batch : dict[str, list[str]]
            a dictionary containing 'text' prompts

        Returns:
        --------
        dict[str, list[list[int]]]
            tokenized inputs with masked loss labels
        '''
        tokenized = self.tokenizer(batch['text'], padding = 'max_length', truncation = True, max_length = 128)
        pw_tokens = self.tokenizer('Password:', add_special_tokens = False)['input_ids']

        labels = []
        for ids in tokenized['input_ids']:
            try:
                idx = next(
                    i
                    for i in range(len(ids) - len(pw_tokens))
                    if ids[i : i + len(pw_tokens)] == pw_tokens
                )
                idx += len(pw_tokens)
            except StopIteration:
                idx = len(ids)

            labels.append([-100] * idx + ids[idx:])

        tokenized['labels'] = labels
        return tokenized

    # --- train on streaming jsonl shards ---
    @laufeyspawn(summoned = True)
    def train(
        self,
        /,
        *,
        batch: int = 0,
        lr: float = 0.0,
        save_steps: int = 0,
        checkpoints: int = 0,
        weight_decay: float = 0.0,
        fp16: bool = None,
        load_best: bool = None,
        save_strategy: str = '',
        resume: bool = False,
        gradient_checkpointing = None
    ):
        '''
        Fine-tunes the language model using HuggingFace's Trainer.

        Parameters:
        -----------
        batch : int
            number of samples per device per step
        lr : float
            initial learning rate for the AdamW optimizer
        save_steps : int
            number of steps between model checkpoint saves
        checkpoints : int
            maximum number of checkpoints to keep
        weight_decay : float
            L2 regularization strength
        fp16 : bool
            whether to use mixed precision training
        load_best : bool
            whether to load the best checkpoint at the end
        save_strategy : str
            checkpoint saving strategy ('steps', 'epoch')
        resume : bool
            if True, resumes training from latest checkpoint
        gradient_checkpointing : bool
            enable memory-efficient gradients
        '''
        fp16 = self.fp16 if fp16 is None else fp16
        load_best = self.load_best if load_best is None else load_best
        gradient_checkpointing = self.gradient_checkpointing if gradient_checkpointing is None else gradient_checkpointing
        batch = batch or self.batch

        # --- count lines if no explicit limit was given ---
        if self.limit is not None:
            r = self.limit
        else:
            with open(self.data, 'r', encoding = 'utf-8') as f:
                r = sum(1 for _ in f)
        steps = r // batch

        # --- pre-training info ---
        model_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_percent = (trainable_parameters / model_parameters) * 100
        heimdahl(f'[TRAINABLE PARAMETERS] {trainable_parameters:,} / {model_parameters:,} ({trainable_percent:.2f}% of total)', unveil = self.vb, threat = 2)

        # --- define training arguments ---
        args = TrainingArguments(
            output_dir = str(self.__class__._GPT2_MODEL),           # where to save checkpoints and model
            max_steps = steps,
            per_device_train_batch_size = batch or self.batch,      # samples per device
            save_steps = save_steps or self.save_steps,             # checkpoint every N steps
            save_total_limit = checkpoints or self.checkpoints,     # keep only 2 checkpoints
            eval_strategy = self.__class__._EVAL_STRATEGY,          # NOTE: no eval strat when streaming
            learning_rate = lr or self.lr,                          # optimizer learning rate
            weight_decay = weight_decay or self.weight_decay,       # l2 regularization
            fp16 = fp16,                                            # enable mixed precision
            load_best_model_at_end = load_best,                     # best model retention
            save_strategy = save_strategy or self.save_strategy,    # checkpoint save timing
            gradient_checkpointing = gradient_checkpointing,        # memory-efficient training
            # --- log stuff ---
            logging_steps = 500,
            logging_strategy = 'steps',
            report_to = 'tensorboard',
        )

        # --- trainer orchestration ---
        heimdahl('[BUILDING TRAINER]', unveil = self.vb)

        self.model.train()

        trainer = Trainer(
            model = self.model,
            args = args,
            train_dataset = self.tokenized,                       # streaming dataset
            processing_class = self.tokenizer,                    # tokenizer
            data_collator = DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm = False
            )
        )

        # --- start training ---
        heimdahl('[STARTING TRAINING]', unveil = self.vb, threat = 1)
        trainer.train(resume_from_checkpoint = resume)

        # --- save after training ---
        heimdahl('[ENDING TRAINING]', unveil = self.vb, threat = 3)
        heimdahl('[SAVING MODEL]', unveil = self.vb)
        self.model.save_pretrained(self.__class__._GPT2_MODEL)
        self.tokenizer.save_pretrained(self.__class__._GPT2_MODEL)

    # --- evaluation (writes yaml results) ---
    @laufeyspawn(summoned = True)
    def evaluate(self, *, test_path = '', save_file = '', max_new_tokens = 24):
        '''
        Runs inference over hash inputs and scores predictions against ground truths.
        Appends results to a YAML file with similarity metrics.

        Parameters:
        -----------
        test_path : str
            path to a yaml file containing test entries (default: class _EVAL_SRC)
        save_file : str
            output file path for saving evaluation results (default: class _EVAL_OUT)
        max_new_tokens : int
            maximum new tokens to generate per prediction (default: 24)
        '''
        heimdahl('[EVALUATING MODEL]', unveil = self.vb, threat = 1)

        # --- switch model to eval mode ---
        self.model.eval()

        # --- disable gradient checkpointing if available ---
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        evals = Path(test_path or self.__class__._EVAL_SRC)
        outpath = Path(save_file or self.__class__._EVAL_OUT)
        outpath.parent.mkdir(parents = True, exist_ok = True)

        # --- dynamically adjust filename based on eval source ---
        stem = self.data.stem
        outpath = outpath.parent / f'inferences_{stem.upper()}_{self.iters + 1}.yaml'

        entries = yaml.safe_load(evals.open())
        results = []

        for entry in entries:
            prompt = f'Hash: {entry["input"]}\nPassword:'
            inputs = self.tokenizer(prompt, return_tensors = 'pt').to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    temperature = 0.01,
                    top_k = 1,
                    pad_token_id = self.tokenizer.eos_token_id
                )

            prediction = self.tokenizer.decode(generated[0], skip_special_tokens = True).split('Password:')[-1].strip()

            metrics = {
                'char_sim': char_similarity(prediction, entry['output']),
                'levenshtein': levenshtein(prediction, entry['output']),
                'jaccard': jaccard(prediction, entry['output'])
            }

            results.append({
                'hash': entry['input'],
                'prediction': prediction,
                'truth': entry['output'],
                'scores': {
                    'aggregate': sum(metrics.values()) / 3,
                    **metrics
                }
            })

        yaml.dump(results, outpath.open('w'), sort_keys = False)
        heimdahl(f'[SAVED EVAL RESULTS] ({outpath.relative_to(self.__class__._ROOT)})', unveil = self.vb)


__all__ = ['GPT2HashTuner']

if __name__ == '__main__':

    for i in range(4):
        tuner = GPT2HashTuner(
            'data/training/shards/shard_sample.jsonl',
            # limit = 100_000_000,
            batch = 32, 
            fp16 = True,
            checkpoints = 3,
            load_best = False,
            vb = True,
            iters = i
        )
        heimdahl(f'[EPOCH] {i + 1}', unveil = True, threat = 3)
        tuner.train(resume = True)
        tuner.evaluate()
        torch.cuda.empty_cache()