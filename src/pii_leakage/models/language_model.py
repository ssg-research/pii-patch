# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
# from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip
from transformer_lens import HookedTransformer


@dataclass
class GeneratedText:
    text: str  # the generated text
    score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])

    def __len__(self):
        return len(self.data) if self.data is not None else 0


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """A wrapper class around a huggingface LM."""
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """Gets the maximum size of the context"""
        if hasattr(self._lm.config, "n_positions"):
            return self._lm.config.n_positions
        else:
            return 1e12
 
    @abstractmethod
    def tokenizer(self):
        """Returns this model's tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def unload(self, verbose: bool = False) -> None:
        """Unloads the model and tokenizer from memory and clears GPU cache."""
        if verbose:
            print("Unloading model from memory...")
        
        # Delete model
        if self._lm is not None:
            # Unwrap DataParallel if wrapped
            if hasattr(self._lm, 'module'):
                del self._lm.module
            del self._lm
            self._lm = None
        
        # Delete tokenizer
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear additional data
        self._data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if verbose:
            print("Model unloaded successfully.")

    def load(self, verbose: bool = False) -> "LanguageModel":
        """Loads the model and tokenizer from the checkpoint."""
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = tokenizer.from_pretrained(
            self.model_args.architecture, use_fast=self.model_args.tokenizer_use_fast
        )
        loaded_peft_model = False

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'."
                )

            if is_valid_url(self.model_args.model_ckpt):
                self.model_args.model_ckpt = download_and_unzip(
                    self.model_args.model_ckpt
                )
            if self.model_args.peft == "none":
                self._lm = model_cls.from_pretrained(
                    self.model_args.model_ckpt, return_dict=True
                )
            elif self.model_args.peft == "lora":
                from peft.peft_model import PeftModel
                self._lm = model_cls.from_pretrained(
                    self.model_args.architecture, return_dict=True
                )
                print(f"Load peft model: lora..")
                self._lm = PeftModel.from_pretrained(
                    self._lm,
                    self.model_args.model_ckpt,
                    return_dict=True,
                )
                loaded_peft_model = True
            else:
                raise NotImplementedError(f"peft mode: {self.model_args.peft}")
            self._lm.eval()
        elif (
            self.model_args.pre_trained
        ):  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(
                    f"> Loading a public, pre-trained {self.model_args.architecture} model."
                )
            self._lm = model_cls.from_pretrained(
                self.model_args.architecture, return_dict=True
            ).eval()
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(
                    f"> Loading an uninitialized {self.model_args.architecture} model."
                )
            self._lm = model_cls(config=self.get_config())

        if self.model_args.peft != "none" and not loaded_peft_model:
            # need to change for different models
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            if self.model_args.peft == "lora":
                from peft import LoraConfig, PromptTuningConfig, PeftModel

                peft_config = LoraConfig(
                    lora_alpha=self.model_args.lora_alpha,
                    lora_dropout=self.model_args.lora_dropout,
                    r=self.model_args.lora_r,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules,
                )
            if peft_config is not None:
                from peft import get_peft_model

                self._lm = get_peft_model(self._lm, peft_config)
                self._lm.print_trainable_parameters()

        self._tokenizer.padding_side = "right"
        
        # Properly handle pad_token for Llama models which don't have one by default
        if self._tokenizer.pad_token is None:
            print("Adding pad_token to tokenizer...")
            # Add a proper PAD token instead of using EOS token
            self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Tokenizer vocab size after adding pad_token: {len(self._tokenizer)}")
            
            # Resize model embeddings to accommodate the new token
            print("Resizing model embeddings...")
            old_vocab_size = self._lm.config.vocab_size
            self._lm.resize_token_embeddings(len(self._tokenizer))
            print(f"Model vocab size: {old_vocab_size} -> {self._lm.config.vocab_size}")
            
            # Set the pad_token_id in the model config
            self._lm.config.pad_token_id = self._tokenizer.pad_token_id
        else:
            # Fallback for models that already have a pad_token
            self._lm.config.pad_token_id = self._tokenizer.pad_token_id
        
        # Device placement - always move to single device first
        self._lm.to(self.env_args.device)
        
        if self.env_args.verbose:
            num_gpus = torch.cuda.device_count()
            print(f"Model loaded on device: {self.env_args.device}")
            print(f"Available GPUs: {num_gpus}")
            if hasattr(self.env_args, 'use_multi_gpu'):
                print(f"Multi-GPU setting: {getattr(self.env_args, 'use_multi_gpu', False)}")
        
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[: seq.index(substring) + len(substring)]
        input_ids = (
            torch.tensor(self._tokenizer.encode(txt, truncation=True))
            .unsqueeze(0)
            .to(self.env_args.device)
        )
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, : input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """Predicts the top-1 most probable next tokens."""
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(
            SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64)
        )
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate(
        self, sampling_args: SamplingArgs, hooked_lm: HookedTransformer = None
    ) -> GeneratedTextList:
        """Generates text using the sampling args."""
        print('Starting generation...')

        # Use smaller batch sizes for high sampling rates to prevent OOM
        max_batch_size = self.env_args.eval_batch_size
        if sampling_args.N > 200:
            effective_batch_size = max(1, max_batch_size // 4)
        elif sampling_args.N > 100:
            effective_batch_size = max(1, max_batch_size // 2)
        else:
            effective_batch_size = max_batch_size

        # Encode the input prompt once
        prompts: List[str] = (
            ["The name on the passport"]
            if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt]
        )

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        # Keep base tensors on CPU to reduce GPU memory pressure
        base_input_ids = inputs["input_ids"].detach()
        base_attention_mask = inputs["attention_mask"].detach()
        
        # Clean up initial inputs
        del inputs

        def generate_batch(input_ids, attention_mask) -> List[GeneratedText]:
            """Helper function to generate a single batch of text."""
            input_len = input_ids.size(1)
            if hooked_lm is not None:
                # Move tensors to GPU just before use and clean up immediately
                gpu_input_ids = input_ids.to(self.env_args.device)
                try:
                    out = hooked_lm.generate(
                        input=gpu_input_ids,
                        max_new_tokens=min(self.n_positions, input_len + sampling_args.seq_len),
                        do_sample=sampling_args.do_sample,
                        top_k=sampling_args.top_k,
                        top_p=sampling_args.top_p,
                        return_type="str",
                    )
                finally:
                    # Immediately clean up GPU tensors
                    del gpu_input_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                generated_texts: List[GeneratedText] = []
                # Create score tensor on CPU to avoid GPU memory accumulation
                score_tensor = torch.tensor(0.0, device='cpu')
                generated_texts.append(GeneratedText(text=out, score=score_tensor))
                # Clean up output string reference
                del out
                return generated_texts
            else:
                out = self._lm.generate(
                    input_ids=input_ids.to(self.env_args.device),
                    attention_mask=attention_mask.to(self.env_args.device),
                    max_new_tokens=min(self.n_positions, input_len + sampling_args.seq_len),
                    do_sample=sampling_args.do_sample,
                    top_k=sampling_args.top_k,
                    top_p=sampling_args.top_p,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                generated_texts: List[GeneratedText] = []
                
                # Process and decode sequences
                decoded_texts = self._tokenizer.batch_decode(
                    out.sequences, skip_special_tokens=False
                )
                
                # Process scores with explicit memory management
                for i, (text, score) in enumerate(zip(decoded_texts, out.scores)):
                    if sampling_args.as_probabilities:
                        processed_score = torch.softmax(score, 1).detach().cpu()
                    else:
                        processed_score = score.detach().cpu()
                    generated_texts.append(GeneratedText(text=text, score=processed_score))
                
                # Explicitly clean up generation outputs to prevent memory accumulation
                del out.sequences, out.scores, decoded_texts
                del out  # Delete the entire output object
                
                return generated_texts

        generated_data: List[GeneratedText] = []
        
        # Calculate number of batches needed
        num_batches = int(np.ceil(sampling_args.N / effective_batch_size))
        
        for batch_idx in tqdm(
            range(num_batches),
            disable=not sampling_args.generate_verbose,
            desc=f"Generating with {'Hooked' if hooked_lm else 'LM'}",
        ):
            # Calculate the actual size for this batch (last batch might be smaller)
            start_idx = batch_idx * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, sampling_args.N)
            current_batch_size = end_idx - start_idx
            
            # Create batch tensors by repeating base tensors for current batch size
            batch_input_ids = base_input_ids.repeat(current_batch_size, 1).detach()
            batch_attention_mask = base_attention_mask.repeat(current_batch_size, 1).detach()
            
            try:
                # Generate batch and immediately process results
                batch_results = generate_batch(batch_input_ids, batch_attention_mask)
                generated_data.extend(batch_results)
                
                # Clean up batch results immediately
                del batch_results
                
            finally:
                # Always clean up batch tensors
                del batch_input_ids, batch_attention_mask
                
                # Force garbage collection and CUDA cache cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if batch_idx % 3 == 0:  # More frequent cleanup for better memory management
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

        # Final cleanup of base tensors
        del base_input_ids, base_attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GeneratedTextList(data=generated_data)

    def tokenize_datasets(
        self, datasets: List[RealDataset], column_name="text", pre_remove_columns=False
    ) -> List:
        """Tokenizes the 'text' column of a list of dataset using this model's tokenizer"""
        def tokenize_function(examples):
            # Tokenize the text
            result = self._tokenizer(
                examples[column_name],
                truncation=True,
                max_length=self.model_args.tokenizer_max_length,
                padding=False,  # Let the data collator handle padding
            )
            # For causal language modeling, DataCollatorForLanguageModeling will automatically 
            # create labels from input_ids, so we don't need to create them here
            return result

        processed_datasets = []
        for dataset in datasets:
            hf_dataset = dataset.get_hf_dataset()
            if pre_remove_columns:
                hf_dataset = hf_dataset.remove_columns(
                    [c for c in hf_dataset.column_names if c not in [column_name]]
                )

            hf_dataset = hf_dataset.map(tokenize_function, batched=True)

            if pre_remove_columns:
                # Keep only the columns needed for language modeling
                columns_to_keep = ["input_ids", "attention_mask", "labels"]
                # Check which columns actually exist in the dataset
                available_columns = hf_dataset.column_names
                columns_to_keep = [col for col in columns_to_keep if col in available_columns]
                
                # Remove all columns except the ones we need
                columns_to_remove = [col for col in available_columns if col not in columns_to_keep]
                if columns_to_remove:
                    print(f"Removing columns: {columns_to_remove}")
                    print(f"Keeping columns: {columns_to_keep}")
                    hf_dataset = hf_dataset.remove_columns(columns_to_remove)
            processed_datasets.append(hf_dataset)
        return processed_datasets

    def perplexity(
        self,
        data: Union[list, str],
        offset=0,
        max_length=0,
        apply_exp=True,
        verbose=True,
        return_as_list: bool = False,
    ) -> float:
        """Compute the perplexity of the model on a string."""
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
            input_ids = (
                torch.tensor(
                    self._tokenizer.encode(
                        txt,
                        truncation=True,
                        max_length=self.model_args.tokenizer_max_length,
                    )
                )
                .unsqueeze(0)
                .to(self.env_args.device)
            )
            target_ids = input_ids.clone()

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = target_ids.size(1) - offset
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length :] = -100
                tgt_len = max_length

            with torch.no_grad():
                outputs = self._lm(input_ids, labels=target_ids)
            loss, logits = outputs[:2]
            if return_as_list:
                nlls.append(loss.cpu().detach().float())
            else:
                nlls.append(loss.cpu().detach().float())
                ctr += tgt_len

        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    def _fine_tune_dp(
        self,
        train_dataset: RealDataset,
        eval_dataset: RealDataset,
        train_args: TrainerArgs,
        privacy_args: PrivacyArgs,
    ):

        with train_args.main_process_first(desc="Tokenizing datasets"):
            eval_dataset = eval_dataset.shuffle().select(
                list(range(train_args.limit_eval_dataset))
            )
            assert (
                not train_args.remove_unused_columns
            ), "DP does not support remove_unused_columns which can not work with GradSampleModule"
            print("Tokenizing Train and Eval Datasets ..")
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets(
                [train_dataset, eval_dataset],
                pre_remove_columns=not train_args.remove_unused_columns,  # if trainer does not remove (e.g., for DP), we will remove columns here (hard-coded may not apply for all)
            )
            print("done")

        # self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(
            self._tokenizer
        )

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp,
        )

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
        )

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({"final_epsilon_prv": eps_prv, "final_epsilon_rdp": eps_rdp})
            print(f"saving model..")
            trainer.save_model()
            # trainer.model is a GradSampleModule.
            trainer.model._module.save_pretrained(trainer.args.output_dir)
            trainer.model._module.config.save_pretrained(trainer.args.output_dir)
            trainer.model._module.generation_config.save_pretrained(
                trainer.args.output_dir
            )
        self._lm.eval()

    def fine_tune(
        self,
        train_dataset,
        eval_dataset,
        train_args: TrainerArgs,
        privacy_args: PrivacyArgs,
    ):
        """Fine-Tune the LM with/without DP"""
        if privacy_args.target_epsilon > 0:
            # return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
            return self._fine_tune_fast_dp(
                train_dataset, eval_dataset, train_args, privacy_args
            )
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune_fast_dp(
        self,
        train_dataset,
        eval_dataset,
        train_args: TrainerArgs,
        privacy_args: PrivacyArgs,
        extra_callbacks: List[TrainerCallback] = None,
    ):
        """Fine-Tune the model and save checkpoints to output directory"""
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=False
        )

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(
            list(range(train_args.limit_eval_dataset))
        )
        
        # For differential privacy, we need to disable remove_unused_columns in the trainer
        # and handle column removal manually during tokenization
        original_remove_unused_columns = train_args.remove_unused_columns
        train_args.remove_unused_columns = False
        
        # train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        train_dataset, eval_dataset = self.tokenize_datasets(
            [train_dataset, eval_dataset],
            pre_remove_columns=True,  # Always remove columns for fast-DP to avoid conflicts
        )
        print("Done Tokenizing!")
        print("model:", self._lm)
        
        # For differential privacy training, we NEVER use DataParallel
        # The fast-DP library has fundamental incompatibilities with DataParallel
        verbose = getattr(self.env_args, 'verbose', False)
        if verbose:
            print("DIFFERENTIAL PRIVACY MODE: Using single GPU only")
            print(f"Model device: {next(self._lm.parameters()).device}")
            # Import torch at the module level is available
            import torch as torch_module
            print(f"Model is DataParallel: {isinstance(self._lm, torch_module.nn.DataParallel)}")
        
        # Ensure model is not wrapped in DataParallel (unwrap if needed)
        import torch as torch_module
        if isinstance(self._lm, torch_module.nn.DataParallel):
            if verbose:
                print("WARNING: Unwrapping DataParallel for differential privacy")
            self._lm = self._lm.module
        
        # Ensure model is on primary device
        self._lm.to(self.env_args.device)
        
        # Store original training args that we'll modify for differential privacy
        original_dataloader_num_workers = getattr(train_args, 'dataloader_num_workers', None)
        original_remove_unused_columns = train_args.remove_unused_columns
        
        # Force single GPU training to prevent DataParallel wrapping
        train_args.local_rank = -1  # Disable distributed training
        train_args.dataloader_num_workers = 0  # Avoid multiprocessing issues
        train_args.remove_unused_columns = False  # Required for fast-DP compatibility
        
        # Override CUDA_VISIBLE_DEVICES to completely hide other GPUs
        import os
        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        # Extract device index properly - env_args.device is a string like "cuda" or "cuda:0"
        if str(self.env_args.device).startswith('cuda:'):
            device_index = str(self.env_args.device).split(':')[1]
        else:
            device_index = '0'  # Default to GPU 0 for "cuda"
        
        os.environ['CUDA_VISIBLE_DEVICES'] = device_index
        
        # Also override torch.cuda.device_count to return 1 for this process
        import torch.cuda
        original_device_count = torch.cuda.device_count
        torch.cuda.device_count = lambda: 1  # Force single device detection
        
        if verbose:
            print(f"Set CUDA_VISIBLE_DEVICES to: {device_index}")
            print(f"Device type: {type(self.env_args.device)}, device value: {self.env_args.device}")
            print(f"Training args local_rank: {getattr(train_args, 'local_rank', 'not set')}")
            print(f"Override torch.cuda.device_count: {torch.cuda.device_count()}")
        
        trainer = Trainer(
            model=self._lm,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=extra_callbacks,
        )
        
        # Force single device for the trainer model - check again after trainer creation
        if verbose:
            print(f"After trainer creation - Model type: {type(trainer.model)}")
            print(f"After trainer creation - Is DataParallel: {isinstance(trainer.model, torch.nn.DataParallel)}")
        
        # Unwrap DataParallel if trainer applied it
        if isinstance(trainer.model, torch.nn.DataParallel):
            if verbose:
                print("WARNING: Trainer applied DataParallel - unwrapping for differential privacy")
            trainer.model = trainer.model.module
            # Also update our reference
            self._lm = trainer.model

        params = tuple(param for param in self._lm.parameters() if param.requires_grad)
        names = tuple(
            name for name, param in self._lm.named_parameters() if param.requires_grad
        )
        num_trainable_params = sum(param.numel() for param in params)
        print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
        print(
            f"Number of total params: {sum(param.numel() for param in self._lm.parameters()) / 1e6:.3f} million"
        )

        # print(json.dumps(names, indent=4))

        # TODO: Using a single gigantic parameter group is okay only when `weight_decay` is 0.
        #   Biases and LM parameters should not be decayed perhaps even with privacy.
        optimizer = torch.optim.AdamW(
            params=params,
            lr=train_args.learning_rate,
            betas=(train_args.adam_beta1, train_args.adam_beta2),
            eps=train_args.adam_epsilon,
        )
        trainer.optimizer = optimizer

        # Create the lr_scheduler.
        try:
            num_GPUs = torch.distributed.get_world_size()
        except:
            num_GPUs = 2

        # if train_args.logical_batch_size!=None:
        #    trainer.args.gradient_accumulation_steps=train_args.logical_batch_size/train_args.per_device_train_batch_size/num_GPUs
        # else:
        logical_batch_size = (
            trainer.args.gradient_accumulation_steps
            * train_args.per_device_train_batch_size
            * num_GPUs
        )

        num_update_steps_per_epoch = (
            len(trainer.get_train_dataloader())
            // trainer.args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
        # if train_args.lr_decay:
        #    trainer.lr_scheduler = get_linear_schedule_with_warmup(
        #        trainer.optimizer,
        #        num_warmup_steps=train_args.warmup_steps,
        #        num_training_steps=t_total,
        #    )
        # else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _: 1.0
        )

        from fastDP import PrivacyEngine

        # Ensure model is on the correct device before PrivacyEngine initialization
        if hasattr(self._lm, 'module'):
            # Unwrap DataParallel if somehow applied earlier
            self._lm = self._lm.module
        self._lm.to(self.env_args.device)

        privacy_engine = PrivacyEngine(
            module=self._lm,
            batch_size=logical_batch_size,
            sample_size=len(train_dataset),
            epochs=train_args.num_train_epochs,
            max_grad_norm=privacy_args.max_grad_norm_dp,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            # accounting_mode=privacy_args.accounting_mode,
            # clipping_mode=privacy_args.clipping_mode,
            # clipping_fn=privacy_args.clipping_fn,
            # clipping_style=privacy_args.clipping_style,
            # origin_params=['wte','wpe'],
            origin_params=None,
            num_GPUs=1,  # Force single GPU for differential privacy to avoid hook conflicts
            torch_seed_is_fixed=True,
        )

        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        # print('privacy_args: ')
        # print(json.dumps(privacy_args.__dict__, indent=4))
        privacy_engine.attach(optimizer) 

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            # Restore original settings
            train_args.remove_unused_columns = original_remove_unused_columns
            if original_dataloader_num_workers is not None:
                train_args.dataloader_num_workers = original_dataloader_num_workers
            
            # Restore CUDA_VISIBLE_DEVICES
            import os
            if original_cuda_visible_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            
            # Restore original device count function
            import torch.cuda
            torch.cuda.device_count = original_device_count
            
            print(f"saving to {trainer.args.output_dir}")
            trainer.save_model()
            
            # For differential privacy training, we don't use DataParallel
            # so no need to unwrap - just save directly
            model_to_save = trainer.model
            model_to_save.save_pretrained(trainer.args.output_dir)
            model_to_save.config.save_pretrained(trainer.args.output_dir)
            model_to_save.generation_config.save_pretrained(trainer.args.output_dir)
        self._lm.eval()

    def _fine_tune(
        self,
        train_dataset,
        eval_dataset,
        train_args: TrainerArgs,
        extra_callbacks: List[TrainerCallback] = None,
    ):
        """Fine-Tune the model and save checkpoints to output directory"""
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)]
        # extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
        #                                                num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, 
            mlm=False,
            pad_to_multiple_of=8,  # Helps with efficiency on GPU
            return_tensors="pt"    # Ensure pytorch tensors are returned
        )

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(
            list(range(train_args.limit_eval_dataset))
        )
        train_dataset, eval_dataset = self.tokenize_datasets(
            [train_dataset, eval_dataset],
            pre_remove_columns=True  # Remove unused columns to avoid data collator issues
        )
        print("Done Tokenizing!")
        
        # Multi-GPU handling for non-differential privacy training
        num_gpus = torch.cuda.device_count()
        use_multi_gpu = getattr(self.env_args, 'use_multi_gpu', None)
        force_single_gpu = getattr(self.env_args, 'force_single_gpu', False)
        
        # Auto-detect if use_multi_gpu is None
        if use_multi_gpu is None:
            use_multi_gpu = num_gpus > 1  # Default to multi-GPU if available
        
        if force_single_gpu or num_gpus <= 1:
            # Single GPU mode - model is already on the correct device
            if self.env_args.verbose:
                print(f"Training with single GPU: {self.env_args.device}")
        elif use_multi_gpu and num_gpus > 1:
            # Multi-GPU mode - use DataParallel
            try:
                self._lm = torch.nn.DataParallel(self._lm)
                if self.env_args.verbose:
                    print(f"Training with DataParallel on {num_gpus} GPUs (primary: {self.env_args.device})")
            except Exception as e:
                if self.env_args.verbose:
                    print(f"Failed to set up DataParallel: {e}")
                    print(f"Falling back to single GPU training on {self.env_args.device}")
        else:
            if self.env_args.verbose:
                print(f"Training with single GPU: {self.env_args.device}")

        # For compatibility with DataParallel, disable automatic column removal
        original_remove_unused_columns = train_args.remove_unused_columns
        train_args.remove_unused_columns = False
        if self.env_args.verbose:
            print(f"Set remove_unused_columns to: {train_args.remove_unused_columns}")

        trainer = Trainer(
            model=self._lm,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=extra_callbacks,
        )

        try:
            trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        finally:
            # Restore original settings
            train_args.remove_unused_columns = original_remove_unused_columns
            
            print(f"saving to {trainer.args.output_dir}")
            trainer.save_model()
            
            # Handle DataParallel wrapper for saving
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(trainer.args.output_dir)
            model_to_save.config.save_pretrained(trainer.args.output_dir)
            model_to_save.generation_config.save_pretrained(trainer.args.output_dir)
            
            # Unwrap DataParallel for future use
            if hasattr(trainer.model, 'module'):
                self._lm = trainer.model.module
        self._lm.eval()
