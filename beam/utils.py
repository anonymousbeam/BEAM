import inspect
import math
import os
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path

from beam.layers import BEAM
import torch

def add_beam(model, bottleneck_dim=4):
    for layer in model.model.layers:
        layer.self_attn = BEAM(
            layer.self_attn,
            device=model.device,
            dtype=model.dtype,
            rank=bottleneck_dim,
            alpha=bottleneck_dim,
        )
    for name, param in model.named_parameters():
        if 'beam' not in name:
            param.requires_grad = False

    return model


def print_trainable_parameters(model, verbose=True):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if verbose:
                print(f'Trainable: {name} | shape: {tuple(param.shape)}')
    print(
        f'\nTotal trainable parameters: {trainable_params:,} / {total_params:,} '
        f'({100 * trainable_params / total_params:.4f}%)'
    )


def prepare_collate_fn(tokenizer):
    def collate_fn(batch, tokenizer):
        input_ids = [example['input_ids'] for example in batch]
        attention_masks = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]

        max_length = max(len(x) for x in input_ids)

        def pad_sequence(seq, pad_value):
            return torch.cat(
                [seq, torch.full((max_length - len(seq),), pad_value, dtype=seq.dtype)]
            )

        input_ids = torch.stack(
            [pad_sequence(x, tokenizer.pad_token_id) for x in input_ids]
        )
        attention_mask = torch.stack([pad_sequence(x, 0) for x in attention_masks])
        labels = torch.stack([pad_sequence(x, -100) for x in labels])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    return partial(
        collate_fn,
        tokenizer=tokenizer,
    )


def save_beam_weights(model, model_name, output_dir='checkpoints'):
    os.makedirs(output_dir, exist_ok=True)

    beam_weights = {}
    for name, param in model.named_parameters():
        if 'beam' in name:
            beam_weights[name] = param

    checkpoint_path = os.path.join(output_dir, f'{model_name}.pt')
    torch.save(beam_weights, checkpoint_path)

    print(f'[Checkpoint] Saved BEAM weights to {checkpoint_path}')


def capture_hparams():
    """Captures both local and global variables ("hyperparameters") from where this function gets called."""
    caller_frame = inspect.currentframe().f_back
    locals_of_caller = caller_frame.f_locals
    globals_of_caller = caller_frame.f_globals  # Capture globals

    hparams = {}

    def process_variables(var_dict, scope='local'):
        for name, value in var_dict.items():
            # ignore certain stuff
            if name == 'state_dict':
                continue

            if value is None or isinstance(value, (int, float, str, bool, Path)):
                hparams[f'{scope}.{name}'] = value
            elif is_dataclass(value):
                hparams[f'{scope}.{name}'] = asdict(value)
            else:
                hparams[f'{scope}.{name}'] = str(
                    value
                )  # Convert unrecognized types to string

    # Process both local and global variables
    process_variables(locals_of_caller, 'local')
    process_variables(globals_of_caller, 'global')

    return hparams


def get_lr(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def load_beam_weights(model, load_path):
    """
    Load BEAM weights into a model.

    Args:
        model: The model to load BEAM weights into
        load_path: Path where weights are saved
    """
    beam_weights = torch.load(load_path)

    # Track which weights were loaded
    loaded_weights = 0

    # Load weights into model
    for name, param in model.named_parameters():
        check_name = '_orig_mod.' + name
        if check_name in beam_weights:
            param.data.copy_(beam_weights[check_name])
            loaded_weights += 1

    print(f'Loaded {loaded_weights}/{len(beam_weights)} BEAM parameters')
