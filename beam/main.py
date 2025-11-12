import math

import torch
from jsonargparse import CLI
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from dataset.long_alpaca import LongAlpacaDataset
from beam.utils import (
    add_beam,
    capture_hparams,
    get_lr,
    prepare_collate_fn,
    print_trainable_parameters,
    save_beam_weights,
)

model_paths = {
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "falcon_10b": "tiiuae/Falcon3-10B-Instruct",
}

def main(
    model_name: str,
    num_epochs=1,
    batch_size=8,
    microbatch_size=1,
    lr=2e-5,
    bottleneck_dim=4,
    run_name='BEAM-run',
    checkpoint_amount=0,
    use_wandb=True,
):
    model_path = model_paths[model_name]
    torch.manual_seed(999)

    checkpoint_amount = max(checkpoint_amount, 1)

    if use_wandb:
        wandb.init(
            project='BEAM',
            name=run_name,
            config=capture_hparams(),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir=".cache/"
    )

    model = add_beam(model, bottleneck_dim=bottleneck_dim)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_trainable_parameters(model, verbose=False)

    dataset = LongAlpacaDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        collate_fn=prepare_collate_fn(tokenizer),
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = (
        math.ceil(len(dataset) // batch_size) * num_epochs
    )

    model.train()
    model = torch.compile(model)

    accumulation_steps = batch_size // microbatch_size
    num_steps_per_checkpoint = total_steps // checkpoint_amount

    print(f'''========== CONFIGURATIONS ==========
Number of epochs: {num_epochs}
Batch Size: {batch_size}
Microbatch Size: {microbatch_size}
Accumulation Steps (How many microsteps per step): {accumulation_steps}

Number of Training Sequences (1 Epoch): {len(dataset)}
Total Steps: {total_steps}
Learning Rate: {lr}
Extension bottleneck: {bottleneck_dim}
Number of Steps per Checkpoint: {num_steps_per_checkpoint}
====================================
''')

    global_step = 0
    running_loss = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            torch.cuda.empty_cache()

            batch = {
                k: v.to(dtype=torch.long, device=model.device) for k, v in batch.items()
            }

            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            running_loss += loss.detach().float()

            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                new_lr: float = get_lr(
                    optimizer.defaults['lr'],
                    global_step + 1,
                    int(total_steps * 0.1),
                    total_steps,
                    lr / 10,
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print(
                    f'Epoch {epoch}, Step {global_step}, LR: {new_lr}, Loss: {running_loss / accumulation_steps:.4f}'
                )

                if use_wandb:
                    wandb.log(
                        {
                            'loss': running_loss / accumulation_steps,
                            'global_step': global_step,
                            'lr': new_lr,
                        }
                    )
                running_loss = 0

                if (global_step) % num_steps_per_checkpoint == 0:
                    save_beam_weights(model, model_name + "_step-" + str(global_step))
                break
        break

    if use_wandb:
        wandb.finish()

    save_beam_weights(model, model_name + "_FINAL-CHECKPOINT")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    CLI(main, as_positional=False)
