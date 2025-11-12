# BEAM: Bottleneck Extraction for Attention Mechanism

This repository provides an open-source implementation of BEAM (Bottleneck Extraction for Attention Mechanism), a lightweight adapter module designed to mitigate the attention fade phenomenon in long-context large language models. By applying feature extraction on attention outputs, BEAM produces corrective perturbations that enhance model performance on long-context understanding tasks, all while adding less than 0.05% additional parameters to existing pre-trained LLMs.

## Problem Overview

Long-context large language models advertise context windows beyond 100K tokens, but performance often degrades after only a few thousand tokens due to a phenomenon called **attention fade**. This occurs because as sequence length grows, softmax-normalized attention weights become increasingly diffuse, making it difficult for the model to maintain sharp focus on specific information across long contexts. As a result, only a small fraction of the advertised context window is actually usable in practice.

This repository addresses the issue by introducing BEAM, a lightweight adapter that:

- Extracts important attention features through a bottleneck mechanism.
- Produces corrective perturbations tailored to each attention layer.
- Enhances long-context understanding without retraining the base model.
- Maintains compatibility with existing pre-trained LLMs.

## Approach

BEAM consists of two main components:

### 1. Bottleneck Network (Feature Extraction)
A lightweight feature extraction module that compresses attention outputs to a reduced dimension, applies a nonlinear activation, and projects back to the original dimension. This forces the adapter to distill essential contextual information into a low-rank representation.

### 2. Scalar Network
A small MLP that computes a learnable scalar value to dynamically scale the perturbation across all positions in the sequence.

The bottleneck output is scaled by the scalar and added as a residual perturbation to the original attention output, enabling the model to recover and amplify salient long-range signals from increasingly diffuse attention distributions.

### Key Characteristics

- **Parameter Efficient**: Adds less than 0.05% trainable parameters to the base model
- **Modular Design**: Can be plugged into any pre-trained transformer without architectural changes
- **Frozen Backbone**: Base model weights remain frozen during fine-tuning
- **Attention-Local**: Operates directly on post-softmax attention outputs

## Setup

### Install conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
sh Miniconda3-latest-Linux-x86_64.sh
```

Source to ensure `conda` CLI is available:

```bash
source ~/.bashrc
```

If you still encounter `conda: command not found`, manually source the conda initialization:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

## Clone this repo

```bash
git clone https://github.com/anonymousbeam/BEAM.git && \
cd BEAM
```

## Create conda env

```bash
conda create --prefix ./.conda python=3.10 && \
conda activate ./.conda
```

## Install CUDA toolkit

```bash
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
```

## Install requirements

```bash
pip install -r requirements.txt
```

## Training

### Dataset Preparation

BEAM is trained on the LongAlpaca dataset, which contains long-context instruction-following examples designed to teach models to understand and reason over extended sequences. The dataset is automatically loaded during training via the `LongAlpacaDataset` class.

For information about dataset preparation and structure, refer to the [`beam/dataset/`](beam/dataset/) folder.

### Model Preparation

BEAM is compatible with most HuggingFace-compatible causal language models. We recommend using instruction-tuned models to preserve conversational capabilities. As per the research, compatible models include:

- `Qwen2.5-1.5B-Instruct`
- `Llama-3.1-8B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Falcon3-10B-Instruct`

### Training

BEAM training is performed using the `beam/main.py` script. 

Within the script, the model name has been mapped and can be adjusted for your use case. The default is as follows:

```python
model_paths = {
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "falcon_10b": "tiiuae/Falcon3-10B-Instruct",
}
```

Below is an example command to BEAM on Llama-3.1-8B-Instruct:

```bash
CUDA_VISIBLE_DEVICES=0 python -m beam.main \
    --model_name llama_8b \
    --num_epochs 1 \
    --batch_size 64 \
    --microbatch_size 1 \
    --lr 2e-5 \
    --bottleneck_dim 4 \
    --run_name "llama_3.1_8b-BEAM" \
    --checkpoint_amount 2 \
    --use_wandb true
```

### Configuration Parameters

- `model_name`: Model identifier (qwen, llama_3b, llama_8b, falcon_10b)
- `num_epochs`: Number of training epochs (default: 1)
- `batch_size`: Effective batch size with gradient accumulation (default: 8)
- `microbatch_size`: Per-GPU micro-batch size (default: 1)
- `lr`: Learning rate (default: 2e-5)
- `bottleneck_dim`: Dimension of the bottleneck layer, controlling compression (default: 4)
- `run_name`: Name of run, used for checkpointing and wandb, if applicable (default: BEAM-run)
- `checkpoint_amount`: Number of checkpoints to save during training, 0 implies only the final checkpoint saved (default: 0)
- `use_wandb`: Enable Weights & Biases logging (default: True)

## Inference

### Using the OpenAI-Compatible API Server

BEAM includes a FastAPI server that provides OpenAI-compatible chat completion endpoints, allowing integration with evaluation benchmarks.

Start the server with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m beam.infer \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --checkpoint checkpoints/llama_8b_FINAL-CHECKPOINT.pt
```

The server will automatically find an available port and start listening for requests.

### Making Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is attention fade?"}
        ],
        "stream": False
    }
)

print(response.json()['choices'][0]['message']['content'])
```

## License

This codebase is licensed under Apache 2.0 as given in LICENSE.
