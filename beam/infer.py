import socket
import torch
import time
import uuid
import uvicorn
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from jsonargparse import CLI

from beam.utils import (
    add_beam,
    load_beam_weights,
)

# Global variables to store model and tokenizer
model = None
tokenizer = None


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "BEAM"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=1000, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "user"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model and tokenizer
    global model, tokenizer
    
    print("Loading model and tokenizer...")
    torch.manual_seed(999)
    
    model_name = app.state.model_name
    checkpoint = app.state.checkpoint
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype='auto',
        trust_remote_code=True,
        cache_dir=".cache/",
    )
    
    model = add_beam(model, bottleneck_dim=4)
    load_beam_weights(model, checkpoint)
    print("Checkpoint uploaded:", checkpoint)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Model loaded successfully: {model_name}")
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down...")
    del model
    del tokenizer
    torch.cuda.empty_cache()


app = FastAPI(title="BEAM OpenAI-Compatible API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "BEAM OpenAI-Compatible API Server"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion with OpenAI-compatible interface"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert messages to format expected by tokenizer
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors='pt',
            tokenize=True,
        ).to(model.device)
        
        input_len = prompt.shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_length": input_len + request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
        }
        
        # Handle stop sequences
        if request.stop:
            if isinstance(request.stop, str):
                gen_kwargs["eos_token_id"] = tokenizer.encode(request.stop, add_special_tokens=False)
            elif isinstance(request.stop, list) and len(request.stop) > 0:
                # For simplicity, using the first stop sequence
                gen_kwargs["eos_token_id"] = tokenizer.encode(request.stop[0], add_special_tokens=False)
        
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(prompt, input_len, gen_kwargs, request.model),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            with torch.no_grad():
                output = model.generate(prompt, **gen_kwargs)
            
            generated_tokens = output[0][input_len:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate token usage
            prompt_tokens = input_len
            completion_tokens = len(generated_tokens)
            total_tokens = prompt_tokens + completion_tokens
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=decoded_output),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
            
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(
    prompt: torch.Tensor,
    input_len: int,
    gen_kwargs: Dict[str, Any],
    model_name: str
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses"""
    
    completion_id = f"{uuid.uuid4().hex}"
    created = int(time.time())
    
    try:
        # For streaming, we'll generate token by token
        # Note: This is a simplified implementation.
        # We can look to upgrade with TextIteratorStreamer from transformers.
        
        with torch.no_grad():
            # Generate incrementally
            past_key_values = None
            generated_tokens = []
            current_input = prompt
            
            for _ in range(gen_kwargs.get("max_length", 1000) - input_len):
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature
                if gen_kwargs.get("temperature", 1.0) > 0:
                    logits = logits / gen_kwargs["temperature"]
                
                # Sample next token
                if gen_kwargs.get("do_sample", False):
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.item())
                
                # Decode the new token
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=False)
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Send the token as a stream chunk
                chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": token_text},
                            finish_reason=None
                        )
                    ]
                )
                
                yield f"data: {chunk.model_dump_json()}\n\n"
                
                # Update input for next iteration
                current_input = next_token
        
        # Send final chunk with finish_reason
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {"error": str(e)}
        yield f"data: {json.dumps(error_chunk)}\n\n"

def find_available_port(start_port=8000, max_attempts=100):
    """
    Find an available port starting from start_port.
    
    Args:
        start_port (int): The port to start checking from
        max_attempts (int): Maximum number of ports to try
        
    Returns:
        int: Available port number
        
    Raises:
        RuntimeError: If no available port is found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            # Port is in use, try the next one
            continue
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


def main(
    model_name: str,
    checkpoint: str,
):
    """
    Start the BEAM OpenAI-Compatible API server.
    
    Args:
        model_name: HuggingFace model name or path to load
        checkpoint: Path to the BEAM checkpoint file
    """
    # Set app state before starting server
    app.state.model_name = model_name
    app.state.checkpoint = checkpoint
    
    torch.set_float32_matmul_precision('high')
    try:
        port = find_available_port(start_port=8000)
        print(f"Starting server on port {port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    CLI(main, as_positional=False)
