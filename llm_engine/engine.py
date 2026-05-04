import gc
from argparse import ArgumentParser
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Check if llama-cpp-python is available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. GGUF models will not be available.")
    print("Install with: pip install llama-cpp-python --break-system-packages")


# arguments
parser = ArgumentParser()
parser.add_argument('--gpu_ids', nargs='+', default=['0','1','2','3'])
parser.add_argument('--quantization', default=False, action='store_true')
parser.add_argument('--model_path', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1')
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--eos_token_id', type=int, default=32021)
parser.add_argument('--pad_token_id', type=int, default=32021)
parser.add_argument('--num_return_sequences', type=int, default=1)
parser.add_argument('--max_repeat_prompt', type=int, default=10)
parser.add_argument('--n_gpu_layers', type=int, default=0, help='Number of layers to offload to GPU for GGUF models (0 = CPU only)')
parser.add_argument('--n_ctx', type=int, default=2048, help='Context window size for GGUF models')
parser.add_argument('--n_threads', type=int, default=8, help='Number of threads for GGUF models')
args = parser.parse_args()


# Detect if model is GGUF format
def is_gguf_model(model_path):
    """Check if the model path points to a GGUF file"""
    return model_path.endswith('.gguf') or os.path.isfile(model_path) and '.gguf' in model_path.lower()


USE_GGUF = is_gguf_model(args.model_path)

if USE_GGUF:
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for GGUF models. "
            "Install it with: pip install llama-cpp-python --break-system-packages"
        )
    print(f"Loading GGUF model from: {args.model_path}")
    
    # Load GGUF model with llama.cpp
    model = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True
    )
    tokenizer = None  # GGUF models have tokenizer built-in
    device = None
    print(f"GGUF model loaded successfully")
    print(f"Context window: {args.n_ctx}")
    print(f"GPU layers: {args.n_gpu_layers}")
    print(f"Threads: {args.n_threads}")

else:
    print(f"Loading transformers model from: {args.model_path}")
    
    # cuda devices
    if torch.cuda.is_available():
        if args.gpu_ids is None:
            device = torch.device("cuda")
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            device = torch.device(f"cuda:{args.gpu_ids[0]}")
            gpu_ids = args.gpu_ids
        print(f"Using GPU(s): {gpu_ids}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        gpu_ids = []
        print("CUDA is not available. Using CPU.")

    # quantization
    if args.quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quantization_config = None

    # load model
    pretrained_model_path = args.model_path
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path,
        quantization_config=quantization_config,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path,
    )
    print("Transformers model loaded successfully")


# flask API
app = Flask(__name__)
CORS(app)

@app.route(f'/completions', methods=['POST'])
def completions():
    content = request.json
    prompt = content['prompt']

    repeat_prompt = content.get('repeat_prompt', 1)

    # parameters - always set defaults first
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    do_sample = args.do_sample
    top_k = args.top_k
    top_p = args.top_p
    num_return_sequences = args.num_return_sequences
    eos_token_id = args.eos_token_id
    pad_token_id = args.pad_token_id
    max_repeat_prompt = args.max_repeat_prompt
    
    # Override with params if provided
    if 'params' in content and content['params'] is not None:
        params: dict = content.get('params')
        max_new_tokens = params.get('max_new_tokens', max_new_tokens)
        temperature = params.get('temperature', temperature)
        do_sample = params.get('do_sample', do_sample)
        top_k = params.get('top_k', top_k)
        top_p = params.get('top_p', top_p)
        num_return_sequences = params.get('num_return_sequences', num_return_sequences)
        eos_token_id = params.get('eos_token_id', eos_token_id)
        pad_token_id = params.get('pad_token_id', pad_token_id)
        max_repeat_prompt = params.get('max_repeat_prompt', max_repeat_prompt)

    if USE_GGUF:
        # GGUF model generation
        try:
            # Format prompt for chat
            if isinstance(prompt, str):
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            else:
                formatted_prompt = prompt
            
            # Ensure parameters are valid (not None)
            generation_params = {
                'max_tokens': max_new_tokens if max_new_tokens is not None else 512,
                'temperature': temperature if temperature is not None else 0.8,
                'top_k': top_k if top_k is not None else 30,
                'top_p': top_p if top_p is not None else 0.9,
                'echo': False,
                'stop': ["</s>", "<|end|>", "<|user|>"]
            }
            
            content_list = []
            for _ in range(repeat_prompt * num_return_sequences):
                output = model(
                    formatted_prompt,
                    **generation_params
                )
                content_list.append(output['choices'][0]['text'].strip())
            
            return jsonify({'content': content_list})
            
        except Exception as e:
            print(f"Error during GGUF generation: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    else:
        # Transformers model generation
        prompt_formatted = [{'role': 'user', 'content': prompt}]
        
        while True:
            inputs = tokenizer.apply_chat_template(prompt_formatted, 
                                                   add_generation_prompt=True, 
                                                   return_tensors='pt')       
            inputs = torch.vstack([inputs] * repeat_prompt).to(model.device)

            try:
                output = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id
                )

            except torch.cuda.OutOfMemoryError as e:
                # clear cache
                gc.collect()
                if torch.cuda.device_count() > 0:
                    torch.cuda.empty_cache()
                continue
            
            content_list = []
            for i, out_ in enumerate(output):
                content_list.append(tokenizer.decode(output[i, len(inputs[i]):], skip_special_tokens=True))
            
            # clear cache
            gc.collect()
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()

            return jsonify({'content': content_list})


if __name__ == '__main__':
    app.run(host=args.host, port=args.port)