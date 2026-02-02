"""
Simple test of MedGemma GGUF without chat template
"""

import time
from llama_cpp import Llama

MODEL_PATH = "quantization/outputs/medgemma-4b-q4_k_s-final.gguf"

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=8,
    verbose=False
)

# Simple completion (not chat format)
prompt = """<start_of_turn>user
What are the symptoms of Type 2 diabetes?<end_of_turn>
<start_of_turn>model
"""

print(f"Prompt: What are the symptoms of Type 2 diabetes?")
print("\nGenerating (max 100 tokens)...")

start = time.time()
output = llm(
    prompt,
    max_tokens=100,
    stop=["<end_of_turn>", "\n\n\n"],
    echo=False,
    temperature=0.3
)
elapsed = time.time() - start

text = output['choices'][0]['text']
tokens = output['usage']['completion_tokens']

print(f"\nResponse:\n{text}")
print(f"\n--- Stats ---")
print(f"{tokens} tokens in {elapsed:.2f}s = {tokens/elapsed:.2f} tok/s")
