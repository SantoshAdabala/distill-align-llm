"""Interactive chat with your fine-tuned model (Mac-compatible)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon (MPS) backend")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU (this will be slow)")

# Load base model — no quantization on Mac
print("Loading model... (this may take a minute)")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map=device if device != "mps" else None,
    dtype=torch.float16,
    trust_remote_code=True,
)

# Move to MPS manually (device_map doesn't support MPS well)
if device == "mps":
    model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load your fine-tuned adapter (pick one)
print("Loading adapter...")
model = PeftModel.from_pretrained(model, "./outputs/sft/final_adapter")
# model = PeftModel.from_pretrained(model, "./outputs/dpo/dpo_adapter")  # or DPO

model.eval()
print("\nChat with your model (type 'quit' or 'exit' to stop)\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("quit", "exit"):
        break
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Model: {response}\n")
