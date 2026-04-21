# ==========================
# MINIGPT GENERATION SCRIPT
# ==========================
import torch
import torch.nn.functional as F
import re

# Load your modules (adjust paths if needed)
from tokenizer import SimpleTokenizer
from model import MiniGPT
from config import Config

print("🔥 Loading MiniGPT...")
print("="*50)

# Load data and tokenizer
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = SimpleTokenizer(text)
Config.vocab_size = len(tokenizer.vocab)

print(f"✅ Tokenizer loaded (vocab: {len(tokenizer.vocab)})")
print(f"✅ Device: {Config.device}")

# Load model
model = MiniGPT().to(Config.device)
model.load_state_dict(torch.load('minigpt_best.pt', map_location=Config.device))
model.eval()
print("✅ Model loaded!")

print("\n" + "="*60)
print("🎭 MINIGPT GENERATION READY!")
print("="*60)

def generate_text(model, tokenizer, prompt="", max_new_tokens=200, temperature=0.8):
    
    # Ensure non-empty input
    if prompt == "":
        context = torch.zeros((1, 1), dtype=torch.long).to(Config.device)
    else:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(Config.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -Config.block_size:]
            logits, _ = model(idx_cond)

            # safety check
            if logits.shape[1] == 0:
                break

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())

# 1. AUTOMATIC GENERATION
print("\n🚀 GENERATING AUTOMATIC TEXT...")
print("-" * 40)

context = torch.tensor([[tokenizer.stoi.get('<start>', 0)]], dtype=torch.long).to(Config.device)
generated_text = generate_text(model, tokenizer, max_new_tokens=200, temperature=0.8)

print("\nGenerated text:")
print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
print("\n" + "="*60)

# 2. INTERACTIVE MODE
print("\n🎮 INTERACTIVE MODE (type 'quit' to exit)")
print("💡 Try prompts like: 'King Henry', 'To be', 'Romeo'")
print("-" * 60)

while True:
    try:
        prompt = input("\n📝 Enter prompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            print("⚠️  Please enter a prompt!")
            continue
            
        print("🤖 Generating...", end=" ")
        completion = generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7)
        
        # Clean completion (remove prompt)
        full_text = completion
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_text = full_text[prompt_tokens:] if len(full_text) > prompt_tokens else ""
        
        print(f"\n✅ Completion ({len(completion_text)} chars):")
        print(f"   '{completion_text[:300]}...'")
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted! Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        continue

print("\n🎉 Session complete!")