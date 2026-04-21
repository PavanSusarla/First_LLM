# ==========================
# train.py (Improved Version with Deep Commentary)
# ==========================

# Core PyTorch imports
# ---------------------------------------------------------
# torch → tensor operations
# optim → optimization algorithms (AdamW)
import torch
import torch.optim as optim

# Project modules
# ---------------------------------------------------------
# WHY:
# - Modular design → separation of concerns
# - Easier debugging, testing, and scaling
from tokenizer import SimpleTokenizer
from dataset import TextDataset
from model import MiniGPT
from config import Config


# --------------------------
# Load Data
# --------------------------
print("Loading data...")

# Reading raw text file
# ---------------------------------------------------------
# WHY:
# - Language models require large text corpora
#
# encoding='utf-8':
# - Ensures compatibility with special characters
# - Prevents decoding errors
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Text length: {len(text)} characters")


# --------------------------
# Tokenization
# --------------------------

# Initialize tokenizer
# ---------------------------------------------------------
# WHY:
# - Converts raw text → numerical tokens
#
# DESIGN:
# - Character-level tokenizer (simple, no OOV issues)
tokenizer = SimpleTokenizer(text)
Config.vocab_size = len(tokenizer.vocab)
# Encode text into integers
# ---------------------------------------------------------
# torch.tensor(..., dtype=torch.long):
# - long dtype required for embedding layers
#
# WHY:
# - Embedding indices must be integers
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

print(f"Vocab size: {len(tokenizer.vocab)}")
print(f"Data length: {len(data)} tokens")


# --------------------------
# Train / Validation Split
# --------------------------

# 90-10 split
# ---------------------------------------------------------
# WHY 90% training:
# - Model needs large data to learn patterns
#
# WHY 10% validation:
# - Monitor generalization performance
#
# TRADE-OFF:
# - More training data → better learning
# - More validation data → better evaluation
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# Dataset wrappers
# ---------------------------------------------------------
# WHY:
# - Provides batch generation logic
# - Keeps training loop clean
train_dataset = TextDataset(train_data)
val_dataset = TextDataset(val_data)


# --------------------------
# Model Initialization
# --------------------------
print("Initializing model...")

# Create model and move to device
# ---------------------------------------------------------
# WHY .to(Config.device):
# - Ensures model runs on GPU (if available)
#
# TRADE-OFF:
# GPU → fast but limited memory
# CPU → slow but more memory
model = MiniGPT().to(Config.device)


# Optimizer
# ---------------------------------------------------------
# AdamW = Adam + weight decay
#
# WHY AdamW:
# - Better generalization than Adam
# - Standard for transformer models
#
# learning_rate:
# - Controlled via Config
optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)


# Print total parameters
# ---------------------------------------------------------
# WHY:
# - Helps understand model scale
# - Useful for debugging and optimization
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# --------------------------
# Training Loop
# --------------------------
print("Starting training...")

# Track best validation loss
# ---------------------------------------------------------
# WHY:
# - Used for saving best model
#
# float('inf'):
# - Ensures any real loss is smaller initially
best_val_loss = float('inf')


# Main training iterations
# ---------------------------------------------------------
# WHY iterations (not epochs):
# - Dataset is sampled randomly → no strict epochs
for iter in range(Config.max_iters):

    # Step 1: Get batch
    # ---------------------------------------------------------
    # xb: input tokens
    # yb: target tokens (shifted)
    xb, yb = train_dataset.get_batch()


    # Step 2: Forward pass
    # ---------------------------------------------------------
    # logits: predictions
    # loss: training loss
    logits, loss = model(xb, yb)


    # Step 3: Backpropagation
    # ---------------------------------------------------------
    # zero_grad:
    # - Clears previous gradients
    #
    # set_to_none=True:
    # - More memory efficient than zeroing
    optimizer.zero_grad(set_to_none=True)

    # Compute gradients
    loss.backward()

    # Update weights
    optimizer.step()


    # --------------------------
    # Evaluation
    # --------------------------
    if iter % Config.eval_interval == 0:

        # Switch to evaluation mode
        # -----------------------------------------------------
        # WHY:
        # - Disables dropout
        # - Uses running statistics (if applicable)
        model.eval()

        # Disable gradient computation
        # -----------------------------------------------------
        # WHY:
        # - Saves memory
        # - Faster inference
        with torch.no_grad():
            xval, yval = val_dataset.get_batch()
            _, val_loss = model(xval, yval)

        # Switch back to training mode
        model.train()

        # Print progress
        print(f"step {iter:4d}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")


        # --------------------------
        # Save Best Model
        # --------------------------
        # WHY:
        # - Prevent losing best-performing weights
        #
        # CONDITION:
        # - Save only if validation improves
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()

            # Save model weights
            # -------------------------------------------------
            # state_dict:
            # - Contains all learnable parameters
            #
            # WHY not save full model:
            # - More flexible (can reload into modified architecture)
            torch.save(model.state_dict(), 'minigpt_best.pt')

            print("  ✅ New best model saved!")


# --------------------------
# Final Save
# --------------------------

# Save final model regardless of performance
# ---------------------------------------------------------
# WHY:
# - Useful for debugging or further training
torch.save(model.state_dict(), 'minigpt_final.pt')


print("Training complete!")
print("Best model: minigpt_best.pt")
print("Final model: minigpt_final.pt")