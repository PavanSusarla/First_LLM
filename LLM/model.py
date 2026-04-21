# Core PyTorch imports
# ---------------------------------------------------------
# torch → tensor operations
# nn → neural network layers
# F → functional utilities (loss, softmax, etc.)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import configuration and building blocks
# ---------------------------------------------------------
# WHY:
# - Config centralizes hyperparameters
# - TransformerBlock is the core computation unit
from config import Config
from transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """
    A minimal GPT-style Language Model.

    WHY THIS MODEL EXISTS:
    ---------------------------------------------------------
    This is a simplified implementation of a GPT (Generative Pretrained Transformer).

    It learns:
        P(next_token | previous_tokens)

    CORE COMPONENTS:
    ---------------------------------------------------------
    1. Token Embeddings → convert tokens → vectors
    2. Positional Embeddings → encode sequence order
    3. Transformer Blocks → learn contextual relationships
    4. Final LayerNorm → stabilize output
    5. Language Modeling Head → map to vocabulary logits

    TRADE-OFF:
    ---------------------------------------------------------
    + Simple and educational
    + Fully functional GPT pipeline
    - Lacks optimizations (Flash Attention, KV cache, etc.)
    """


    def __init__(self):
        """
        Initializes all model components.
        """

        super().__init__()

        # Token Embedding Table
        # ---------------------------------------------------------
        # Maps token IDs → dense vectors
        #
        # SHAPE:
        # (vocab_size, n_embd)
        #
        # WHY:
        # - Neural networks cannot process raw integers meaningfully
        #
        # WHAT IF SMALLER:
        # - Less expressive representations
        #
        # WHAT IF LARGER:
        # - Better representations but more parameters
        self.token_embedding_table = nn.Embedding(Config.vocab_size, Config.n_embd)


        # Positional Embedding Table
        # ---------------------------------------------------------
        # Encodes position of tokens in sequence
        #
        # SHAPE:
        # (block_size, n_embd)
        #
        # WHY:
        # - Transformer has NO inherent sense of order
        #
        # WHAT IF OMITTED:
        # - Model treats sentence as a bag of words
        #
        # DESIGN:
        # Learnable positional embeddings (instead of sinusoidal)
        self.position_embedding_table = nn.Embedding(Config.block_size, Config.n_embd)


        # Transformer Blocks (stacked)
        # ---------------------------------------------------------
        # Sequential container of TransformerBlock
        #
        # WHY nn.Sequential:
        # - Automatically chains forward pass
        #
        # NUMBER OF LAYERS:
        # Config.n_layer (depth of model)
        #
        # TRADE-OFF:
        # - More layers → better learning but slower
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(Config.n_layer)]
        )


        # Final Layer Normalization
        # ---------------------------------------------------------
        # Applied after all transformer blocks
        #
        # WHY:
        # - Stabilizes final representations
        # - Improves training convergence
        self.ln_f = nn.LayerNorm(Config.n_embd)


        # Language Modeling Head
        # ---------------------------------------------------------
        # Maps embeddings → vocabulary logits
        #
        # SHAPE:
        # (n_embd → vocab_size)
        #
        # WHY:
        # - Converts hidden representation into probabilities over tokens
        #
        # OUTPUT:
        # logits (unnormalized probabilities)
        self.lm_head = nn.Linear(Config.n_embd, Config.vocab_size)


    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        PARAMETERS:
        ---------------------------------------------------------
        idx : Tensor (B, T)
            Input token indices

        targets : Tensor (B, T), optional
            Target tokens for loss computation

        RETURNS:
        ---------------------------------------------------------
        logits : (B, T, vocab_size)
        loss   : scalar (if targets provided)
        """

        # Extract batch size and sequence length
        B, T = idx.shape


        # Step 1: Token embeddings
        # ---------------------------------------------------------
        # Converts token IDs → vectors
        #
        # SHAPE:
        # (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)


        # Step 2: Positional embeddings
        # ---------------------------------------------------------
        # torch.arange(T):
        # - Generates positions [0, 1, ..., T-1]
        #
        # WHY device=Config.device:
        # - Ensures tensor is on same device as model
        #
        # SHAPE:
        # (T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=Config.device)
        )


        # Step 3: Combine embeddings
        # ---------------------------------------------------------
        # Broadcasting:
        # pos_emb (T, C) → expanded to (B, T, C)
        #
        # WHY addition:
        # - Injects positional information into token embeddings
        x = tok_emb + pos_emb


        # Step 4: Pass through Transformer blocks
        # ---------------------------------------------------------
        # Each block:
        # - Applies attention + FFN
        # - Refines representation
        x = self.blocks(x)


        # Step 5: Final normalization
        # ---------------------------------------------------------
        # Stabilizes output before prediction
        x = self.ln_f(x)


        # Step 6: Compute logits
        # ---------------------------------------------------------
        # Output shape:
        # (B, T, vocab_size)
        logits = self.lm_head(x)


        # Step 7: Compute loss (if targets provided)
        # ---------------------------------------------------------
        loss = None

        if targets is not None:
            # Flatten logits
            # -----------------------------------------------------
            # WHY reshape:
            # CrossEntropy expects:
            # (N, C) where N = total tokens
            #
            # Original:
            # (B, T, vocab_size)
            #
            # After reshape:
            # (B*T, vocab_size)
            logits = logits.view(-1, logits.size(-1))

            # Flatten targets
            # -----------------------------------------------------
            # (B, T) → (B*T)
            targets = targets.view(-1)

            # Cross-entropy loss
            # -----------------------------------------------------
            # Combines:
            # - Softmax
            # - Negative log likelihood
            #
            # WHY:
            # Standard loss for classification problems
            loss = F.cross_entropy(logits, targets)


        return logits, loss


    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens autoregressively.

        PARAMETERS:
        ---------------------------------------------------------
        idx : Tensor (B, T)
            Initial context

        max_new_tokens : int
            Number of tokens to generate

        RETURNS:
        ---------------------------------------------------------
        Tensor (B, T + max_new_tokens)
        """

        # Loop for generating tokens step-by-step
        for _ in range(max_new_tokens):

            # Step 1: Crop context to block size
            # -----------------------------------------------------
            # WHY:
            # Model can only handle sequences up to block_size
            #
            # Keeps most recent tokens
            idx_cond = idx[:, -Config.block_size:]


            # Step 2: Forward pass
            logits, _ = self(idx_cond)


            # Step 3: Focus on last token
            # -----------------------------------------------------
            # WHY:
            # We only care about predicting next token
            #
            # SHAPE:
            # (B, vocab_size)
            logits = logits[:, -1, :]


            # Step 4: Convert to probabilities
            probs = F.softmax(logits, dim=-1)


            # Step 5: Sample next token
            # -----------------------------------------------------
            # torch.multinomial:
            # - Samples based on probability distribution
            #
            # WHY sampling (not argmax):
            # - Produces diverse outputs
            #
            # TRADE-OFF:
            # - Sampling → creative but less deterministic
            # - Argmax → deterministic but repetitive
            next_token = torch.multinomial(probs, num_samples=1)


            # Step 6: Append new token
            # -----------------------------------------------------
            # Concatenate along sequence dimension
            idx = torch.cat((idx, next_token), dim=1)


        return idx