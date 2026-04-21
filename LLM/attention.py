# Core PyTorch imports
# ---------------------------------------------------------
# torch → tensor operations
# nn → neural network building blocks
# F → functional APIs (stateless operations like softmax)
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# NOTE:
# Config is assumed to be imported externally.
# It contains hyperparameters like embedding size, block size, dropout, etc.
# from config import Config


class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism.

    WHY THIS MODULE EXISTS:
    ---------------------------------------------------------
    Self-attention allows each token in a sequence to:
    - "Look at" (attend to) other tokens
    - Learn contextual relationships dynamically

    Example:
    Sentence: "The cat sat on the mat"
    → "sat" can attend to "cat" to understand subject

    DESIGN CHOICE:
    ---------------------------------------------------------
    This is a SINGLE attention head (not multi-head).

    WHY single-head?
    - Simpler to understand and debug
    - Used as a building block for multi-head attention

    TRADE-OFF:
    - Less expressive than multi-head attention
    - Cannot capture multiple relationship types simultaneously
    """


    def __init__(self, head_size):
        """
        Initializes projection layers and attention mask.

        PARAMETER JUSTIFICATION:
        ---------------------------------------------------------
        head_size : int
            Dimensionality of each attention head.

        WHY necessary:
        - Defines output dimension of attention projections
        - In multi-head attention, total embedding is split across heads

        CONSTRAINT:
        - Typically: Config.n_embd % num_heads == 0
        """

        # Initialize parent class
        super().__init__()

        # Key projection
        # ---------------------------------------------------------
        # Maps input embeddings → key vectors
        #
        # WHY Linear layer:
        # - Learns transformation from embedding space to key space
        #
        # WHY bias=False:
        # - Reduces parameters slightly
        # - Common practice in attention layers (bias not critical here)
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)

        # Query projection
        # ---------------------------------------------------------
        # Maps input embeddings → query vectors
        #
        # ROLE:
        # - Queries "ask questions"
        # - Keys "answer questions"
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)

        # Value projection
        # ---------------------------------------------------------
        # Maps input embeddings → value vectors
        #
        # ROLE:
        # - Values contain actual information to aggregate
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)


        # Causal mask (lower triangular matrix)
        # ---------------------------------------------------------
        # torch.tril → lower triangular matrix
        #
        # WHY:
        # Prevents tokens from attending to future tokens
        #
        # Example:
        # Token at position i can only attend to positions ≤ i
        #
        # This enforces "causality" → required for autoregressive models (like GPT)
        #
        # register_buffer:
        # - Stores tensor as part of model (moves with device)
        # - NOT a trainable parameter
        #
        # WHY not parameter?
        # - Mask is fixed, not learned
        #
        # SHAPE:
        # (block_size, block_size)
        #
        # TRADE-OFF:
        # - Precomputing saves time during forward pass
        # - Uses extra memory O(n^2)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(Config.block_size, Config.block_size))
        )

        # Dropout layer
        # ---------------------------------------------------------
        # Applied to attention weights (not values)
        #
        # WHY:
        # - Prevents over-reliance on specific tokens
        # - Improves generalization
        #
        # VALUE:
        # Config.dropout (typically 0.1)
        self.dropout = nn.Dropout(Config.dropout)


    def forward(self, x):
        """
        Forward pass of self-attention.

        PARAMETER:
        ---------------------------------------------------------
        x : Tensor of shape (B, T, C)
            B = batch size
            T = sequence length (time steps)
            C = embedding dimension

        RETURNS:
        ---------------------------------------------------------
        out : Tensor of shape (B, T, head_size)
        """

        # Extract dimensions
        # ---------------------------------------------------------
        # WHY unpack:
        # - Improves readability
        # - Avoids repeated shape indexing
        B, T, C = x.shape


        # Step 1: Compute Key and Query matrices
        # ---------------------------------------------------------
        # Shapes:
        # k → (B, T, head_size)
        # q → (B, T, head_size)
        #
        # WHY:
        # - These projections transform embeddings into attention space
        k = self.key(x)
        q = self.query(x)


        # Step 2: Compute attention scores (scaled dot-product)
        # ---------------------------------------------------------
        # Formula:
        # attention = Q × K^T
        #
        # Shapes:
        # q → (B, T, head_size)
        # k^T → (B, head_size, T)
        #
        # Result:
        # wei → (B, T, T)
        #
        # INTERPRETATION:
        # wei[i, j] = how much token i attends to token j
        #
        # SCALING FACTOR:
        # (k.shape[-1] ** -0.5) = 1 / sqrt(head_size)
        #
        # WHY scaling:
        # - Prevents large dot-product values
        # - Stabilizes gradients during training
        #
        # WITHOUT SCALING:
        # - Softmax becomes too peaky → poor learning
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)


        # Step 3: Apply causal mask
        # ---------------------------------------------------------
        # self.tril[:T, :T] → crop mask to current sequence length
        #
        # masked_fill:
        # - Replace future positions with -inf
        #
        # WHY -inf:
        # - softmax(-inf) → 0
        # - effectively removes attention to future tokens
        #
        # RESULT:
        # Model cannot "cheat" by looking ahead
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))


        # Step 4: Normalize with softmax
        # ---------------------------------------------------------
        # Converts scores into probabilities
        #
        # dim=-1:
        # - Normalize across tokens (last dimension)
        #
        # RESULT:
        # Each row sums to 1 → probability distribution
        wei = F.softmax(wei, dim=-1)


        # Step 5: Apply dropout
        # ---------------------------------------------------------
        # WHY here:
        # - Regularizes attention distribution
        # - Prevents overfitting
        wei = self.dropout(wei)


        # Step 6: Compute Value projections
        # ---------------------------------------------------------
        # Shape:
        # v → (B, T, head_size)
        #
        # WHY:
        # - Values contain actual information to aggregate
        v = self.value(x)


        # Step 7: Weighted aggregation
        # ---------------------------------------------------------
        # Formula:
        # output = attention_weights × values
        #
        # Shapes:
        # wei → (B, T, T)
        # v   → (B, T, head_size)
        #
        # Result:
        # out → (B, T, head_size)
        #
        # INTERPRETATION:
        # Each token becomes a weighted combination of all tokens
        out = wei @ v


        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    WHY THIS MODULE EXISTS:
    ---------------------------------------------------------
    A single attention head can only learn ONE type of relationship
    between tokens (e.g., syntactic or semantic).

    Multi-head attention allows the model to:
    - Learn multiple relationships in parallel
    - Capture richer representations

    Example:
    Sentence: "The bank of the river"
    - Head 1 → learns "bank" relates to "river" (context meaning)
    - Head 2 → learns positional structure
    - Head 3 → focuses on nearby dependencies

    CORE IDEA:
    ---------------------------------------------------------
    Instead of one attention operation:
        Attention(Q, K, V)

    We perform multiple in parallel:
        head_1, head_2, ..., head_n

    Then concatenate their outputs.

    TRADE-OFF:
    ---------------------------------------------------------
    + More expressive model
    + Better context understanding
    - Slightly more computation
    """


    def __init__(self):
        """
        Initializes multiple attention heads and output projection.

        DESIGN CHOICE:
        ---------------------------------------------------------
        No parameters passed explicitly → uses Config class

        WHY:
        - Keeps API clean
        - Centralizes hyperparameters

        TRADE-OFF:
        - Less flexible for dynamic configurations
        """

        super().__init__()

        # Step 1: Compute head size
        # ---------------------------------------------------------
        # head_size = embedding_dim / number_of_heads
        #
        # WHY:
        # - Each head works on a subset of embedding dimensions
        #
        # CONSTRAINT:
        # Config.n_embd % Config.n_head == 0
        #
        # WHAT IF NOT DIVISIBLE:
        # - Dimension mismatch → runtime error
        #
        # EXAMPLE:
        # n_embd = 128, n_head = 4 → head_size = 32
        head_size = Config.n_embd // Config.n_head


        # Step 2: Create multiple attention heads
        # ---------------------------------------------------------
        # nn.ModuleList:
        # - Stores submodules properly so PyTorch tracks parameters
        #
        # WHY not a normal list?
        # - PyTorch won't register parameters → no training
        #
        # Each head:
        # - Independent SelfAttention module
        # - Learns different relationships
        #
        # NUMBER OF HEADS:
        # Config.n_head (e.g., 4)
        #
        # TIME COMPLEXITY:
        # O(n_head × attention_cost)
        self.heads = nn.ModuleList(
            [SelfAttention(head_size) for _ in range(Config.n_head)]
        )


        # Step 3: Output projection layer
        # ---------------------------------------------------------
        # After concatenation, we project back to original embedding size
        #
        # WHY needed:
        # - Concatenation increases dimensionality:
        #   (head_size × n_head) = n_embd
        #
        # - Projection mixes information from all heads
        #
        # DESIGN:
        # Linear(n_embd → n_embd)
        #
        # INTERPRETATION:
        # - Combines features learned by different heads
        #
        # TRADE-OFF:
        # - Adds parameters and computation
        # - Essential for expressiveness
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)


        # Step 4: Dropout layer
        # ---------------------------------------------------------
        # Applied AFTER projection
        #
        # WHY:
        # - Regularizes combined attention output
        # - Prevents overfitting
        #
        # VALUE:
        # Config.dropout (typically 0.1)
        self.dropout = nn.Dropout(Config.dropout)


    def forward(self, x):
        """
        Forward pass of multi-head attention.

        PARAMETER:
        ---------------------------------------------------------
        x : Tensor of shape (B, T, C)
            B = batch size
            T = sequence length
            C = embedding dimension (Config.n_embd)

        RETURNS:
        ---------------------------------------------------------
        Tensor of shape (B, T, C)
        """

        # Step 1: Apply all attention heads in parallel
        # ---------------------------------------------------------
        # Each head processes the SAME input independently
        #
        # WHY:
        # - Each head learns different patterns
        #
        # Output of each head:
        # (B, T, head_size)
        #
        # torch.cat(..., dim=-1):
        # - Concatenate along feature dimension
        #
        # Result shape:
        # (B, T, head_size × n_head) = (B, T, n_embd)
        #
        # WHY concatenate (not sum):
        # - Preserves information from each head
        #
        # TRADE-OFF:
        # - Concatenation increases memory usage
        out = torch.cat([h(x) for h in self.heads], dim=-1)


        # Step 2: Project concatenated output
        # ---------------------------------------------------------
        # WHY:
        # - Mixes information across heads
        # - Restores original embedding structure
        #
        # SHAPE:
        # Input:  (B, T, n_embd)
        # Output: (B, T, n_embd)
        out = self.proj(out)


        # Step 3: Apply dropout
        # ---------------------------------------------------------
        # WHY:
        # - Prevents overfitting
        # - Encourages robustness
        #
        # FINAL OUTPUT:
        # (B, T, n_embd)
        return self.dropout(out)