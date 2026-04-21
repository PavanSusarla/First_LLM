import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from config import Config

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    WHY THIS MODULE EXISTS:
    ---------------------------------------------------------
    In Transformers, attention mixes information ACROSS tokens,
    but it does NOT add non-linearity or transform features deeply.

    The FeedForward network:
    - Processes each token independently (position-wise)
    - Adds non-linearity
    - Expands and compresses feature space

    CORE IDEA:
    ---------------------------------------------------------
    For each token vector x:
        FFN(x) = max(0, xW1 + b1)W2 + b2

    (Linear → Activation → Linear)

    WHY POSITION-WISE:
    ---------------------------------------------------------
    - Same network applied to every token
    - No interaction between tokens here
    - Keeps computation efficient

    TRADE-OFF:
    ---------------------------------------------------------
    + Adds expressive power
    + Introduces non-linearity
    - Adds parameters and compute cost
    """


    def __init__(self):
        """
        Initializes the feed-forward layers.

        DESIGN CHOICE:
        ---------------------------------------------------------
        Using nn.Sequential for simplicity and readability.

        WHY:
        - Clean pipeline of operations
        - No need for custom forward logic

        TRADE-OFF:
        - Less flexible than manual forward definition
        """

        super().__init__()

        # Step 1: Define feed-forward network
        # ---------------------------------------------------------
        self.net = nn.Sequential(

            # First Linear Layer (Expansion)
            # ---------------------------------------------------------
            # Input:  (B, T, n_embd)
            # Output: (B, T, 4 * n_embd)
            #
            # WHY expand by 4x:
            # - Standard practice in Transformers (GPT, BERT)
            # - Allows model to learn richer feature combinations
            #
            # INTUITION:
            # - Think of this as "thinking space"
            # - Model temporarily increases capacity before compressing
            #
            # WHAT IF SMALLER (e.g., 2x):
            # - Less expressive power
            #
            # WHAT IF LARGER (e.g., 8x):
            # - More capacity but higher compute cost
            #
            # TRADE-OFF:
            # Capacity vs computation
            nn.Linear(Config.n_embd, 4 * Config.n_embd),


            # Activation Function
            # ---------------------------------------------------------
            # ReLU(x) = max(0, x)
            #
            # WHY ReLU:
            # - Simple and efficient
            # - Prevents vanishing gradients (compared to sigmoid/tanh)
            #
            # WHAT IF OMITTED:
            # - Entire FFN becomes linear → no additional learning power
            #
            # ALTERNATIVES:
            # - GELU (used in GPT/BERT, smoother)
            # - Swish (better performance but slower)
            #
            # TRADE-OFF:
            # ReLU → faster
            # GELU → better performance in large models
            nn.ReLU(),


            # Second Linear Layer (Compression)
            # ---------------------------------------------------------
            # Input:  (B, T, 4 * n_embd)
            # Output: (B, T, n_embd)
            #
            # WHY:
            # - Projects back to original embedding size
            # - Required for residual connections in Transformer block
            #
            # INTERPRETATION:
            # - Compress learned features into usable representation
            nn.Linear(4 * Config.n_embd, Config.n_embd),


            # Dropout Layer
            # ---------------------------------------------------------
            # WHY:
            # - Prevents overfitting
            # - Randomly zeroes some activations during training
            #
            # VALUE:
            # Config.dropout (typically 0.1)
            #
            # WHAT IF HIGH (e.g., 0.5):
            # - Strong regularization but risk of underfitting
            #
            # WHAT IF ZERO:
            # - Faster learning but risk of overfitting
            nn.Dropout(Config.dropout),
        )


    def forward(self, x):
        """
        Forward pass of FeedForward network.

        PARAMETER:
        ---------------------------------------------------------
        x : Tensor of shape (B, T, C)
            B = batch size
            T = sequence length
            C = embedding dimension (n_embd)

        RETURNS:
        ---------------------------------------------------------
        Tensor of same shape (B, T, C)

        WHY SAME SHAPE:
        ---------------------------------------------------------
        - Required for residual connections:
          output = x + FFN(x)
        """

        # Pass input through sequential network
        # ---------------------------------------------------------
        # Each token is processed independently
        #
        # TIME COMPLEXITY:
        # O(B × T × n_embd × 4*n_embd)
        #
        # NOTE:
        # This is often more computationally expensive than attention
        # in smaller sequence lengths
        return self.net(x)
class TransformerBlock(nn.Module):
    """
    Core Transformer Block (GPT-style, Pre-LayerNorm).

    WHY THIS MODULE EXISTS:
    ---------------------------------------------------------
    This is the fundamental building block of Transformer models
    like GPT, BERT, etc.

    Each block performs:
    1. Multi-Head Self-Attention (context mixing across tokens)
    2. Feed-Forward Network (non-linear transformation per token)
    3. Residual Connections (stability + gradient flow)
    4. Layer Normalization (training stability)

    STACKING:
    ---------------------------------------------------------
    Multiple TransformerBlocks are stacked to build deep models.

    DESIGN CHOICE:
    ---------------------------------------------------------
    This implementation uses **Pre-LayerNorm (Pre-LN)**:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    WHY Pre-LN (instead of Post-LN):
    ---------------------------------------------------------
    + More stable gradients (especially for deep networks)
    + Enables better training convergence
    + Used in modern GPT architectures

    TRADE-OFF:
    - Slightly different optimization dynamics compared to Post-LN
    """


    def __init__(self):
        """
        Initializes submodules for attention, feed-forward,
        and normalization layers.
        """

        super().__init__()

        # Multi-Head Self-Attention module
        # ---------------------------------------------------------
        # Responsible for mixing information across tokens
        #
        # WHY:
        # - Captures dependencies between words/tokens
        #
        # OUTPUT SHAPE:
        # (B, T, n_embd)
        self.sa = MultiHeadAttention()


        # Feed-Forward Network
        # ---------------------------------------------------------
        # Processes each token independently
        #
        # WHY:
        # - Adds non-linearity
        # - Enhances feature representation
        self.ffwd = FeedForward()


        # Layer Normalization (before attention)
        # ---------------------------------------------------------
        # Normalizes input across feature dimension
        #
        # WHY:
        # - Stabilizes training
        # - Prevents exploding/vanishing activations
        #
        # WHY applied BEFORE attention:
        # - Pre-LN improves gradient flow
        #
        # NORMALIZATION:
        # mean = 0, variance = 1 (per token)
        self.ln1 = nn.LayerNorm(Config.n_embd)


        # Layer Normalization (before FFN)
        # ---------------------------------------------------------
        # Separate normalization for second sub-layer
        #
        # WHY separate:
        # - Each sub-layer learns independently
        # - Improves stability and flexibility
        self.ln2 = nn.LayerNorm(Config.n_embd)


    def forward(self, x):
        """
        Forward pass of Transformer block.

        PARAMETER:
        ---------------------------------------------------------
        x : Tensor of shape (B, T, C)
            B = batch size
            T = sequence length
            C = embedding dimension

        RETURNS:
        ---------------------------------------------------------
        Tensor of shape (B, T, C)

        CORE FLOW:
        ---------------------------------------------------------
        1. Normalize input → apply attention → add residual
        2. Normalize result → apply FFN → add residual
        """

        # Step 1: Self-Attention with Residual Connection
        # ---------------------------------------------------------
        # LN(x):
        # - Normalize input before attention
        #
        # self.sa(...):
        # - Apply multi-head attention
        #
        # x + ...:
        # - Residual connection
        #
        # WHY residual:
        # - Helps gradient flow in deep networks
        # - Prevents degradation problem
        #
        # INTERPRETATION:
        # Output = original information + contextual information
        x = x + self.sa(self.ln1(x))


        # Step 2: Feed-Forward with Residual Connection
        # ---------------------------------------------------------
        # LN(x):
        # - Normalize again before FFN
        #
        # self.ffwd(...):
        # - Apply non-linear transformation
        #
        # x + ...:
        # - Residual connection
        #
        # INTERPRETATION:
        # Output = contextual info + refined features
        x = x + self.ffwd(self.ln2(x))


        # Final output
        # ---------------------------------------------------------
        # Shape remains (B, T, C)
        return x