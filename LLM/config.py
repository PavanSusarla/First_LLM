import torch
class Config:
    """
    Central configuration class for model architecture, training setup,
    and system-level parameters.

    WHY THIS CLASS EXISTS:
    ---------------------------------------------------------
    - Avoids hardcoding values across multiple files
    - Makes experimentation easier (change values in one place)
    - Improves readability and maintainability

    DESIGN CHOICE:
    ---------------------------------------------------------
    Using a class instead of a dictionary:
    - Enables dot notation (Config.batch_size)
    - Cleaner and more structured access
    - Easier to extend with methods if needed
    """


    # Vocabulary size
    # ---------------------------------------------------------
    # Total number of unique tokens the model can handle.
    #
    # WHY 5000:
    # - A moderate size for small-scale experiments
    # - Balances expressiveness vs memory usage
    #
    # WHAT IF INCREASED:
    # - More expressive model (captures more words/subwords)
    # - Higher memory usage (embedding matrix grows)
    #
    # WHAT IF DECREASED:
    # - Faster training
    # - But more "unknown" or poorly represented tokens
    #
    # IMPACT:
    # Embedding layer size = vocab_size × n_embd
    vocab_size = 5000


    # Context window size (sequence length)
    # ---------------------------------------------------------
    # Number of tokens the model looks at in one forward pass.
    #
    # WHY 128:
    # - Standard small context size for experimentation
    # - Fits easily in GPU memory
    #
    # WHAT IF INCREASED (e.g., 512, 1024):
    # - Model captures longer dependencies
    # - BUT attention complexity becomes O(n^2)
    #   → significantly slower and more memory-intensive
    #
    # WHAT IF DECREASED:
    # - Faster training
    # - But poor long-range understanding
    #
    # TRADE-OFF:
    # Context length vs computational cost
    block_size = 128


    # Embedding dimension
    # ---------------------------------------------------------
    # Size of vector representation for each token.
    #
    # WHY 128:
    # - Lightweight model for learning/debugging
    # - Reduces training time and memory usage
    #
    # WHAT IF INCREASED:
    # - Better representation capacity
    # - More parameters → risk of overfitting
    #
    # WHAT IF DECREASED:
    # - Faster but weaker model
    #
    # RELATIONSHIP:
    # Must be divisible by n_head (for multi-head attention)
    n_embd = 128


    # Number of attention heads
    # ---------------------------------------------------------
    # Multi-head attention splits embedding into multiple parts.
    #
    # WHY 4:
    # - Allows model to learn multiple relationships in parallel
    # - Keeps computation manageable
    #
    # CONSTRAINT:
    # n_embd % n_head == 0
    # (Each head gets equal dimension)
    #
    # WHAT IF INCREASED:
    # - Better modeling of complex relationships
    # - More computation and memory
    #
    # WHAT IF DECREASED:
    # - Simpler but less expressive attention
    n_head = 4


    # Number of transformer layers
    # ---------------------------------------------------------
    # Depth of the model (number of stacked blocks).
    #
    # WHY 4:
    # - Enough depth to learn meaningful patterns
    # - Still lightweight for experimentation
    #
    # WHAT IF INCREASED:
    # - Better learning capacity
    # - Risk of overfitting + slower training
    #
    # WHAT IF DECREASED:
    # - Faster but shallow understanding
    #
    # TRADE-OFF:
    # Depth vs training time & overfitting
    n_layer = 4


    # Dropout rate
    # ---------------------------------------------------------
    # Regularization technique to prevent overfitting.
    #
    # WHY 0.1:
    # - Standard default in many transformer models
    # - Drops 10% of neurons randomly during training
    #
    # WHAT IF INCREASED (e.g., 0.3, 0.5):
    # - Stronger regularization
    # - Risk of underfitting
    #
    # WHAT IF DECREASED (or 0):
    # - Faster convergence
    # - Risk of overfitting
    #
    # APPLIES TO:
    # - Attention weights
    # - Feed-forward layers
    dropout = 0.1


    # Batch size
    # ---------------------------------------------------------
    # Number of samples processed in one training step.
    #
    # WHY 32:
    # - Good balance between stability and memory usage
    #
    # WHAT IF INCREASED:
    # - More stable gradients
    # - Requires more GPU memory
    #
    # WHAT IF DECREASED:
    # - Noisy gradients (can help generalization)
    # - Faster iterations but less stable training
    #
    # TRADE-OFF:
    # Stability vs memory usage
    batch_size = 32


    # Learning rate
    # ---------------------------------------------------------
    # Step size for updating model weights.
    #
    # WHY 3e-4 (0.0003):
    # - Common default for Adam optimizer in transformers
    #
    # WHAT IF TOO HIGH:
    # - Training becomes unstable (loss explodes)
    #
    # WHAT IF TOO LOW:
    # - Very slow convergence
    #
    # BEST PRACTICE:
    # - Often combined with learning rate schedulers
    learning_rate = 3e-4


    # Maximum training iterations
    # ---------------------------------------------------------
    # Total number of training steps.
    #
    # WHY 2000:
    # - Suitable for small experiments or debugging
    #
    # WHAT IF INCREASED:
    # - Better convergence (if not overfitting)
    #
    # WHAT IF DECREASED:
    # - Model may underfit (not fully trained)
    #
    # NOTE:
    # Iterations ≠ epochs (depends on dataset size)
    max_iters = 2000


    # Evaluation interval
    # ---------------------------------------------------------
    # Frequency of evaluating model performance.
    #
    # WHY 200:
    # - Periodic monitoring without slowing training too much
    #
    # WHAT IF TOO FREQUENT:
    # - Slows down training
    #
    # WHAT IF TOO RARE:
    # - Hard to track model progress
    #
    # PURPOSE:
    # - Logging loss
    # - Checking overfitting
    eval_interval = 200


    # Device configuration
    # ---------------------------------------------------------
    # Specifies where computations will run.
    #
    # OPTIONS:
    # - 'cuda' → GPU (faster, requires CUDA support)
    # - 'cpu'  → CPU (slower, more compatible)
    #
    # WHY 'cuda':
    # - Significant speedup for deep learning workloads
    #
    # WHAT IF GPU NOT AVAILABLE:
    # - Must switch to 'cpu' to avoid runtime errors
    #
    # BEST PRACTICE:
    # Use dynamic detection:
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # TRADE-OFF:
    # GPU → fast but limited memory
    # CPU → slow but widely available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("all good ")