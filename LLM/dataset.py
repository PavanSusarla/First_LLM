# Importing PyTorch
# ---------------------------------------------------------
# WHY:
# PyTorch is used here for tensor operations and efficient batch handling.
# Specifically:
# - torch.randint → for random sampling
# - torch.stack → for batch creation
#
# DEPENDENCY JUSTIFICATION:
# PyTorch is preferred in deep learning pipelines because:
# - Native GPU support (CUDA)
# - Efficient tensor computations
# - Seamless integration with training loops
#
# TRADE-OFF:
# Compared to NumPy, PyTorch introduces slightly more overhead
# but is essential for GPU acceleration and model training.
import torch


# Importing configuration values
# ---------------------------------------------------------
# WHY:
# Centralizing hyperparameters (like batch_size, block_size, device)
# improves maintainability and avoids hardcoding values.
#
# WHAT IF OMITTED:
# - Values would be scattered across code → harder to modify
# - Less flexible for experimentation
#
# DESIGN CHOICE:
# Using a Config class instead of constants allows structured access.
from config import Config


class TextDataset:
    """
    A dataset class responsible for generating training batches
    for sequence models (e.g., LLMs, RNNs, Transformers).

    CORE IDEA:
    ---------------------------------------------------------
    Given a long sequence of tokenized data, this class:
    - Randomly samples chunks (subsequences)
    - Creates input-output pairs for next-token prediction

    WHY THIS APPROACH:
    ---------------------------------------------------------
    Language models are trained using:
        "predict next token given previous tokens"

    So:
        Input (x):  [t1, t2, t3, ..., tN]
        Target (y): [t2, t3, t4, ..., tN+1]

    This is called "causal language modeling".

    TRADE-OFF:
    ---------------------------------------------------------
    + Efficient: avoids storing all possible sequences
    + Random sampling improves generalization
    - Loses sequential continuity across batches
    """


    def __init__(self, data):
        """
        Initializes the dataset.

        PARAMETER JUSTIFICATION:
        ---------------------------------------------------------
        data : torch.Tensor or list
            A long sequence of tokenized text.

        WHY necessary:
        - This is the raw dataset from which training samples are drawn.

        WHAT IF OMITTED:
        - No source data → batch generation impossible

        DESIGN CHOICE:
        ---------------------------------------------------------
        Data is kept as a single continuous sequence instead of
        pre-splitting into chunks.

        WHY:
        - Memory efficient
        - Flexible random sampling

        TRADE-OFF:
        - Slight runtime cost when slicing repeatedly
        """
        self.data = data


    def get_batch(self):
        """
        Generates a batch of input-target pairs for training.

        RETURNS:
        ---------------------------------------------------------
        x : Tensor of shape (batch_size, block_size)
            Input sequences

        y : Tensor of shape (batch_size, block_size)
            Target sequences (shifted by +1)

        WHY THIS FUNCTION EXISTS:
        ---------------------------------------------------------
        - Training requires batches, not single samples
        - Random sampling prevents overfitting to sequence order
        """

        # Step 1: Generate random starting indices
        # ---------------------------------------------------------
        # torch.randint(high, size):
        #   Generates random integers in range [0, high)
        #
        # WHY (len(self.data) - Config.block_size):
        #   Ensures we don't go out of bounds when slicing
        #
        #   Example:
        #   If block_size = 8 and data length = 100
        #   Max valid start index = 92
        #
        # WHAT IF NOT SUBTRACTED:
        #   Index out-of-range error during slicing
        #
        # WHY shape = (Config.batch_size,):
        #   We want one starting index per batch sample
        #
        # TIME COMPLEXITY:
        #   O(batch_size)
        ix = torch.randint(len(self.data) - Config.block_size, (Config.batch_size,))


        # Step 2: Create input batch (x)
        # ---------------------------------------------------------
        # For each index i:
        #   Take a chunk of length block_size
        #
        # Example:
        #   data = [1,2,3,4,5,6]
        #   block_size = 3
        #   i = 1 → [2,3,4]
        #
        # WHY torch.stack():
        #   - Converts list of tensors → single tensor
        #   - Required for batch processing
        #
        # OUTPUT SHAPE:
        #   (batch_size, block_size)
        #
        # TIME COMPLEXITY:
        #   O(batch_size * block_size)
        x = torch.stack([self.data[i:i + Config.block_size] for i in ix])


        # Step 3: Create target batch (y)
        # ---------------------------------------------------------
        # This is the "shifted" version of x
        #
        # For each index i:
        #   Target = next tokens
        #
        # Example:
        #   x = [2,3,4]
        #   y = [3,4,5]
        #
        # WHY +1 shift:
        #   This aligns with next-token prediction objective
        #
        # WHAT IF NOT SHIFTED:
        #   Model would learn identity function (useless)
        #
        # OUTPUT SHAPE:
        #   Same as x → (batch_size, block_size)
        y = torch.stack([self.data[i + 1:i + Config.block_size + 1] for i in ix])


        # Step 4: Move tensors to device (CPU/GPU)
        # ---------------------------------------------------------
        # WHY:
        #   Model and data must be on same device
        #
        # Config.device could be:
        #   - 'cpu'
        #   - 'cuda'
        #
        # TRADE-OFF:
        #   - GPU → faster but memory limited
        #   - CPU → slower but more memory available
        #
        # IMPORTANT:
        #   Moving inside this function ensures consistency
        #   but may introduce slight overhead per batch
        #
        # OPTIMIZATION:
        #   In large-scale training, preloading to GPU is preferred
        return x.to(Config.device), y.to(Config.device)