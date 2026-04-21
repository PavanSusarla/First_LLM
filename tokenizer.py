# Importing 're' (regular expressions module)
# ---------------------------------------------------------
# WHY:
# Although not used in the current implementation, 're' is typically
# imported in tokenizer-related scripts for advanced text preprocessing
# such as removing punctuation, splitting words, or pattern matching.
#
# TRADE-OFF:
# Keeping unused imports is not ideal in production code because it
# increases clutter and may confuse readers. It should be removed
# unless future extensions (like regex-based tokenization) are planned.
import re


class SimpleTokenizer:
    """
    A very basic character-level tokenizer.

    WHY THIS CLASS EXISTS:
    ---------------------------------------------------------
    Tokenization is the first step in any NLP/LLM pipeline.
    This class converts raw text into numerical representations
    (tokens) and vice versa.

    DESIGN CHOICE:
    ---------------------------------------------------------
    This is a CHARACTER-LEVEL tokenizer (not word-level or subword).
    
    WHY character-level?
    - Simpler to implement
    - No need for complex vocabulary building (like BPE or WordPiece)
    - Works for any text without unknown tokens

    TRADE-OFF:
    - Pros: Simplicity, no OOV (Out-Of-Vocabulary) issues
    - Cons: Larger sequence lengths, less semantic meaning per token
    """


    def __init__(self, text):
        """
        Constructor that builds the vocabulary and mappings.

        PARAMETER JUSTIFICATION:
        ---------------------------------------------------------
        text : str
            The input corpus used to build the vocabulary.

        WHY this parameter is necessary:
        - Tokenizers need a vocabulary.
        - Instead of predefined vocab, we dynamically extract unique characters.

        WHAT IF OMITTED:
        - Without 'text', we cannot build mappings → tokenizer becomes unusable.
        """

        # Step 1: Extract unique characters
        # ---------------------------------------------------------
        # set(text):
        #   Removes duplicate characters → ensures each character appears once
        #
        # WHY:
        #   Vocabulary should contain only unique tokens.
        #
        # list(...):
        #   Converts set back to list because sets are unordered and not indexable.
        #
        # sorted(...):
        #   Ensures deterministic ordering (IMPORTANT).
        #
        # WHY SORT?
        #   - Guarantees reproducibility
        #   - Same input always produces same token IDs
        #
        # TRADE-OFF:
        #   Sorting adds O(n log n) overhead, but vocab size is usually small.
        self.vocab = sorted(list(set(text)))

        # Step 2: Create string-to-index mapping (stoi)
        # ---------------------------------------------------------
        # Dictionary comprehension:
        #   Maps each character → unique integer ID
        #
        # WHY enumerate():
        #   - Provides index automatically
        #   - Efficient and clean
        #
        # WHY index starts at 0:
        #   - Standard practice in Python
        #   - Compatible with arrays, embeddings, and ML frameworks
        #
        # WHAT IF STARTED AT 1:
        #   - Wastes index 0 or requires padding logic adjustments
        #
        # TIME COMPLEXITY:
        #   O(V) where V = vocab size
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

        # Step 3: Create reverse mapping (itos)
        # ---------------------------------------------------------
        # WHY needed:
        #   Encoding maps text → numbers
        #   Decoding requires reverse mapping (numbers → text)
        #
        # DESIGN:
        #   Invert 'stoi' dictionary
        #
        # TRADE-OFF:
        #   Uses extra memory O(V), but necessary for decoding
        self.itos = {i: ch for ch, i in self.stoi.items()}


    def encode(self, s):
        """
        Converts a string into a list of token IDs.

        PARAMETER JUSTIFICATION:
        ---------------------------------------------------------
        s : str
            Input string to encode.

        WHY necessary:
        - This is the core functionality: converting human-readable
          text into machine-readable numeric format.

        WHAT IF OMITTED:
        - Tokenizer would only define vocab but not be usable.
        """

        # List comprehension for efficient transformation
        # ---------------------------------------------------------
        # For each character 'c' in string 's':
        #   Look up its corresponding integer ID in 'stoi'
        #
        # WHY direct lookup:
        #   - Dictionary lookup is O(1) average
        #   - Very efficient for encoding
        #
        # EDGE CASE:
        #   If 'c' not in vocab → KeyError
        #
        # WHY this happens:
        #   Vocabulary is built only from initial text
        #
        # TRADE-OFF:
        #   - Simplicity vs robustness
        #   - Production tokenizers handle unknown tokens (<UNK>)
        #
        # TIME COMPLEXITY:
        #   O(n) where n = length of input string
        return [self.stoi[c] for c in s]


    def decode(self, tokens):
        """
        Converts a list of token IDs back into a string.

        PARAMETER JUSTIFICATION:
        ---------------------------------------------------------
        tokens : list[int]
            List of integer token IDs.

        WHY necessary:
        - Reverse operation of encoding
        - Required for generating human-readable outputs
        """

        # Step 1: Convert each token ID → character
        # ---------------------------------------------------------
        # Using list comprehension for efficiency
        #
        # WHY:
        #   - Faster than manual loops in Python
        #   - Cleaner syntax
        #
        # EDGE CASE:
        #   If token not in 'itos' → KeyError
        #
        # TIME COMPLEXITY:
        #   O(n)
        chars = [self.itos[t] for t in tokens]

        # Step 2: Join characters into a single string
        # ---------------------------------------------------------
        # WHY ''.join():
        #   - More efficient than concatenation in loops
        #   - Avoids O(n^2) complexity
        #
        # FINAL OUTPUT:
        #   Reconstructed string
        return ''.join(chars)