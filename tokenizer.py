from __future__ import annotations
from typing import List, Union

import tiktoken
import numpy as np



class TiktokenWrapper:
    """
    A wrapper around tiktoken that provides simplified access to encoding/decoding
    with optional conversion to numpy arrays.
    """

    def __init__(self, encoding_name: str | None = "cl100k_base"):
        """
        Initialize the wrapper with the specified encoding.

        Args:
            encoding_name (str): The name of the tiktoken encoding to use.
                           Default is "cl100k_base" (used by many Claude and GPT models).
                           Other options include "p50k_base", "r50k_base", etc.
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name
        except KeyError:
            available_encodings = tiktoken.list_encoding_names()
            raise ValueError(
                f"Encoding '{encoding_name}' not found. Available encodings: {available_encodings}"
            )

    def __repr__(self) -> str:
        return f"TiktokenWrapper(encoding_name='{self.encoding_name}')"

    def encode(
        self,
        text: str,
        as_numpy: bool  | None = False,
        allowed_special: Union[set, str] | None = "all",
    ) -> Union[List[int], np.ndarray]:
        """
        Encode text into tokens.

        Args:
            text (str): The text to encode
            as_numpy (bool): If True, return a numpy array instead of a list
            allowed_special (Union[set, str]): Set of special tokens to allow in the encoding.
                             Use "all" to allow all special tokens (default),
                             use set() to disallow all special tokens, or
                             provide a set of token strings to allow specific ones.

        Returns:
            A list of token IDs or numpy array (if as_numpy=True)
        """
        # Tiktoken can't handle None for allowed_special
        if allowed_special is None:
            allowed_special = set()

        tokens = self.encoding.encode(text, allowed_special=allowed_special)
        if as_numpy:
            return np.array(tokens)
        return tokens

    def decode(self, tokens: Union[List[int], np.ndarray]) -> str:
        """
        Decode tokens back into text.

        Args:
            tokens (Union[List[int], np.ndarray]): List or numpy array of token IDs

        Returns:
            str: The decoded text
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text without storing the full token list.

        Args:
            text (str): The text to count tokens for

        Returns:
            int: The number of tokens
        """
        return len(self.encoding.encode(text, allowed_special="all"))

    def get_token_bytes(self) -> List[bytes]:
        """
        Get the byte values for each token in the encoding.

        Returns:
            List[bytes]: List of bytes for each token
        """
        return self.encoding.decode_single_token_bytes


# Example usage
if __name__ == "__main__":
    # Create a tokenizer with default encoding (cl100k_base)
    tokenizer = TiktokenWrapper()

    # Encode text
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")

    # Encode as numpy array
    tokens_np = tokenizer.encode(text, as_numpy=True)
    print(f"Numpy tokens: {tokens_np}")
    print(f"Type: {type(tokens_np)}")

    # Decode back to text
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")

    # Count tokens
    count = tokenizer.count_tokens("This is a test of the token counter.")
    print(f"Token count: {count}")

    # Create with different encoding
    try:
        tokenizer_p50k = TiktokenWrapper("p50k_base")
        print(f"Created tokenizer with p50k_base encoding")
    except ValueError as e:
        print(f"Error: {e}")
