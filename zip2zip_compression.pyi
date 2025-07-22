from typing import Dict, List, Optional, Tuple, Union

class CompressionConfig:
    """
    Class for storing the configuration of the codebook.
    """

    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the CompressionConfig.

        Args:
            initial_vocab_size: The initial size of the vocabulary.
            max_codebook_size: The maximum size of the codebook.
            max_subtokens: The maximum number of subtokens per entry.
            pad_token_id: The token id to use for padding.
            disabled_ids: A list of token ids to disable.
        """
        ...

class Codebook:
    """
    Class for storing the LZW codebook.
    """

    def to_list(self, use_padding: bool) -> List[List[int]]:
        """
        Getter for the codebook as a list of lists.

        Args:
            use_padding: Whether to use padding. If True, the codebook will be padded with the pad_token_id
            up to the maximum length of the codebook. Each entry in the codebook will also be padded with
            the pad_token_id up to the maximum number of subtokens per entry.

        Returns:
            The codebook as a list of lists.
        """
        ...

    def to_dict(self) -> Dict[int, List[int]]:
        """
        Get the codebook as a dictionary.
        """
        ...

    def get_subtokens(self, id: int) -> Optional[List[int]]:
        """
        Get the subtokens for a given token id.

        Args:
            id: The token id.

        Returns:
            The subtokens for the given token id.
        """
        ...

class CompressionState:
    """
    Class for storing the state of the compression.
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the CompressionState.

        Args:
            config: The configuration for the compression.
        """
        ...

class LZWCompressor:
    """
    Class for compressing and decompressing sequences of tokens.
    """

    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the LZWCompressor.

        Args:
            initial_vocab_size: The initial size of the vocabulary.
            max_codebook_size: The maximum size of the codebook.
            max_subtokens: The maximum number of subtokens per entry.
            pad_token_id: The token id to use for padding.
            disabled_ids: A list of token ids to disable.
        """
        ...
    def encode(
        self,
        ids: List[int],
        padding: Union[str, bool] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int], Codebook]:
        """
        Encode (or compress) a sequence of tokens.

        Args:
            ids: The sequence of tokens to encode.
            padding: The padding strategy to use. Can be a string (one of "longest", "max_length", "do_not_pad")
            or a boolean (True or False). Default is False.
            truncation: Whether to truncate the sequence if it is too long. Default is False.
            max_length: The maximum length of the sequence. Default is None.

        Returns:
            A tuple containing the encoded sequence, the attention mask, and the codebook.
        """
        ...
    def decode(self, compressed_ids: List[int]) -> Tuple[List[int], Codebook]:
        """
        Decode (or decompress) a sequence of compressed tokens.

        Args:
            compressed_ids: The sequence of compressed tokens to decode.

        Returns:
            A tuple containing the decoded sequence of tokens and the codebook.
        """
        ...
    def batch_encode(
        self,
        ids: List[List[int]],
        padding: Union[str, bool],
        truncation: bool,
        max_length: Optional[int],
    ) -> Tuple[List[List[int]], List[List[int]], List[Codebook]]:
        """
        Encode (or compress) a batch of sequences of tokens.

        Args:
            ids: The batch of sequences of tokens to encode.
            padding: The padding strategy to use. Can be a string (one of "longest", "max_length", "do_not_pad")
            or a boolean (True or False). Default is False.
            truncation: Whether to truncate the sequences if they are too long. Default is False.
            max_length: The maximum length of the sequences. Default is None.

        Returns:
            A tuple containing the encoded sequences, the attention masks, and the codebooks.
        """
        ...
    def batch_decode(
        self,
        compressed_ids: List[List[int]],
    ) -> List[Tuple[List[int], Codebook]]:
        """
        Decode (or decompress) a batch of sequences of compressed tokens.

        Args:
            compressed_ids: The batch of sequences of compressed tokens to decode.

        Returns:
            A list of tuples, each containing the decoded sequence of tokens and the corresponding codebook.
        """
        ...

    def continuous_batch_encode(
        self,
        ids: List[List[int]],
        max_length: int,
        min_length: Optional[int] = 0,
        use_padding: Optional[bool] = True,
    ) -> Tuple[List[List[int]], List[List[List[int]]]]:
        """
        Encode a batch of sequences of tokens in a continuous manner. This method will try to consume
        the sequences as much as possible, and return the compressed ids and the codebooks.

        Args:
            ids: The batch of sequences of tokens to encode.
            max_length: The maximum length of the sequences.
            min_length: The minimum length of the sequences. This can be used to limit the padding
            and discard the sequences that are too short. Default is 0.
            use_padding: If the compressed ids are padded to the `max_length` and the codebooks are
            padded to the maximum size (max subtokens, and max entries).

        Returns:
            A tuple containing the compressed ids and the codebooks.
        """
        ...

class CodebookManager:
    """
    Class for managing the codebooks. During generation, the codebook manager will update the codebooks
    with the new tokens.
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the CodebookManager.

        Args:
            config: The configuration for the codebook.
        """
        ...
    def update_codebooks(
        self, ids: List[List[int]], states: List[CompressionState], use_padding: bool
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Update the codebooks with the new tokens.

        Args:
            ids: The new tokens to update the codebooks with.
            states: The states of the compression.
            use_padding: Whether to use padding. If True, the codebook will be padded with the pad_token_id
            up to the maximum length of the codebook. Each entry in the codebook will also be padded with
            the pad_token_id up to the maximum number of subtokens per entry. If False, the codebook will
            not be padded.

        Returns:
            A tuple containing the new entries in the codebooks and the indices of the new entries.
        """
        ...
