"""LogPrime: a word co-occurrence map, exactly decodable sentence codes, and word vectors."""

from .codec import Codec, Decoded
from .core import LogPrime
from .semantic import Semantic

__version__ = "0.4.0"
__all__ = ["LogPrime", "Codec", "Decoded", "Semantic"]
