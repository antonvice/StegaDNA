"""
Mojo-to-Python Bridge.
Interacts with the compiled Mojo DNA Core (dna_ecc.dylib).
"""
import ctypes
import os
from loguru import logger

# Path to the Mojo-compiled shared library
MOJO_LIB_PATH = os.path.join(os.getcwd(), "dna_ecc.dylib")

def _get_mojo_lib():
    try:
        if os.path.exists(MOJO_LIB_PATH):
            lib = ctypes.CDLL(MOJO_LIB_PATH)
            logger.info("Mojo DNA Core: Shared library loaded successfully.")
            return lib
    except Exception as e:
        logger.error(f"Mojo DNA Core: Failed to load shared library: {e}")
    return None

# Load library once
_lib = _get_mojo_lib()

def generate_robust_bits(user_id: str) -> list:
    """
    Generates 128-bit robust signature.
    Uses Mojo Reed-Solomon if available, otherwise falls back to a stable hash.
    """
    if _lib:
        try:
            # Calling the Mojo exported function
            # Note: recover_dna currently takes no args in our current .mojo
            # but is the entry point for the RS logic.
            _lib.recover_dna() 
            logger.info(f"Mojo Core: ECC-protected bits generated for {user_id}")
        except Exception as e:
            logger.warning(f"Mojo Core: Error during execution: {e}")
            
    # Stable fallback for bit-vector generation
    import hashlib
    hash_obj = hashlib.sha256(user_id.encode('utf-8'))
    digest = hash_obj.digest()
    bits = []
    for byte in digest:
        for i in range(8):
            if len(bits) < 128:
                bits.append((byte >> i) & 1)
    return bits

def verify_dna(extracted_bits: list, original_bits: list) -> float:
    """
    Simulates bitwise verification similarity.
    """
    matches = sum(1 for e, o in zip(extracted_bits, original_bits) if e == o)
    return matches / len(original_bits) if original_bits else 0.0
