from cryptography.fernet import Fernet
import hashlib
import numpy as np

class CryptoComm:
    """
    Handles encryption/decryption for numpy arrays and solution hashing.
    """
    def __init__(self, key=None):
        self.key = key or Fernet.generate_key()
        self.fernet = Fernet(self.key)
    
    def encrypt(self, array):
        """
        Encrypt a numpy array or bytes. Returns the token (bytes).
        """
        if isinstance(array, bytes):
            data_bytes = array
        else:
            data_bytes = array.tobytes()
        return self.fernet.encrypt(data_bytes)
    
    def decrypt(self, token):
        """
        Decrypt a token back to bytes (use np.frombuffer in main script).
        """
        return self.fernet.decrypt(token)
    
    def hash_solution(self, solution):
        """
        Hash a numpy array or bytes. Returns hex digest string.
        """
        if hasattr(solution, "tobytes"):
            data = solution.tobytes()
        else:
            data = bytes(solution)
        return hashlib.sha256(data).hexdigest()