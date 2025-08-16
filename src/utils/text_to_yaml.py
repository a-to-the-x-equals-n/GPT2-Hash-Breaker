import hashlib
import yaml
import secrets
import sys
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from argon2 import PasswordHasher


class Cryptatoes:
    '''
    A wrapper class for password hashing using Argon2, Scrypt, or Bcrypt.

    Parameters:
    -----------
    method : str
        The name of the hashing algorithm to use.

    Raises:
    -------
    ValueError
        If the given method is not supported.
    '''
    
    HASH_DIR = Path(__file__).resolve().parent.parent / 'train-data'

    def __init__(self, method: str, input: str = None) -> None:
        # normalize and store the chosen hashing method
        self.method = method.lower()

        if self.method == 'argon2':
            # Create an Argon2 password hasher instance with configured parameters
            # - time_cost = 1: number of iterations (how many times to repeat the hashing operation)
            # - memory_cost = 1024: memory used in kibibytes (e.g., 1024 = 1MB); higher means more RAM used per hash
            # - parallelism = 1: number of parallel threads used during hashing
            # - hash_len = 32: desired length of the hash output in bytes (32 bytes = 256 bits)
            self.hasher = PasswordHasher(time_cost = 1, memory_cost = 1024, parallelism = 1, hash_len = 32, salt_len = 8)

        elif self.method == 'scrypt':
            # generate random salt for scrypt hashing
            self.scrypt_salt = secrets.token_bytes(16)
        else:
            raise ValueError(f'Unsupported method: {method}')

    def hash(self, password: str) -> str:
        '''
        Hashes a single password using the configured algorithm.

        Parameters:
        -----------
        password : str
            The plaintext password to hash.

        Returns:
        --------
        str
            The resulting hash.
        '''
        
        # hash using argon2
        if self.method == 'argon2':
            return self.hasher.hash(password)
        
        # hash using scrypt, return as hex string
        elif self.method == 'scrypt':
            # hash the password using scrypt — a memory-hard password hashing function
            # - password.encode(): convert the string password to bytes
            # - salt = self.scrypt_salt: a fixed/random 16-byte salt to make hashes unique and resistant to rainbow tables
            # - n = 1024: CPU/memory cost parameter (must be a power of 2); higher = more secure but slower
            # - r = 8: block size parameter, controls memory usage
            # - p = 1: parallelization parameter; how many independent computations can run (higher = more CPU threads)
            # - dklen = 64: desired output length of the hash in bytes (64 bytes = 512 bits)
            hashed = hashlib.scrypt(password.encode(), salt = b'', n = 1024, r = 8, p = 1, dklen = 32)
            return hashed.hex()


def hash_to_yaml(passwords: list[str], hasher: Cryptatoes, output_file: Path):
    output_file.parent.mkdir(parents = True, exist_ok = True)

    with output_file.open('w') as out:
        with Pool(8) as pool:
            for password, hashed in tqdm(zip(passwords, pool.imap(hasher.hash, passwords)), total = len(passwords), desc = f"[{hasher.method.upper()} STREAM-YAML]"):
                yaml.dump(
                    [{'input': hashed, 'output': password}],
                    out,
                    sort_keys = False,
                    explicit_start = False,
                    default_flow_style = False
                )


__all__ = ['Cryptatoes']

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python hashville.py <passwords.txt> <argon2|scrypt>')
        sys.exit(1)

    pw_file = Path(sys.argv[1])
    method = sys.argv[2].lower()
    base = pw_file.stem

    with pw_file.open('r') as f:
        passwords = [line.strip() for line in f]

    hasher = Cryptatoes(method)
    output_yaml = hasher.HASH_DIR / method / f'{method}_{base}.yaml'

    start = time.perf_counter()
    hash_to_yaml(passwords, hasher, output_yaml)
    end = time.perf_counter()

    print(f'\n[✓ YAML]: {output_yaml}')
    print(f'[AVERAGE TIME]: {(end - start)/len(passwords):.6f} sec')
    print(f'[TOTAL TIME]: {end - start:.2f} sec')