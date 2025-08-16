from pathlib import Path
from tqdm import tqdm
import json
import yaml
import torch

torch.cuda.empty_cache()

ROOT = Path.cwd().parent
SOURCE = ROOT / 'data/scrypt/scrypt_pw_10mil.yaml'
SHARD_DIR = ROOT / 'data/scrypt_shards'
SHARD_SIZE = 500_000

def _stream_yaml_items(path: Path):
    '''
    Lazily parses a YAML list file item-by-item.

    Parameters:
    -----------
    path : Path
        Path to the YAML file.

    Yields:
    -------
    dict
        Parsed YAML list item as a dictionary.
    '''
    buf = []
    with path.open(encoding = 'utf-8') as f:
        for line in f:
            if line.startswith('- '):
                if buf:
                    yield yaml.safe_load(''.join(buf))
                    buf.clear()
                buf.append(line[2:])                        
            else:
                buf.append(line[2:] if line.startswith('  ') else line)
        if buf:
            yield yaml.safe_load(''.join(buf))

def shard_yaml_to_jsonl(source: Path, out_dir: Path, shard_size: int = 500_000):
    '''
    Splits a large YAML list file into multiple JSONL shard files.

    Parameters:
    -----------
    source : Path
        Path to the source YAML file.
    out_dir : Path
        Directory to write JSONL shards.
    shard_size : int
        Number of entries per shard.
    '''
    out_dir.mkdir(parents = True, exist_ok = True)

    with source.open(encoding = 'utf-8') as f:
        total_rows = sum(1 for line in f if line.startswith('- '))

    shard_records = []
    rows_in_shard = 0
    shard_idx = 0

    for rec in tqdm(_stream_yaml_items(source), total = total_rows, desc = 'Sharding', unit = 'row'):
        shard_records.append(rec)
        rows_in_shard += 1

        if rows_in_shard == shard_size:
            shard_path = out_dir / f'scrypt_{shard_idx:05d}.jsonl'
            with shard_path.open('w') as f:
                for r in shard_records:
                    json.dump(r, f)
                    f.write('\n')
            shard_idx += 1
            shard_records = []
            rows_in_shard = 0

    # write any final partial shard
    if shard_records:
        shard_path = out_dir / f'scrypt_{shard_idx:05d}.jsonl'
        with shard_path.open('w') as f:
            for r in shard_records:
                json.dump(r, f)
                f.write('\n')

__all__ = ['ROOT', 'SOURCE', 'SHARD_DIR', 'SHARD_SIZE', 'shard_yaml_to_jsonl']

if __name__ == '__main__':
    shard_yaml_to_jsonl(SOURCE, SHARD_DIR, SHARD_SIZE)