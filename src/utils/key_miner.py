import json
import yaml
import csv
from pathlib import Path
from glob import iglob

class KeyMiner:
    '''
    A one-shot callable class to extract values for a given dictionary key from JSON, JSONL, or YAML files,
    and append them to a TXT, CSV, or TSV file.
    '''

    def __new__(cls, key: str, /, *, file: str, save_as: str, root: str | Path = None, dedup: bool = False) -> list:
        instance = super().__new__(cls)
        return instance(key, f = file, ext = save_as, root = root, dedup = dedup)

    def __call__(self, k: str, /, *, f: str, ext: str, root: str | Path, dedup: bool) -> list:
        '''
        Extracts values from all matching files and appends them to a single output file.

        Parameters:
        -----------
        key : str
            The dictionary key to extract values for.
        file : str
            Glob pattern or file path.
        save_as : str
            Output file format: 'txt', 'csv', or 'tsv'.
        root : str | Path
            Optional root path override.
        dedup : bool
            Whether to remove duplicate values before writing.

        Returns:
        --------
        list
            All extracted values across all files.
        '''
        # smart glob 
        if any(sym in f for sym in ['*', '?', '[']):
            paths = [Path(p) for p in iglob(f, recursive = True)]
            paths = sorted(paths, key = lambda p: str(p))
        else:
            paths = [Path(f)]


        if not paths:
            raise FileNotFoundError(f'no files matched: {f}')

        v = []
        for path in paths:
            suffix = path.suffix.lower()
            data = self._load(path, suffix)
            v = self._extract_v(data, k)
            v.extend(v)

        if dedup:
            v = list(dict.fromkeys(v))  # preserves order

        out_dir = Path(root) if root else Path.cwd()
        out_dir.mkdir(parents = True, exist_ok = True)
        self._save(k, v, o = out_dir, ext = ext.lower())

        return v

    @staticmethod
    def _load(path: Path, suffix: str, /) -> list:
        # load data depending on file suffix
        if suffix == '.json':
            with open(path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
        elif suffix == '.jsonl':
            with open(path, 'r', encoding = 'utf-8') as f:
                data = [json.loads(l) for l in f if l.strip()]
        elif suffix in {'.yaml', '.yml'}:
            with open(path, 'r', encoding = 'utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f'unsupported input file type: {suffix}')

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError('data must be a list of dictionaries or a dictionary')

    @staticmethod
    def _extract_v(d: list, k: str, /) -> list:
        # recursively extract values for the given key from nested structures
        def _recurse(entry):
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if key == k:
                        yield value
                    else:
                        yield from _recurse(value)
            elif isinstance(entry, list):
                for item in entry:
                    yield from _recurse(item)

        values = []
        for entry in d:
            values.extend(_recurse(entry))
        return values


    @staticmethod
    def _save(k: str, v: list, /, *, o: Path, ext: str):
        # append values to output file based on output type
        ofile = o / f'{k}.{ext}'

        if ext == 'txt':
            with open(ofile, 'a', encoding = 'utf-8') as f:
                for _v in v:
                    f.write(str(_v) + '\n')
        elif ext in {'csv', 'tsv'}:
            delimiter = ',' if ext == 'csv' else '\t'
            with open(ofile, 'a', newline = '', encoding = 'utf-8') as f:
                writer = csv.writer(f, delimiter = delimiter)
                for _v in v:
                    writer.writerow([_v])
        else:
            raise ValueError(f'unsupported output file type: {ext}')

if __name__ == '__main__':
    

    f = f'/home/athena--/.__projects/school/nlp/final/results/gpt/training_metrics.jsonl'
    mined = KeyMiner('epoch', file = f, save_as = 'csv', root = Path.cwd().parent / 'metrics')