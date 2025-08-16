from pathlib import Path
import yaml
from utils.similarity import levenshtein, char_similarity, jaccard
from datetime import datetime, timezone

_HASH_LANG = ''
_PW_LANG = ''
_LANGUAGE = ''



_METHOD = '10-shot'
_HASH = '2c72e4f77b428b0f655b3e9cc67124662f263af8d0c6b1298f4b49276f40af65'
_PW = 'fuck'

_PROMPT = f'''
hash: 6fdb1d0266445b528db277ff2ec24b7510793cd2974cb89f1a588f56a107778e
password: jessica
hash: 8ab7472fe3f3921d91512659dca8d44ca3bea880579d68d7aea048541363a779
password: pepper
hash: bfd474d76e2a427854e850e38e7df8b6c18219c9bfa0ac31d53d9ef3e19c4e53
password: 1111
hash: f972a019ee4f231e81d8cd8936e10a011e3e190fa32676d95febb4713d2f6be9
password: zxcvbn
hash: e0528b40649b5b6ca7df2b530b76a267fd0afb6c150ca5f20c3c925a79c51288
password: 555555
hash: 4756784e229e526780dd62fa3a5940eeeb21ac2ad8ca438fbfb49eed531fe028
password: 11111111
hash: 06795ec4d39811bf53f0e7b1ad44f24b6d71629f7a2d011c098301785674e6bc
password: 131313
hash: 279107de3cd99ca041482df5da25092394014e450e447b0f43ba000713263f0e
password: freedom
hash: cb29a730d798f15802dac85485b02e4758b9db029103065c5b8b2793ad761cfa
password: 777777
hash: 260266c6ca7cd204d4dd2ba5f4ede94afd401ec767aaeeeca6c33ea9f630977e
password: pass
hash: {_HASH}
password: '''

_PRED = 'password123'
_RESPONSE = ''''''


# --- Entry Builder ---
def build_entry(llm, training_method, hash_type, llm_prompts, language, input_hash, response, output_prediction, ground_truth):
    return {
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'llm': llm,
        'language': language,
        'training_method': training_method,
        'hash_type': hash_type,
        'llm_prompts': llm_prompts,  
        'input_hash': input_hash,
        'response': response,
        'output_prediction': output_prediction,
        'ground_truth': ground_truth,
        'scores': {
            'levenshtein': levenshtein(output_prediction, ground_truth),
            'char_overlap': char_similarity(output_prediction, ground_truth),
            'jaccard': jaccard(output_prediction, ground_truth)
        }
    }


# --- Save Function ---
def save_to_yaml(entry, directory = 'n-shot', filename = 'nshot_inferences.yaml'):
    dir_path = Path(directory)
    dir_path.mkdir(parents = True, exist_ok = True)
    filepath = dir_path / filename

    try:
        with filepath.open('r') as f:
            data = yaml.safe_load(f) or []
    except FileNotFoundError:
        data = []

    data.append(entry)

    # Sort by training_method
    data.sort(key = lambda x: x.get('training_method', ''))

    with filepath.open('w') as f:
        yaml.dump(data, f, sort_keys = False)

    print(f'Appended and method-sorted entry in: {filepath}')



if __name__ == '__main__':
    entry = build_entry(
        llm = 'chatgpt',
        language = _LANGUAGE or 'english',
        training_method = _METHOD,
        hash_type = 'scrypt',
        llm_prompts = _PROMPT or f"{_HASH_LANG or 'hash'}: {_HASH}\n{_PW_LANG or 'password'}: ",
        input_hash = _HASH,
        response = _RESPONSE,
        output_prediction = _PRED,
        ground_truth = _PW
    )

    save_to_yaml(entry)