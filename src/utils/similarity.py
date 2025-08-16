
# --- similarity Functions ---
def char_similarity(pred: str, truth: str) -> float:
    if pred == '':
        return 0.0
    length = max(len(pred), len(truth))
    pred = pred.ljust(length)
    truth = truth.ljust(length)
    return sum(p == t for p, t in zip(pred, truth)) / length


def _levenshtein_helper(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein_helper(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)
    
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein(pred: str, truth: str) -> float:
    if pred == '':
        return 0.0
    dist = _levenshtein_helper(pred, truth)
    max_len = max(len(pred), len(truth))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def jaccard(pred: str, truth: str) -> float:
    if pred == '':
        return 0.0
    set_pred = set(pred)
    set_truth = set(truth)
    intersection = set_pred & set_truth
    union = set_pred | set_truth
    return len(intersection) / len(union) if union else 1.0

__all__ = ['char_similarity', 'levenshtein', 'jaccard']