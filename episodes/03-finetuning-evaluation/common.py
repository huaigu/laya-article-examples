"""Portable helpers; no project-private imports or GPU work on import."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_bundle(root):
    root = Path(root); manifest = read(root / 'manifest.json')
    assert manifest['test_included'] is False
    assert 'test.json' not in manifest['files']
    for name, digest in manifest['files'].items():
        assert Path(name).name == name and '/' not in name and '\\' not in name and name not in ('.', '..'), 'Only flat, allowlisted files are accepted'
        assert sha(root / name) == digest, name
    return manifest


def optimizer_groups(model):
    groups = [[], []]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith('act_head.'))
        if parameter.requires_grad:
            groups[0 if parameter.ndim >= 2 else 1].append(parameter)
    return [{'params': groups[0], 'weight_decay': .01}, {'params': groups[1], 'weight_decay': 0.}]


def tensor_digest(named_tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        h.update(name.encode('utf-8'))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
