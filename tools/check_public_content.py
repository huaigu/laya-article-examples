"""Check publication scope. This is a heuristic guard, not approval to publish."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'api-token-like': re.compile(r'(?:sk|ak)-[A-Za-z0-9_-]{20,}'),
    'github-token-like': re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}'),
    'private-key-block': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'personal-windows-path': re.compile(r'[A-Za-z]:[/\\](?:Users|github)[/\\]', re.I),
}


def scan_text(text):
    return [label for label, pattern in PATTERNS.items() if pattern.search(text)]


def check(root=ROOT):
    allowed = set((root / 'PUBLIC_FILES.txt').read_text(encoding='utf-8').splitlines())
    issues = []
    for name in allowed:
        path = Path(name)
        if not name or path.is_absolute() or '..' in path.parts:
            issues.append({'file': name, 'issue': 'invalid-allowlist-entry'})
    if issues:
        return {'ok': False, 'issues': issues}
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        capture_output=True, check=True,
    )
    candidates = {name for name in result.stdout.decode('utf-8').split(chr(0)) if name}
    for name in sorted(candidates - allowed):
        issues.append({'file': name, 'issue': 'not-allowlisted'})
    for name in sorted(allowed):
        path = root / name
        if path.is_symlink() or not path.is_file():
            issues.append({'file': name, 'issue': 'missing-file-or-symlink'})
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            issues.append({'file': name, 'issue': 'unexpected-binary-file'})
            continue
        for category in scan_text(text):
            # Report only the file and category, never the matching secret.
            issues.append({'file': name, 'issue': category})
    return {'ok': not issues, 'allowlisted_files': len(allowed),
            'candidate_files': len(candidates), 'issues': issues,
            'note': 'Heuristic content check only. Human review and a license decision are still required.'}


def main():
    result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
