"""Validate CItest summary completeness without reimplementing evaluation."""
import argparse
import json
import math
from pathlib import Path


def validate_results(cases, notice):
    errors = []
    expected = {name: case['evalscope']['datasets'] for name, case in cases.items()
                if case.get('type') == 'precision'}
    if not expected:
        raise ValueError('No precision cases configured')
    if not isinstance(notice, dict) or not isinstance(notice.get('test_result'), list):
        raise ValueError('CItest summary must contain a test_result list')
    results = {}
    for entry in notice['test_result']:
        if not isinstance(entry, dict) or not isinstance(entry.get('model_name'), str):
            errors.append('Invalid model result entry')
            continue
        name = entry['model_name']
        if name in results:
            errors.append(f'Duplicate model result: {name}')
        results[name] = entry
        if name not in expected:
            errors.append(f'Unexpected model: {name}')
    cells = []
    for name, datasets in expected.items():
        if len(set(datasets)) != len(datasets) or not datasets:
            raise ValueError(f'Empty or duplicate datasets configured: {name}')
        for dataset in datasets:
            key = dataset.replace('_', '-')
            score = results.get(name, {}).get(key)
            valid = (not isinstance(score, bool) and isinstance(score, (int, float))
                     and math.isfinite(score) and 0 <= score <= 100)
            cells.append({'model': name, 'dataset': dataset,
                          'status': 'present' if valid else 'missing_or_invalid',
                          'score': score if valid else None})
            if not valid:
                errors.append(f'Missing or invalid score: {name}/{dataset}')
    return {'complete': not errors, 'expected_result_count': len(cells),
            'valid_result_count': sum(cell['status'] == 'present' for cell in cells),
            'results': cells, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-conf', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--evaluation-outcome', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = {'complete': False, 'errors': []}
    try:
        import yaml
        cases = yaml.safe_load(Path(args.case_conf).read_text(encoding='utf-8'))
        if not isinstance(cases, dict):
            raise ValueError('Case config must be a mapping')
        notice = json.loads((root / 'notificate.json').read_text(encoding='utf-8'))
        result = validate_results(cases, notice)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result['errors'].append(str(exc))
    if args.evaluation_outcome != 'success':
        result['complete'] = False
        result['errors'].append(f'CItest step outcome: {args.evaluation_outcome}')
    (root / 'completeness.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 模型测试完整性', '', f"状态：{'完整' if result['complete'] else '失败或结果不完整'}", '',
             f"有效结果：{result.get('valid_result_count', 0)} / {result.get('expected_result_count', '未知')}", '',
             '本检查只验证执行与评分完整性，尚未判定相对基线的精度回归。', '',
             '| 模型 | 数据集 | 状态 | 分数（%） |', '|---|---|---|---:|']
    for cell in result.get('results', []):
        lines.append(f"| {cell['model']} | {cell['dataset']} | {cell['status']} | {cell['score'] if cell['score'] is not None else '—'} |")
    lines += ['', *[f'- {error}' for error in result['errors']]]
    (root / 'completeness.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for error in result['errors']:
        print(error)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
