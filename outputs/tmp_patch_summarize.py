from pathlib import Path
path = Path('src/causal_gen_guard/evaluation/summarize_results.py')
text = path.read_text(encoding='utf-8')
old = '''    if 'smartguard only' in method or 'smartgen' in method or 'causaltof' in method or 'ablation' in text:
        return 'ablation'
'''
new = '''    ablation_methods = (
        'smartguard only',
        'smartguard + smartgen',
        'smartguard + causal',
        'smartguard + smartgen + causal',
        'full causalgenguard',
    )
    if any(item in method for item in ablation_methods) or 'causaltof' in method or 'ablation' in text:
        return 'ablation'
'''
if old not in text:
    raise SystemExit('target classification block not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
