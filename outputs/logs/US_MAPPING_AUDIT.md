# US Mapping Audit

## Source Availability

- `../SmartGuard/data/data/us/` exists: `False`
- `../SmartGuard/data/data/us/dictionary.py` exists: `False`
- `../SmartGen/SmartGen/us_keys_best.txt` exists: `True`
- `../SmartGen/SmartGen/dictionary.py` exists: `True`
- `../SmartGen/SmartGen/IoT_data/us` exists: `True`
- `../SmartGen/SmartGen/attack/us` exists: `True`

## SmartGen US Dictionaries

- `us_devices_dict` count: `40`
- `us_actions` count: `268`
- `us_keys_best.txt` device lines: `39`

## CausalGenGuard Existing US Outputs

- processed/synthetic US files: `[]`
- current mapping report: `{'exists': True, 'dataset': 'us', 'mapping_type': 'smartgen_textual', 'device_count': 40, 'control_count': 268, 'source_path': '/home/heyang/projects/SmartGen/SmartGen/dictionary.py', 'keys_best_path': '/home/heyang/projects/SmartGen/SmartGen/us_keys_best.txt', 'missing_key_controls': []}`

## US Data Format Samples

- `winter_trn`: `{'exists': True, 'record_count': 7049, 'sample_type': 'list', 'sample_preview': '[2, 0, 34, 229, 2, 0, 20, 122, 2, 0, 1, 14, 2, 0, 35, 240, 2, 0, 35, 245, 2, 0, 35, 240, 2, 0, 20, 123, 2, 0, 35, 240, 2, 0, 20, 123, 2, 0, 3, 38]', 'numeric_smartgen_flat4': True, 'textual_smartgen_flat4': False}`
- `spring_test`: `{'exists': True, 'record_count': 822, 'sample_type': 'list', 'sample_preview': '[0, 0, 10, 81, 0, 5, 10, 81, 1, 1, 10, 81, 2, 3, 10, 81, 6, 5, 10, 81]', 'numeric_smartgen_flat4': True, 'textual_smartgen_flat4': False}`
- `spring_synthetic`: `{'exists': True, 'record_count': 99, 'sample_type': 'list', 'sample_preview': '[0, 2, 2, 33, 0, 2, 15, 108, 0, 2, 10, 81, 0, 2, 10, 75]', 'numeric_smartgen_flat4': True, 'textual_smartgen_flat4': False}`
- `spring_attack`: `{'exists': True, 'record_count': 3016, 'sample_type': 'tuple', 'sample_preview': '([0, 0, 36, 257], 1)', 'numeric_smartgen_flat4': True, 'textual_smartgen_flat4': False}`

## Decision

- US data appears numeric SmartGen flat-4 style: `True`
- US data appears textual SmartGen flat-4 style: `False`
- SmartGuard US dictionary is unavailable, so SmartGuard numeric mapping cannot be generated.
- SmartGen US textual/action dictionary is available, so CausalGenGuard uses `mapping_type=smartgen_textual` generated from SmartGen `us_devices_dict/us_actions`.
- This does not reuse FR/SP mapping values and does not modify SmartGen.
