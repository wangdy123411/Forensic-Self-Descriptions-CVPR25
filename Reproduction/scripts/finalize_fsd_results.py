from project_paths import PROJECT_ROOT, CODE_ROOT, WEIGHTS_ROOT

import json
from pathlib import Path
import runpy
from verify_fsd_results import verify, digest

project = PROJECT_ROOT
results = project/'Results'/'tiny_genimage_smoke_200'
checked = verify(results, project/'Data'/'tiny_genimage_smoke_200')
summary = json.loads((results/'summary.json').read_text(encoding='utf-8'))
for name, expected in summary['source_sha256'].items():
    assert digest(CODE_ROOT/name) == expected, 'Source mismatch: '+name
for name, expected in summary['weight_sha256'].items():
    assert digest(WEIGHTS_ROOT/name) == expected, 'Weight mismatch: '+name
checked['original_source_files_verified'] = len(summary['source_sha256'])
checked['weight_files_verified'] = len(summary['weight_sha256'])
(results/'verification.json').write_text(json.dumps(checked, indent=2)+'\n',encoding='utf-8')
print(json.dumps(checked), flush=True)
runpy.run_path(str(Path(__file__).parent/'write_inference_report.py'))
