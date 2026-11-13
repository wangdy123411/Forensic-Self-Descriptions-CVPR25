from project_paths import WEIGHTS_ROOT

import hashlib
import json
from pathlib import Path
import urllib.request

root = WEIGHTS_ROOT
root.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
release_url = 'https://api.github.com/repos/ductai199x/Forensic-Self-Descriptions-CVPR25/releases/tags/v1.2.0'
with opener.open(urllib.request.Request(release_url, headers={'User-Agent':'CE7454-FSD'}), timeout=30) as response:
    release = json.load(response)
assets = {a['name']: a for a in release['assets']}
records = []
for name in ['config.json', 'fre.pt', 'gmm.pt', 'fsd_transforms.pt']:
    asset = assets[name]
    target = root / name
    expected = asset.get('digest', '')
    def valid(path):
        if not path.exists() or path.stat().st_size != asset['size']:
            return False
        if expected.startswith('sha256:'):
            return hashlib.sha256(path.read_bytes()).hexdigest() == expected.split(':')[1]
        return True
    if not valid(target):
        partial = target.with_name(name + '.part')
        print('Downloading', name, asset['size'], flush=True)
        request = urllib.request.Request(asset['browser_download_url'], headers={'User-Agent':'CE7454-FSD'})
        with opener.open(request, timeout=60) as response, partial.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if not valid(partial):
            raise RuntimeError('Download validation failed: '+name)
        partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    records.append({'name':name, 'bytes':target.stat().st_size, 'sha256':digest, 'release_digest':expected, 'url':asset['browser_download_url']})
    print('Verified',name,digest,flush=True)
(root/'download_manifest.json').write_text(json.dumps({'release':release['html_url'],'assets':records},indent=2),encoding='utf-8')
print((root/'config.json').read_text(),flush=True)
