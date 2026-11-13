from project_paths import PROJECT_ROOT, DATA_ROOT

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


SEED_PREFIX = '20261111:'
SOURCES = {'ImageNet-1k', 'ProGAN', 'StyleGAN3', 'GLIDE', 'DALLE 3', 'FireFly'}
ONEDRIVE = PROJECT_ROOT
REPRO = PROJECT_ROOT / 'Results/reproduction'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def index_by_id(rows):
    indexed = {r['image_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate image IDs in upstream index')
    return indexed


def checked_file(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f'{path}: file SHA256 mismatch')
    return path


def sample_rows(rows):
    if [int(r['row']) for r in rows] != list(range(len(rows))):
        raise ValueError('upstream index order differs from row column')
    index_by_id(rows)
    groups = defaultdict(list)
    for row in rows:
        key = digest((SEED_PREFIX + row['image_id']).encode('ascii'))
        groups[row['source']].append((key, row['image_id'], row))
    if set(groups) != SOURCES or any(len(items) < 100 for items in groups.values()):
        raise ValueError('unexpected source set or sample availability')
    selected = []
    for items in groups.values():
        for rank, (key, _, row) in enumerate(sorted(items)[:100], start=1):
            selected.append((int(row['row']), rank, key, row))
    return sorted(selected)


def prepare(data, repro, output):
    formal = repro / 'section43_ossia_progan_20261110/4.3'
    integrated = repro / 'section41_final_20261110/integrated'
    supplement = data / 'supplemental/OSSIA_ProGAN_test1000'
    expected = json.loads((formal / 'checksums.json').read_text())
    run = json.loads(checked_file(formal / 'run.json', expected['run.json']).read_text())
    unique_manifest = data / 'manifests/eval_unique.csv'
    provenance = json.loads((integrated / 'provenance.json').read_text())
    original_rows = index_by_id(csv_rows(checked_file(unique_manifest, provenance['manifest_sha256'])))
    original_feat = index_by_id(csv_rows(integrated / 'feature_index.csv'))
    og = np.load(checked_file(integrated / 'raw_fsd_960.npy', run['original_integrated_npy_sha256']), mmap_mode='r')
    pg = np.load(checked_file(formal / 'replacement_features.npy', expected['replacement_features.npy']), mmap_mode='r')
    repl_manifest = checked_file(formal / 'replacement_manifest.csv', expected['replacement_manifest.csv'])
    checked_file(supplement / 'manifest.csv', expected['replacement_manifest.csv'])
    replacement_rows = index_by_id([dict(r, feature_row=i) for i, r in enumerate(csv_rows(repl_manifest))])
    rows = csv_rows(checked_file(formal / 'input_index.csv', expected['input_index.csv']))
    selected = sample_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new output directory')
    with TemporaryDirectory(dir=output.parent, prefix='prepare-') as scratch:
        stage = Path(scratch) / 'input'
        (stage / 'images').mkdir(parents=True)
        manifest, vectors = [], []
        for new_row, (prior_row, rank, key, item) in enumerate(selected):
            image_id, source = item['image_id'], item['source']
            if source == 'ProGAN':
                image = replacement_rows[image_id]
                vector = pg[image['feature_row']]
                root = supplement / 'data'
            else:
                image = original_rows[image_id]
                feature = original_feat[image_id]
                vector = og[int(feature['row'])]
                root = data
                if (feature['source'], feature['image_sha256'], feature['feature_sha256']) != (source, item['image_sha256'], item['feature_sha256']):
                    raise ValueError(f'{image_id}: integrated feature index mismatch')
            if (image['source'], image['split'], image['sha256']) != (source, 'test', item['image_sha256']):
                raise ValueError(f'{image_id}: image provenance mismatch')
            path = root / image['relative_path']
            raw = path.read_bytes()
            if digest(raw) != item['image_sha256']:
                raise ValueError(f'{image_id}: native image SHA256 mismatch')
            if vector.dtype.str != '<f8' or vector.shape != (960,) or not np.isfinite(vector).all() or digest(vector.tobytes()) != item['feature_sha256']:
                raise ValueError(f'{image_id}: descriptor differs from official prior row')
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                width, height = im.size
            target = Path('images') / (image_id + path.suffix.lower())
            (stage / target).write_bytes(raw)
            manifest.append(dict(row=new_row, image_id=image_id, source=source, relative_path=target.as_posix(),
                                 image_sha256=item['image_sha256'], feature_sha256=item['feature_sha256'], width=width,
                                 height=height, prior_row=prior_row, source_rank=rank, selection_sha256=key,
                                 upstream_relative_path=path.relative_to(data).as_posix()))
            vectors.append(vector)
        for field in ('image_id', 'relative_path', 'image_sha256', 'feature_sha256'):
            if len({r[field] for r in manifest}) != 600:
                raise ValueError(f'unexpected duplicated {field} in selected cohort')
        with (stage / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(manifest)
        np.save(stage / 'baseline.npy', np.stack(vectors), allow_pickle=False)

        written = np.load(stage / 'baseline.npy', allow_pickle=False)
        for image, row in zip(csv_rows(stage / 'manifest.csv'), written, strict=True):
            checked_file(stage / image['relative_path'], image['image_sha256'])
            if digest(row.tobytes()) != image['feature_sha256']:
                raise ValueError(f"baseline row hash mismatch after write: {image['image_id']}")
        info = dict(seed_prefix=SEED_PREFIX, selection='first 100 per source by SHA256(ASCII seed_prefix + image_id), sorted by prior 4.3 row',
                    source_counts=dict(Counter(r['source'] for r in manifest)), prior_index_sha256=expected['input_index.csv'],
                    progan_manifest_sha256=expected['replacement_manifest.csv'], prior_source_counts=run['source_counts'],
                    manifest_sha256=file_digest(stage / 'manifest.csv'), baseline_sha256=file_digest(stage / 'baseline.npy'),
                    size_bytes_images=sum((stage / r['relative_path']).stat().st_size for r in manifest),
                    verified_unique_images=600, verified_unique_feature_rows=600, shape=list(written.shape), dtype=written.dtype.str)
        (stage / 'selection.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        del written
        os.rename(stage, output)
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare')
    parser.add_argument('--data', type=Path, default=DATA_ROOT)
    parser.add_argument('--reproduction', type=Path, default=REPRO)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'input')
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.reproduction, args.out), indent=2))
