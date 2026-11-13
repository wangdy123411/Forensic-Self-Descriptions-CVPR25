from project_paths import PROJECT_ROOT, CODE_ROOT, WEIGHTS_ROOT

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

root = PROJECT_ROOT
sys.path.insert(0, str(CODE_ROOT))
from fsd import FSDDetector
import fsd.fsd_computation as computation

torch.set_num_threads(8)
data = root/'Data'/'tiny_genimage_smoke_200'
results = root/'Results'/'tiny_genimage_smoke_200'
output = results/'visualizations'/'intermediates'
output.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((results/'predictions.csv').open(encoding='utf-8-sig',newline='')))
real = sorted([r for r in rows if r['label']=='0' and r['correct']=='1'], key=lambda r:float(r['z_score']))
fake = sorted([r for r in rows if r['label']=='1' and r['correct']=='1'], key=lambda r:float(r['z_score']))
errors = [r for r in rows if r['correct']=='0']
assert len(errors)==1 and errors[0]['label']=='0'
selected = [('real_correct',real[len(real)//2]),('fake_correct',fake[len(fake)//2]),('false_positive',errors[0])]
detector = FSDDetector.load(weights_dir=WEIGHTS_ROOT,device='cuda')
captured = {}

def capture_fre(module, args, value):
    captured['residual_full'] = value.detach().clone()

def capture_transform(index):
    def hook(module, args, value):
        if index==0:
            captured['transform_input'] = args[0].detach().cpu().numpy().copy().reshape(-1)
        captured.setdefault('transforms',[]).append(value.detach().cpu().numpy().copy().reshape(-1))
    return hook

handles = [detector.fre.register_forward_hook(capture_fre)]
for index, transform in enumerate(detector.projections):
    handles.append(transform.register_forward_hook(capture_transform(index)))
original_solver = computation._solve_constrained_least_squares

def capture_solver(*args,**kwargs):
    vector = original_solver(*args,**kwargs)
    captured['fsd'] = vector.detach().cpu().numpy().copy()
    return vector

computation._solve_constrained_least_squares = capture_solver

def preview(tensor, maximum=192):
    h,w = tensor.shape[-2:]
    scale = min(1.0, maximum/max(h,w))
    shape = (max(1,round(h*scale)),max(1,round(w*scale)))
    display = F.interpolate(tensor[None],size=shape,mode='bilinear',align_corners=False,antialias=False)[0]
    return display.cpu().numpy().astype(np.float32)

actual_filters = detector.fre.conv.constrain(detector.fre.conv.w.double(), detector.fre.conv.one_middle.double())
arrays = {'filters':actual_filters.detach().cpu().numpy()[:,0].copy()}
cases = []
try:
    for index,(role,row) in enumerate(selected):
        captured.clear()
        prefix = 'c'+str(index)
        path = data/row['relative_path']
        with Image.open(path) as image:
            original_size = list(image.size)
            gray = np.array(image.convert('L'))
        result = detector.score(path)
        difference = abs(result.z_score-float(row['z_score']))
        if difference > 1e-8:
            raise RuntimeError(f'Changed inference score for {path}: {difference}')
        arrays[prefix+'_gray'] = gray
        border = detector.fre.conv.kernel_size//2
        residual = captured['residual_full'][:,border:-border,border:-border]
        residual_shape = list(residual.shape)
        arrays[prefix+'_residual'] = preview(residual)
        cfg = detector.config['fsd']
        max_size = cfg['max_size']
        if 'resize' in cfg['resize_mode']:
            h,w = residual.shape[-2:]
            factor = max_size/min(h,w)
            residual = F.interpolate(residual[None],size=(round(h*factor),round(w*factor)),mode='bilinear',antialias=False,align_corners=False)[0]
        if 'crop' in cfg['resize_mode']:
            h,w = residual.shape[-2:]
            crop_h,crop_w = min(max_size,h),min(max_size,w)
            start_h,start_w = (h-crop_h)//2,(w-crop_w)//2
            residual = residual[:,start_h:start_h+crop_h,start_w:start_w+crop_w]
        shapes = []
        for scale in range(cfg['num_scales']):
            scaled = F.interpolate(residual[None],scale_factor=1/2**scale,mode='bilinear',antialias=False,align_corners=False)[0]
            shapes.append(list(scaled.shape))
            arrays[prefix+'_scale'+str(scale)] = preview(scaled)
            h,w = scaled.shape[-2:]
            half = cfg['kernel_size']//2
            arrays[prefix+'_patch'+str(scale)] = scaled[:,h//2-half:h//2+half+1,w//2-half:w//2+half+1].cpu().numpy().copy()
        vector = captured['fsd']
        transformed = captured['transforms'][-1] if captured.get('transforms') else vector
        if captured.get('transforms'):
            assert np.array_equal(vector,captured['transform_input'])
        arrays[prefix+'_fsd'] = vector
        arrays[prefix+'_transformed'] = transformed
        arrays[prefix+'_transform_steps'] = np.stack([vector]+captured.get('transforms',[]))
        with torch.no_grad():
            x = torch.from_numpy(transformed).to(device=detector.fre.device,dtype=torch.float64).unsqueeze(0)
            components = detector.gmm._estimate_weighted_log_prob(x).detach().cpu()
            reconstructed = torch.logsumexp(components,dim=1).item()
        assert abs(reconstructed-result.raw_score)<1e-8
        arrays[prefix+'_component_log_prob'] = components.numpy()[0]
        cases.append({'id':prefix,'role':role,'relative_path':row['relative_path'],'label':int(row['label']),'generator':row['generator'],'z_score':result.z_score,'raw_score':result.raw_score,'threshold':result.threshold,'pred_label':int(result.is_fake),'original_size':original_size,'residual_shape':residual_shape,'scale_shapes':shapes,'score_match_abs_error':difference,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps(cases[-1]),flush=True)
        captured.clear()
        torch.cuda.empty_cache()
finally:
    computation._solve_constrained_least_squares = original_solver
    for handle in handles:
        handle.remove()

np.savez_compressed(output/'intermediates.npz',**arrays)
metadata = {'model_config':detector.config,'scoring':detector.config['scoring'],'transform_count':len(detector.projections),'cases':cases,'selection':'Median z-score correct real, median z-score correct fake, and the sole false positive. These are illustrative cases, not an additional evaluation set.','capture_method':'Passive forward hooks and a solver-return wrapper on the original inference path. Exported scores match the previously saved predictions within 1e-8.','notes':['Inference is unchanged float64. Feature vectors, patches and component scores are float64.','Residual and scale preview arrays alone are bilinearly downsampled to max side 192 and cast to float32 for display; they are never fed back into inference.','FRE residuals are signed pixel prediction errors, not forgery localization or attribution maps.','All three scales jointly fit one 960-dimensional FSD. Feature positions are coefficients, not image coordinates.','GMM components represent a learned density model, not generators or binary classes. Component values are weighted log-density terms, not probabilities.','Grayscale is applied before FRE. Residual resizing/cropping occurs after FRE.']}
(output/'export_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('EXPORT_COMPLETE', (output/'intermediates.npz').stat().st_size,flush=True)
