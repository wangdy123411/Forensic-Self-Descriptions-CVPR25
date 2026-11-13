import argparse
import csv
import json
import math
import shutil
from importlib.util import find_spec
from pathlib import Path


SOURCE_COLORS = {
    "ImageNet-1k": "#1f77b4",
    "ProGAN": "#ff7f0e",
    "StyleGAN3": "#2ca02c",
    "GLIDE": "#d62728",
    "SD1.5": "#9467bd",
    "DALLE 3": "#8c564b",
    "MJ v6": "#e377c2",
    "FireFly": "#7f7f7f",
}
SOURCE_LABELS = {"ImageNet-1k": "ImageNet-1k (real)", "DALLE 3": "DALL·E 3", "FireFly": "Firefly"}
CLUSTER_COLORS = ["#0072B2", "#D55E00", "#009E73", "#8D67B4", "#B18708", "#C95D91"]


def plotly_bundle(specified):
    if specified:
        return specified.read_text(encoding="utf-8")
    spec = find_spec("plotly")
    candidates = []
    if spec and spec.submodule_search_locations:
        candidates.extend(Path(root) / "package_data" / "plotly.min.js"
                          for root in spec.submodule_search_locations)

    python_on_path = shutil.which("python")
    if python_on_path:
        candidates.append(Path(python_on_path).parent / "Lib/site-packages/plotly/package_data/plotly.min.js")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("Pass --plotly-js /path/to/plotly.min.js, or install plotly in this environment.")


def load_data(directory):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8-sig"))
    ratios = metrics["explained_variance_ratio"]
    if len(ratios) != 3 or any(not 0 <= val <= 1 for val in ratios) or sum(ratios) > 1.000001:
        raise ValueError("Expected variance fractions for PC1, PC2 and PC3.")
    if metrics["k"] != len(CLUSTER_COLORS):
        raise ValueError("Expected the archived K=6 experiment.")
    groups = {}
    ids = set()
    with (directory / "coordinates.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id, source = row["image_id"], row["source"]
            if not image_id or image_id in ids or source not in SOURCE_COLORS:
                raise ValueError(f"Unexpected source or invalid image ID in row {reader.line_num}.")
            ids.add(image_id)
            group = groups.setdefault(source, {"points": [], "color": SOURCE_COLORS[source],
                                               "key": source, "label": SOURCE_LABELS.get(source, source)})
            coords = [float(row[f"pc{i}"]) for i in (1, 2, 3)]
            if not all(math.isfinite(v) for v in coords):
                raise ValueError(f"Non-finite PCA coordinate in row {reader.line_num}.")
            clusters = [int(row[key]) for key in ("cluster_960d", "cluster_3d")]
            if any(v < 0 or v >= metrics["k"] for v in clusters):
                raise ValueError(f"Invalid cluster assignment in row {reader.line_num}.")
            group["points"].append([*coords, image_id, *clusters])
    if not ids or len(ids) != metrics["n_images"]:
        raise ValueError("Image count differs between coordinates and metrics.")
    return {
        "groups": [groups[key] for key in SOURCE_COLORS if key in groups],
        "clusterColors": CLUSTER_COLORS,
        "n": len(ids),
        "variance": ratios,
        "nmi960": metrics["nmi_960d"],
        "nmi3": metrics["nmi_3d"],
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FSD source clustering in three dimensions</title>
<style>
*{box-sizing:border-box}
body{margin:0;color:#18262e;background:#f6f7f5;font:15px/1.45 system-ui,sans-serif}
main{max-width:1520px;margin:auto;padding:clamp(16px,3.1vw,48px)}
header{margin-bottom:22px} h1{font:normal clamp(28px,3.2vw,46px)/1.13 Georgia,serif;margin:0 0 8px}
header p,.hint,.detail{color:#4c5e65} header p{margin:0}
.panel{display:grid;grid-template-columns:minmax(0,1fr) 262px;background:#fff;border:1px solid #cfd8d8;border-radius:6px;overflow:hidden}
figure{margin:0;min-width:0;display:flex;flex-direction:column}
#plot{height:clamp(440px,70vh,820px);width:100%;flex:1 0 auto}
figcaption{padding:10px 16px;border-top:1px solid #e2e7e6;font-size:13px}
aside{padding:20px;border-left:1px solid #dbe2e1;display:flex;flex-direction:column;gap:20px}
fieldset{padding:0;margin:0;border:0;min-width:0} legend{font-size:14px;font-weight:700;margin-bottom:8px;padding:0}
label{display:flex;align-items:center;gap:7px;cursor:pointer;min-height:26px}
input{accent-color:#235e69;width:15px;height:15px;margin:0}
button{background:#fff;color:#264a54;border:1px solid #b8c8c8;border-radius:4px;padding:5px 9px;font:inherit;font-size:13px;cursor:pointer}
button:hover{background:#eef4f4} button:focus-visible,input:focus-visible{outline:2px solid #176276;outline-offset:3px}
.actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.swatch{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
.count{margin-left:auto;color:#5b6b71;font-size:12px;font-variant-numeric:tabular-nums}
#cluster-key{display:grid;grid-template-columns:repeat(3,1fr);gap:3px 5px;margin:8px 0}
#cluster-key span{display:flex;align-items:center;gap:6px;font-size:13px}
.detail{margin:8px 0 0;font-size:13px}.hint{margin:0}.status{border-top:1px solid #e2e7e6;padding-top:12px}
#visible{margin:0 0 8px;color:#344952;font-size:13px}
#error{padding:12px;color:#8c342b}
@media(max-width:760px){main{padding:14px}.panel{grid-template-columns:1fr}
aside{border-left:0;border-top:1px solid #dbe2e1;display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));align-items:start;padding:14px;gap:14px}
#plot{height:clamp(310px,58vh,570px)}.status{border-top:0;padding:0}}
</style>
</head>
<body><main>
<header><h1>FSD source clustering in three dimensions</h1><p id="summary"></p></header>
<section class="panel" aria-label="Interactive principal component plot">
<figure><div id="plot" role="img" aria-label="Rotatable scatter plot on the first three principal components"></div>
<figcaption class="hint">Drag to rotate · Scroll or pinch to zoom · Hover over a point for its image ID and cluster assignments.</figcaption></figure>
<aside>
<fieldset id="modes"><legend>Color points by</legend>
<label><input type="radio" name="mode" value="source" checked>Image source</label>
<label><input type="radio" name="mode" value="full">960D K-means cluster</label>
<label><input type="radio" name="mode" value="refit">3D K-means cluster</label>
<div id="cluster-key"></div><p id="mode-detail" class="detail"></p></fieldset>
<fieldset><legend>Visible sources</legend><div id="sources"></div>
<div class="actions"><button type="button" id="focus">ProGAN + GLIDE</button><button type="button" id="all">Show all</button></div></fieldset>
<div class="status"><p id="visible" aria-live="polite"></p><button type="button" id="reset">Reset view</button></div>
</aside></section><div role="alert" id="error" hidden></div>
<noscript>Enable JavaScript to rotate the scatter plot.</noscript></main>
<script>__PLOTLY__</script>
<script type="application/json" id="dataset">__DATA__</script>
<script>
'use strict';
const d = JSON.parse(document.getElementById('dataset').textContent);
const el = id => document.getElementById(id);
const plot = el('plot');
const count = v => v.toLocaleString('en-US');
const pct = v => (v * 100).toFixed(1) + '%';
const initialCamera = {eye:{x:1.55,y:1.6,z:1.05},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0},projection:{type:'orthographic'}};
const copyCamera = () => JSON.parse(JSON.stringify(initialCamera));
const checked = [], sourceDots = [];
const colorSets = {source:[],full:[],refit:[]};
el('summary').textContent = count(d.n) + ' images · ' + d.groups.length + ' sources · ' +
  pct(d.variance.reduce((a,b) => a+b, 0)) + ' of feature variance in PC1–PC3';
function swatch(color) {
  const dot = document.createElement('i');
  dot.className = 'swatch'; dot.style.backgroundColor = color;
  return dot;
}
const traces = d.groups.map((g, i) => {
  const label = document.createElement('label');
  const box = document.createElement('input'); box.type = 'checkbox'; box.checked = true;
  box.value = g.key;
  const dot = swatch(g.color), n = document.createElement('span');
  n.className = 'count'; n.textContent = count(g.points.length);
  label.append(box, dot, document.createTextNode(g.label), n);
  el('sources').append(label); checked.push(box); sourceDots.push(dot);
  colorSets.source.push(g.color);
  colorSets.full.push(g.points.map(p => d.clusterColors[p[4]]));
  colorSets.refit.push(g.points.map(p => d.clusterColors[p[5]]));
  return {type:'scatter3d',mode:'markers',name:g.label,
    x:g.points.map(p => p[0]),y:g.points.map(p => p[1]),z:g.points.map(p => p[2]),
    customdata:g.points.map(p => [p[3],p[4],p[5]]),
    hovertemplate:'<b>' + g.label + '</b><br>Image %{customdata[0]}<br>960D K-means C%{customdata[1]}<br>3D K-means C%{customdata[2]}<br>PC1 %{x:.3f} · PC2 %{y:.3f} · PC3 %{z:.3f}<extra></extra>',
    marker:{size:2.7,opacity:0.85,color:g.color,line:{width:0}},showlegend:false};
});
d.clusterColors.forEach((color, i) => {
  const item = document.createElement('span'); item.append(swatch(color),document.createTextNode('C'+i));
  el('cluster-key').append(item);
});
const sourceBounds = d.groups.map(g => [0,1,2].map(i => {
  let low=Infinity, high=-Infinity;
  for (const p of g.points) {low=Math.min(low,p[i]); high=Math.max(high,p[i]);}
  return [low,high];
}));
function fittedRanges(visible) {
  const active = sourceBounds.filter((_,i) => visible[i]);
  const bounds = active.length ? active : sourceBounds;
  return [0,1,2].map(i => {
    const low=Math.min(...bounds.map(b => b[i][0])), high=Math.max(...bounds.map(b => b[i][1]));
    const pad=low === high ? 1 : (high-low)*0.06;
    return [low-pad,high+pad];
  });
}
const initialRanges = fittedRanges(checked.map(() => true));
function axis(i) {
  return {title:{text:'PC' + (i+1) + ' (' + pct(d.variance[i]) + ')',font:{size:13}},
    range:initialRanges[i],autorange:false,nticks:5,showbackground:true,backgroundcolor:'#fafbf9',
    gridcolor:'#dae1e0',zerolinecolor:'#bbc8c9',tickfont:{size:10}};
}
const layout = {margin:{l:0,r:0,t:8,b:0},paper_bgcolor:'#fff',showlegend:false,
  font:{family:'Arial, sans-serif',color:'#263a43'},uirevision:'principal-components',
  scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:'data',camera:copyCamera(),dragmode:'orbit'}};
const config = {responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:'hover',
  modeBarButtonsToRemove:['toImage','sendDataToCloud','resetCameraDefault3d','resetCameraLastSave3d']};
function showError(error) {
  el('error').hidden = false;
  el('error').textContent = 'Could not display the 3D plot. Check that WebGL is enabled in the browser. ' + String(error);
}
function updateControls(mode) {
  const visible = d.groups.reduce((n,g,i) => n + (checked[i].checked ? g.points.length : 0), 0);
  el('visible').textContent = count(visible) + ' of ' + count(d.n) + ' images visible';
  sourceDots.forEach(dot => {dot.style.display = mode === 'source' ? '' : 'none';});
  el('cluster-key').style.display = mode === 'source' ? 'none' : 'grid';
  const note = el('mode-detail');
  note.hidden = mode === 'source';
  if (!note.hidden) {
    note.textContent = (mode === 'full' ? 'Fit on the full 960D features.' :
      'Fit on PC1–PC3. Cluster IDs are matched to the 960D fit by maximum overlap.') +
      ' Source NMI (all images): ' + (mode === 'full' ? d.nmi960 : d.nmi3).toFixed(3) + '.';
  }
}
let latest = Promise.resolve();
function redraw(fitAxes) {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const visible = checked.map(box => box.checked);
  updateControls(mode);
  const style = {visible:visible,'marker.color':colorSets[mode]};
  if (fitAxes) {
    const ranges = fittedRanges(visible);
    const axes = {'scene.xaxis.range':ranges[0],'scene.yaxis.range':ranges[1],'scene.zaxis.range':ranges[2]};
    latest = latest.then(() => Plotly.update(plot,style,axes)).catch(showError);
  } else {
    latest = latest.then(() => Plotly.restyle(plot,style)).catch(showError);
  }
}
updateControls('source');
el('focus').hidden = !['ProGAN','GLIDE'].every(key => d.groups.some(g => g.key === key));
Plotly.newPlot(plot,traces,layout,config).then(() => {
  checked.forEach(box => box.addEventListener('change', () => redraw(true)));
  el('modes').addEventListener('change', () => redraw(false));
  el('all').addEventListener('click', () => {checked.forEach(box => box.checked=true); redraw(true);});
  el('focus').addEventListener('click', () => {
    checked.forEach(box => box.checked = ['ProGAN','GLIDE'].includes(box.value)); redraw(true);
  });
  el('reset').addEventListener('click', () => {
    latest = latest.then(() => Plotly.relayout(plot, {'scene.camera':copyCamera()})).catch(showError);
  });
}).catch(showError);
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description='viewer')
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "output",
                        help="Directory containing coordinates.csv and metrics.json")
    parser.add_argument("--output", type=Path, help="Default: cluster_3d.html in --data-dir")
    parser.add_argument("--plotly-js", type=Path, help="Path to a local plotly.min.js bundle")
    args = parser.parse_args()
    data = load_data(args.data_dir)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", r"\u003c")
    dest = args.output or args.data_dir / "cluster_3d.html"
    dest.write_text(HTML.replace("__DATA__", payload).replace("__PLOTLY__", plotly_bundle(args.plotly_js), 1),
                    encoding="utf-8")
    print(f"Wrote {dest} with {data['n']:,} images.")


if __name__ == "__main__":
    main()
