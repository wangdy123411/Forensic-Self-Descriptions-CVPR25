from __future__ import annotations

from project_paths import EXTENSION_ROOT, REFERENCE_PATH


import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import io
import json
import math
from pathlib import Path
import sys

if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))
from zeroshot_common import (
    FAKE_SOURCES, MANIFEST_FIELDS, PAPER_COUNTS, REAL_SOURCES,
    file_hash, metrics_for, write_csv, write_json,
)

DEFAULT_REFERENCE = REFERENCE_PATH
LOCAL_METHOD = "FSD v1.2.0 (local GPU measurement)"
PAPER_METHOD = "FSD (paper published reference; 24 generators)"
CAVEATS = [
    "本地论文 v1 的 zero-shot 方法为 §4.1，评测为 §5.3；§4.2 是开放集来源归因。本报告仅验证 zero-shot 检测。",
    "AUC 的正类为假图，排序分数为 -z_score；逐个真实来源 × 生成器计算，再不加权宏平均。缺失项不补零、不插值。",
    "本地结果使用发布的 FSD v1.2.0 权重与实现，不是论文原始数值的严格复现：发布实现采用 FRE 15×15 与 KKT 最小二乘；论文 §5.1 描述 FRE 11×11 与 AdamW 拟合。",
    "ImageNet-22k 来自 timm 上游处理版的 train 分片；本地 split=test 仅表示评测用途，不能解释为官方 held-out 测试集。",
    "未取得作者的原始测试图 ID 及训练排除清单，不能证明与 FSD 训练数据无重叠；不重新训练，也不在测试图上调阈值。",
    "论文参考值来自原论文，已经四舍五入；不是在本机复跑的基线。当前仅运行 FSD，不声称复跑了其他方法。",
    "不同来源公开版本、采样范围和去重会影响结果。部分覆盖平均与论文完整 96 对平均的分母不同，不能作严格同划分比较。",
    "只有该来源清单中的所有图像均推理成功后才计算相应 AUC；待处理或有失败的来源显示 N/A。",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_prediction_snapshot(path: Path) -> tuple[list[dict], bool]:

    if not path.exists():
        return [], False
    raw = path.read_bytes()
    incomplete_tail = bool(raw) and not raw.endswith(b"\n")
    if incomplete_tail:
        raw = raw[: raw.rfind(b"\n") + 1]
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))), incomplete_tail


def validate_predictions(manifest: list[dict], predictions: list[dict]) -> None:
    expected = {r["image_id"]: r for r in manifest}
    if not manifest or len(expected) != len(manifest):
        raise ValueError("Manifest must be nonempty and contain unique image IDs")
    seen = set()
    for row in predictions:
        key = row.get("image_id")
        if key not in expected or key in seen:
            raise ValueError(f"Unknown or duplicate prediction image ID: {key}")
        seen.add(key)
        if any(row.get(k) != expected[key].get(k) for k in MANIFEST_FIELDS):
            raise ValueError(f"Prediction does not match manifest metadata: {key}")
        if row.get("status") not in ("ok", "error"):
            raise ValueError(f"Invalid prediction status: {key}")
        if row["status"] == "ok":
            z, fake, raw, threshold = (float(row[k]) for k in ("z_score", "fake_score", "raw_score", "threshold"))
            if not all(math.isfinite(v) for v in (z, fake, raw, threshold)) or fake != -z:
                raise ValueError(f"Nonfinite or incorrectly oriented score: {key}")
            if row.get("predicted_fake") != str(int(z < threshold)):
                raise ValueError(f"Prediction violates fixed threshold: {key}")


def build_report(manifest: list[dict], predictions: list[dict], reference: dict, dataset_status: dict | None = None) -> dict:
    validate_predictions(manifest, predictions)
    dataset_status = dataset_status or {}
    expected = Counter(r["source"] for r in manifest)
    done = Counter(r["source"] for r in predictions)
    ok = Counter(r["source"] for r in predictions if r["status"] == "ok")
    errors = Counter(r["source"] for r in predictions if r["status"] == "error")
    groups = defaultdict(list)
    for row in predictions:
        if row["status"] == "ok":
            groups[row["source"]].append(row)
    generators = reference.get("generator_order", list(FAKE_SOURCES))
    reals = reference.get("real_source_order", list(REAL_SOURCES))
    if set(generators) != set(FAKE_SOURCES) or set(reals) != set(REAL_SOURCES):
        raise ValueError("Reference source names do not match paper protocol")
    complete_sources = {s for s in expected if ok[s] == expected[s]}
    state = ("PARTIAL" if len(predictions) < len(manifest) else
             "COMPLETED_WITH_ERRORS" if sum(errors.values()) else "COMPLETE_AVAILABLE_DATA")

    source_rows = []
    for source in [*reals, *generators]:
        status = ("missing_skipped" if not expected[source] else
                  "complete" if source in complete_sources else
                  "inference_errors" if errors[source] else "pending")
        reason = dataset_status.get("missing_sources", {}).get(source, "Not present in the evaluation manifest") if not expected[source] else ""
        source_rows.append(dict(source=source, label=0 if source in reals else 1,
                                paper_count=PAPER_COUNTS[source], selected_count=expected[source],
                                processed_count=done[source], successful_count=ok[source],
                                failed_count=errors[source], status=status, reason=reason))

    pair_rows = []
    for real in reals:
        for fake in generators:
            if real in complete_sources and fake in complete_sources:
                pair_rows.append({"real_source": real, "fake_source": fake, **metrics_for(groups[real] + groups[fake])})
    pair_lookup = {(r["real_source"], r["fake_source"]): r for r in pair_rows}
    measured_row = {"method": LOCAL_METHOD, "value_kind": "local_measured", "run_status": state}
    reference_row = {"method": PAPER_METHOD, "value_kind": "paper_reference", "run_status": "published"}
    subset_row = {"method": "FSD (paper reference; same finished generator subset, rounded values)",
                  "value_kind": "paper_reference_subset", "run_status": state}
    source_status = {r["source"]: r["status"] for r in source_rows}
    appendix, worst_rows, subset_values = [], [], []
    paper_fsd = {}
    for number, table in reference.get("appendix_tables", {}).items():
        fsd_rows = [r for r in table["rows"] if r["method"] in ("Ours", "FSD")]
        if len(fsd_rows) != 1:
            raise ValueError(f"Expected one FSD paper row in appendix table {number}")
        paper_fsd[table["real_source"]] = fsd_rows[0]["auc"]

    for number, real in enumerate(reals, start=10):
        pairs = [r for r in pair_rows if r["real_source"] == real]
        measured_row[real] = sum(r["roc_auc"] for r in pairs) / len(pairs) if pairs else None
        measured_row[real + "_pairs"] = len(pairs)
        reference_row[real] = reference.get("main_table_1_fsd", {}).get(real)
        reference_row[real + "_pairs"] = 24 if reference_row[real] is not None else 0
        matching_reference = [paper_fsd.get(real, {}).get(r["fake_source"]) for r in pairs]
        matching_reference = [x for x in matching_reference if x is not None]
        subset_row[real] = sum(matching_reference) / len(matching_reference) if matching_reference else None
        subset_row[real + "_pairs"] = len(matching_reference)
        subset_values.extend(matching_reference)
        worst = min((r["roc_auc"] for r in pairs), default=None)
        worst_generators = [r["fake_source"] for r in pairs if r["roc_auc"] == worst]
        published_worst = reference.get("main_table_2_fsd", {}).get(real, {})
        worst_rows.append(dict(real_source=real, measured_worst_auc=worst, measured_generators="; ".join(worst_generators),
                               evaluated_generator_count=len(pairs), paper_worst_auc=published_worst.get("auc"),
                               paper_generator=published_worst.get("generator", ""), paper_generator_count=24,
                               run_status=state, real_source_status=source_status[real]))
        rows = []
        for fake in generators:
            pair = pair_lookup.get((real, fake))
            pair_status = ("complete" if pair else
                           "missing_real_skipped" if not expected[real] else
                           "missing_generator_skipped" if not expected[fake] else
                           "source_inference_errors" if errors[real] or errors[fake] else "pending")
            rows.append(dict(generator=fake, paper_reference_auc=paper_fsd.get(real, {}).get(fake),
                             local_measured_auc=pair["roc_auc"] if pair else None,
                             n_real=expected[real], n_fake=expected[fake], status=pair_status))
        appendix.append(dict(table_number=number, real_source=real, rows=rows))
    measured_row["Average"] = sum(r["roc_auc"] for r in pair_rows) / len(pair_rows) if pair_rows else None
    measured_row["Average_pairs"] = len(pair_rows)
    reference_row["Average"] = reference.get("main_table_1_fsd", {}).get("Average")
    reference_row["Average_pairs"] = 96 if reference_row["Average"] is not None else 0
    subset_row["Average"] = sum(subset_values) / len(subset_values) if subset_values else None
    subset_row["Average_pairs"] = len(subset_values)
    return dict(
        schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(), run_status=state,
        complete_available_data=state == "COMPLETE_AVAILABLE_DATA", exact_paper_reproduction=False,
        selected_count=len(manifest), processed_count=len(predictions), successful_count=sum(ok.values()),
        failed_count=sum(errors.values()), remaining_count=len(manifest) - len(predictions),
        paper_expected_pairs=96, measured_complete_pairs=len(pair_rows),
        planned_available_pairs=sum(bool(expected[r]) for r in reals) * sum(bool(expected[g]) for g in generators),
        paper_96_pair_macro_auc=None, macro_auc_available_finished_pairs=measured_row["Average"],
        real_source_order=reals, generator_order=generators,
        main_table_1=[reference_row, subset_row, measured_row], main_table_2=worst_rows,
        appendix_tables=appendix, source_counts=source_rows, pairwise_metrics=pair_rows,
        caveats=CAVEATS + dataset_status.get("provenance_notes", []),
        excluded_nonpaper_sources=sorted(set(expected) - set(reals) - set(generators)),
        paper_source_url=reference.get("source_url", "https://arxiv.org/abs/2503.21003v1"),
    )


def display(value) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def html_table(headers, rows) -> str:
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(display(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def latex_escape(value) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in display(value))


def latex_table(caption: str, headers: list[str], rows: list[list]) -> str:
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", "\\caption{" + latex_escape(caption) + "}",
             r"\begin{tabular}{" + "l" * len(headers) + "}", r"\hline",
             " & ".join(latex_escape(h) for h in headers) + r" \\", r"\hline"]
    lines.extend(" & ".join(latex_escape(c) for c in row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def export_files(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "table_report.json", report)
    reals = report["real_source_order"]
    main_fields = ["method", "value_kind", "run_status"] + [c for r in [*reals, "Average"] for c in (r, r + "_pairs")]
    write_csv(output / "main_table_1_macro_auc.csv", main_fields, report["main_table_1"])
    write_csv(output / "main_table_2_worst_auc.csv", list(report["main_table_2"][0]), report["main_table_2"])
    write_csv(output / "source_counts.csv", list(report["source_counts"][0]), report["source_counts"])
    pair_fields = ["real_source", "fake_source", "n", "n_real", "n_fake", "roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "false_positive_rate", "tn", "fp", "fn", "tp"]
    write_csv(output / "pairwise_metrics.csv", pair_fields, report["pairwise_metrics"])
    main_headers = ["Method / scope", *reals, "Average", "Pairs"]
    main_rows = [[r["method"], *[r[x] for x in reals], r["Average"], r["Average_pairs"]] for r in report["main_table_1"]]
    worst_headers = ["Real source", "Local worst AUC", "Local generator(s)", "Pairs", "Paper worst AUC", "Paper generator"]
    worst_rows = [[r[k] for k in ("real_source", "measured_worst_auc", "measured_generators", "evaluated_generator_count", "paper_worst_auc", "paper_generator")] for r in report["main_table_2"]]

    tex_main_rows = [[real, r[real], r[real + "_pairs"], report["main_table_1"][0][real]] for real in reals for r in report["main_table_1"] if r["value_kind"] == "local_measured"]
    tex = "% FSD zero-shot validation. Generated from image-level predictions.\n% Status: " + report["run_status"] + "\n% Reference values are published, not rerun. Missing sources are skipped.\n% Requires no nonstandard LaTeX packages. All AUCs are fractions.\n"
    tex += latex_table("Main Table 1 equivalent: local macro AUC over completed available generators. " + report["run_status"], ["Real source", "Local AUC", "Pairs", "Paper (24)"], tex_main_rows)
    tex += "% Local overall available-pair macro AUC: " + display(report["macro_auc_available_finished_pairs"]) + "; denominator=" + str(report["measured_complete_pairs"]) + "/96.\n"
    tex += latex_table("Main Table 2 equivalent: worst AUC among completed available generators. " + report["run_status"], ["Real source", "Local worst", "Generator(s)", "Pairs"], [r[:4] for r in worst_rows])
    body = "<h1>FSD zero-shot 本地 GPU 验证</h1>"
    body += f"<p class='status'>{html.escape(report['run_status'])}</p>"
    body += f"<p>已成功 {report['successful_count']:,} / {report['selected_count']:,} 张；失败 {report['failed_count']:,}；待处理 {report['remaining_count']:,}。已完整评测 {report['measured_complete_pairs']} / {report['planned_available_pairs']} 个可用来源对（论文完整协议为 96 对）。</p>"
    if not report["complete_available_data"]:
        body += "<p class='warning'>当前为未完成结果：N/A 表示缺失、尚未完成或有推理错误。不得作为完整评测结果引用。</p>"
    else:
        body += "<p>可用数据评测已完成。缺失来源已跳过；该结果仍属于论文数据集的部分覆盖验证。</p>"
    body += "<h2>正文表 1：宏平均 ROC AUC</h2>" + html_table(main_headers, main_rows)
    body += "<p>每个真实来源列的分母见 CSV 中的 *_pairs。Average 直接对已完成来源对取不加权平均。论文同子集参考值由附录中已四舍五入的 AUC 求平均，仅供描述性对照。</p>"
    body += "<h2>正文表 2：最低 ROC AUC 及生成器</h2>" + html_table(worst_headers, worst_rows)
    for table in report["appendix_tables"]:
        number, real = table["table_number"], table["real_source"]
        filename = f"appendix_table_{number}_{real.replace('-', '_')}.csv"
        write_csv(output / filename, list(table["rows"][0]), table["rows"])
        heads = ["Generator", "Paper reference", "Local measured", "N real", "N fake", "Status"]
        rows = [[r[k] for k in ("generator", "paper_reference_auc", "local_measured_auc", "n_real", "n_fake", "status")] for r in table["rows"]]
        body += f"<h2>附录表 {number}：{html.escape(real)}</h2>" + html_table(heads, rows)
        tex += latex_table(f"Appendix Table {number} equivalent, {real}. Paper values are published references. {report['run_status']}", ["Generator", "Paper", "Local", "N fake", "Status"], [[r[0], r[1], r[2], r[4], r[5]] for r in rows])
    body += "<h2>样本数与缺失项</h2>" + html_table(["Source", "Paper N", "Selected", "Scored", "Failed", "Status"], [[r[k] for k in ("source", "paper_count", "selected_count", "successful_count", "failed_count", "status")] for r in report["source_counts"]])
    body += "<h2>协议与解释边界</h2><ul>" + "".join("<li>" + html.escape(x) + "</li>" for x in report["caveats"]) + "</ul>"
    body += "<p>参考：<a href='" + html.escape(report["paper_source_url"], quote=True) + "'>论文 arXiv v1</a>。全部数值以 CSV/JSON 未舍入结果为准。</p>"
    body += "<p class='small'>Generated UTC: " + html.escape(report["generated_utc"]) + "</p>"
    css = "body{font:16px/1.65 system-ui,Arial,sans-serif;margin:40px auto;padding:0 24px;max-width:1200px;color:#18253b;background:#f6f8fb}h1,h2{line-height:1.25}h2{margin-top:38px}.table-wrap{overflow-x:auto;background:white;border:1px solid #d9e0eb;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 13px;border-bottom:1px solid #e1e7ef;font-variant-numeric:tabular-nums}th{background:#e9eff8;white-space:nowrap}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafbfd}.status{font-weight:700;color:#143e70}.warning{padding:12px;background:#fff2d4;border-left:4px solid #b87500}.small{color:#657185;font-size:12px}li{margin:8px 0}@media print{body{margin:12px;max-width:none;background:white;font-size:11px}table{font-size:9px}th,td{padding:4px 6px}.table-wrap{overflow:visible}h2{page-break-after:avoid}tr{page-break-inside:avoid}}"
    (output / "report.html").write_text("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>FSD zero-shot validation</title><style>" + css + "</style></head><body>" + body + "</body></html>\n", encoding="utf-8")
    (output / "paper_tables.tex").write_text(tex, encoding="utf-8")


def export_run(run_dir: Path, manifest_path: Path, reference_path: Path = DEFAULT_REFERENCE, output_dir: Path | None = None, dataset_status_path: Path | None = None) -> dict:
    manifest = read_csv(manifest_path)
    predictions_path = run_dir / "predictions.csv"
    predictions, incomplete_tail = read_prediction_snapshot(predictions_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    dataset_status_path = dataset_status_path or manifest_path.parent.parent / "dataset_status.json"
    dataset_status = json.loads(dataset_status_path.read_text(encoding="utf-8-sig")) if dataset_status_path.is_file() else {}
    report = build_report(manifest, predictions, reference, dataset_status)
    report["input_provenance"] = dict(manifest=str(manifest_path.resolve()), manifest_sha256=file_hash(manifest_path),
                                      predictions=str(predictions_path.resolve()), prediction_snapshot_rows=len(predictions),
                                      final_unterminated_record_ignored=incomplete_tail,
                                      reference=str(reference_path.resolve()), reference_sha256=file_hash(reference_path))
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        report["run_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    export_files(report, output_dir or run_dir / "tables")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='export tables')
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-status", type=Path)
    args = parser.parse_args()
    report = export_run(args.run_dir, args.manifest, args.reference, args.output_dir, args.dataset_status)
    print(json.dumps({k: report[k] for k in ("run_status", "selected_count", "successful_count", "failed_count", "measured_complete_pairs", "macro_auc_available_finished_pairs")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
