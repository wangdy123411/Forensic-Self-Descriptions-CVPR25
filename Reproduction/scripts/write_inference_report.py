from project_paths import PROJECT_ROOT

import json
from pathlib import Path

project = PROJECT_ROOT
results = project/'Results'/'tiny_genimage_smoke_200'
summary = json.loads((results/'summary.json').read_text(encoding='utf-8'))
verification = json.loads((results/'verification.json').read_text(encoding='utf-8'))
assert verification['status'] == 'passed'
m = summary['metrics']
cm = m['confusion_matrix']
report = f'''FSD 推理流程验证结果
===================

完成图片：{summary['successful_images']} / {summary['selected_images']}
失败图片：{summary['failed_images']}
真实图：100 张；AI 生成图：100 张
默认阈值：z_score < {summary['threshold']} 判为生成图

准确率 Accuracy：{m['accuracy']:.2%}
ROC AUC：{m['roc_auc']:.4f}
生成图 Precision：{m['precision_fake']:.2%}
生成图 Recall：{m['recall_fake']:.2%}
生成图 F1：{m['f1_fake']:.4f}

混淆矩阵（行是真实标签，列是预测标签，顺序均为 real、fake）：
真实图：预测真实 {cm[0][0]}，误判生成 {cm[0][1]}
生成图：漏判真实 {cm[1][0]}，预测生成 {cm[1][1]}

推理耗时：{summary['inference_seconds']:.1f} 秒
单图平均耗时：{summary['mean_image_seconds']:.3f} 秒
GPU：{summary['gpu']['name']}
显存峰值（PyTorch allocated）：{summary['peak_gpu_allocated_mib']/1024:.2f} GiB
Python：{summary['python'].split()[0]}
PyTorch：{summary['packages']['torch']}
CUDA runtime：{summary['torch_cuda_version']}
代码提交：{summary['code_commit']}
预训练权重：作者 v1.2.0 检测权重

运行说明
--------
直接调用作者源码中的 FSDDetector，保留原始 float64、1024 尺寸和默认阈值，
未训练、微调或重新校准模型。只运行真假检测，未运行生成器来源归因。
为复用服务器已有 GPU 软件，本次使用 Python 3.10 / PyTorch 2.8 环境，
低于 pyproject.toml 声明的 Python >=3.12 / PyTorch >=2.10。
本次已完成全部图片推理及独立结果校验；这不等同于作者原始环境的复现。

数据来自 Tiny-GenImage 社区版本验证集的便利抽样，每类100张，
只用于走通流程，不能将本结果直接与论文的完整测试结果比较。
ROC AUC 以 fake=1 为正类，使用 -z_score；分数不是概率。

文件
----
predictions.csv：逐图路径、真实标签、预测标签、z_score、耗时、显存及状态。
summary.json：指标、各生成器统计、完整配置、软件版本和权重校验值。
run_info.json：开始运行时的配置记录。
verification.json：本地独立校验结果，包含10000次真假分数配对计算的AUC。

实验目录：{project}

复跑命令（选择一个尚不存在的结果目录）：
python Reproduction/run.py script scripts/run_fsd_eval.py --device cuda --threads 8 --output-dir Results/new_run

来源：https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25
数据：https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
'''
(results/'README_results.txt').write_text(report,encoding='utf-8')
print(report)
