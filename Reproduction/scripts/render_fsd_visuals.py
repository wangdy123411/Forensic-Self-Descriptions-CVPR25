from __future__ import annotations

from project_paths import PROJECT_ROOT


import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = PROJECT_ROOT
NAVY = "#142C44"
MUTED = "#5F7185"
REAL = "#168B83"
FAKE = "#D75A4A"
AMBER = "#BA7B16"
BLUE = "#4069B1"
BG = "#F5F8FC"
LINE = "#DFE7F0"


def configure_style():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font_name, "font.size": 12, "axes.unicode_minus": False,
        "axes.edgecolor": LINE, "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": BG, "axes.facecolor": "white", "savefig.facecolor": BG,
        "legend.frameon": False, "grid.color": LINE, "grid.alpha": .7,
    })


class Report:
    def __init__(self, root, output=None, intermediates=None):
        self.root = root
        self.results = root / "Results/tiny_genimage_smoke_200"
        self.data = root / "Data/tiny_genimage_smoke_200"
        self.output = output or self.results / "visualizations"
        self.output.mkdir(parents=True, exist_ok=True)
        source = intermediates or self.results / "visualizations/intermediates"
        self.meta = json.loads((source / "export_metadata.json").read_text(encoding="utf-8"))
        self.arrays = np.load(source / "intermediates.npz", allow_pickle=False)
        self.cases = self.meta["cases"]
        assert len(self.cases) == 3
        self.summary = json.loads((self.results / "summary.json").read_text(encoding="utf-8"))
        with (self.results / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        assert len(self.rows) == 200 and all(r["status"] == "ok" for r in self.rows)
        self.z = np.array([float(r["z_score"]) for r in self.rows])
        self.y = np.array([int(r["label"]) for r in self.rows])
        self.threshold = float(self.summary["threshold"])
        self.entries = []

    def array(self, case, key):
        return self.arrays[f"{case['id']}_{key}"]

    def image(self, case):
        with Image.open(self.data / case["relative_path"]) as image:
            return image.convert("RGB").copy()

    @staticmethod
    def role(case):
        return {"real_correct": "正确识别的真实图", "fake_correct": "正确识别的生成图", "false_positive": "唯一误判的真实图"}[case["role"]]

    @staticmethod
    def letter(case):
        return {"c0": "A", "c1": "B", "c2": "C"}[case["id"]]

    @staticmethod
    def color(case):
        return AMBER if case["role"] == "false_positive" else REAL if int(case["label"]) == 0 else FAKE

    def title_case(self, case):
        return f"{self.letter(case)}  {self.role(case)}"

    def figure(self, step, title, subtitle, size=(16, 9)):
        fig = plt.figure(figsize=size)
        fig.text(.045, .946, f"{step}  /  {title}", fontsize=25, weight="bold", va="top", color=NAVY)
        fig.text(.045, .889, subtitle, fontsize=12.3, color=MUTED, va="top")
        fig.text(.045, .027, "FSD · CVPR 2025  |  Tiny-GenImage 200 张流程验证  |  实际模型计算与已核验预测结果", fontsize=9.5, color=MUTED)
        return fig

    def save(self, fig, filename, title, caption):
        path = self.output / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        with Image.open(path) as image:
            size = list(image.size)
        entry = {"file": filename, "title": title, "caption": caption, "pixels": size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.entries.append(entry)
        print("RENDERED", filename, size, flush=True)

    @staticmethod
    def clean_image_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def limit(values, percentile=99):
        limit = float(np.percentile(np.abs(values), percentile))
        return max(limit, 1e-12)

    def overview(self):
        case = self.cases[1]
        fig = self.figure("00", "一次推理，怎样从图片得到真假判断？", "流程顺序为示意；卡片内的图片、特征、打分与统计均来自本次实际运行。", (16, 10))
        grid = fig.add_gridspec(2, 4, left=.045, right=.96, top=.82, bottom=.23, hspace=.50, wspace=.28)
        titles = ["01 输入图片", "02 灰度化", "03 提取残差", "04 三尺度分析", "05 960维 FSD", "06 特征变换", "07 GMM打分", "08 阈值判断"]
        subtitles = ["保留原始图像", "转为 L，像素值0–255", "8个预测误差通道", "1024 / 512 / 256", "共同拟合，8×120", "学习到的残差MLP", "5个混合分量", "z < −2 判为生成图"]
        for index in range(8):
            ax = fig.add_subplot(grid[index // 4, index % 4])
            ax.set_title(titles[index], fontsize=14, loc="left", pad=9)
            if index == 0:
                ax.imshow(self.image(case)); self.clean_image_axis(ax)
            elif index == 1:
                ax.imshow(self.array(case, "gray"), cmap="gray", vmin=0, vmax=255); self.clean_image_axis(ax)
            elif index == 2:
                residual = self.array(case, "residual")[0]
                v = self.limit(residual)
                ax.imshow(residual, cmap="RdBu_r", vmin=-v, vmax=v); self.clean_image_axis(ax)
            elif index == 3:
                for scale in range(3):
                    inset = ax.inset_axes([scale * .335, .1, .31, .80])
                    arr = self.array(case, f"scale{scale}")[0]
                    v = self.limit(arr)
                    inset.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v)
                    self.clean_image_axis(inset)
                    inset.set_title(str(1024 // 2**scale), fontsize=10)
                ax.axis("off")
            elif index == 4:
                arr = self.array(case, "fsd").reshape(8, 120)
                v = self.limit(arr)
                ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
                ax.set_xlabel("系数索引", fontsize=10); ax.set_ylabel("通道", fontsize=10)
            elif index == 5:
                arr = self.array(case, "transform_steps")
                v = self.limit(arr)
                ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
                ax.set_xlabel("特征索引", fontsize=10); ax.set_ylabel("变换阶段", fontsize=10)
            elif index == 6:
                vals = self.array(case, "component_log_prob")
                ax.bar(np.arange(1, 6), vals, color=BLUE, width=.65)
                ax.set_xticks(np.arange(1, 6)); ax.set_xlabel("分量编号（不是类别）", fontsize=10)
                ax.tick_params(labelsize=9)
            else:
                ax.axis("off")
                ax.text(.5, .68, f"z = {case['z_score']:.3f}", ha="center", fontsize=23, weight="bold", color=FAKE)
                ax.text(.5, .39, "预测：AI生成", ha="center", fontsize=22, color=FAKE)
                ax.text(.5, .17, "低于默认阈值 −2", ha="center", fontsize=12, color=MUTED)
            ax.text(.0, -.24, subtitles[index], transform=ax.transAxes, fontsize=10.5, color=MUTED)
        fig.text(.045, .093, "全量结果：200/200 成功  ·  Accuracy 99.5%  ·  ROC AUC 1.000  ·  1张真实图误判", fontsize=17, weight="bold", color=NAVY)
        self.save(fig, "00_pipeline_overview.png", "流程总览", "从输入到真假判断的8个阶段。图中示例为生成图B；底部指标来自全部200张。残差和特征图只作展示，不是伪造区域定位。")

    def inputs(self):
        fig = self.figure("01", "输入图片：贯穿全流程的三个实际案例", "A、B分别取正确分类样本中z分数的中间位置；C为全部200张中的唯一误判。")
        grid = fig.add_gridspec(1, 3, left=.045, right=.955, top=.80, bottom=.23, wspace=.14)
        for ax, case in zip(grid.subplots(), self.cases):
            ax.imshow(self.image(case)); self.clean_image_axis(ax)
            ax.set_title(self.title_case(case), fontsize=15, color=self.color(case), pad=13)
            w, h = case["original_size"]
            truth = "真实照片" if int(case["label"]) == 0 else f"AI生成 · {case['generator']}"
            ax.text(.5, -.11, f"{truth}  |  {w} × {h} 像素", transform=ax.transAxes, ha="center", fontsize=12)
            ax.text(.5, -.20, Path(case["relative_path"]).name, transform=ax.transAxes, ha="center", fontsize=10.5, color=MUTED)
        fig.text(.045, .09, "输入文件保留下载时的字节；此处仅缩略显示。后续各步均追踪这三张图。", fontsize=13, color=MUTED)
        self.save(fig, "01_input_images.png", "输入图片", "三个案例均来自本次验证集；选择规则已明确，不把单个案例当作统计结论。显示图片不会改写源文件。")

    def grayscale(self):
        fig = self.figure("02", "灰度化：模型使用亮度结构", "实际处理：PIL Image.convert('L')，随后转为float64张量；此步不改变图像尺寸。", (16, 10))
        grid = fig.add_gridspec(2, 3, left=.065, right=.955, top=.81, bottom=.12, hspace=.30, wspace=.15)
        for j, case in enumerate(self.cases):
            ax = fig.add_subplot(grid[0, j]); ax.imshow(self.image(case)); self.clean_image_axis(ax)
            ax.set_title(self.title_case(case), fontsize=13, color=self.color(case))
            ax = fig.add_subplot(grid[1, j]); ax.imshow(self.array(case, "gray"), cmap="gray", vmin=0, vmax=255); self.clean_image_axis(ax)
            ax.set_title("实际灰度输入  /  0–255", fontsize=12)
        fig.text(.022, .66, "原\n图", fontsize=14, color=MUTED, va="center")
        fig.text(.022, .28, "灰\n度", fontsize=14, color=MUTED, va="center")
        self.save(fig, "02_grayscale.png", "灰度化", "使用实际导出的L灰度数组。FRE在原尺寸灰度图上运行；尺寸统一发生在残差提取之后。")

    def filters(self):
        weights = self.arrays["filters"]
        fig = self.figure("03A", "FRE：8个学习得到的预测滤波器", "真实权重：15×15卷积核；中心系数为0，非中心系数和约为1。残差 = 灰度图 − 邻域预测。")
        grid = fig.add_gridspec(2, 4, left=.065, right=.91, top=.81, bottom=.13, hspace=.26, wspace=.28)
        v = self.limit(weights, 100)
        for k, ax in enumerate(grid.subplots().flat):
            im = ax.imshow(weights[k], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.add_patch(Rectangle((6.5, 6.5), 1, 1, fill=False, edgecolor="#18283C", lw=1.6))
            ax.set_title(f"通道 {k+1}  |  Σw = {weights[k].sum():.4f}", fontsize=12)
            ax.set_xticks([0, 7, 14]); ax.set_yticks([0, 7, 14]); ax.tick_params(labelsize=9)
        cax = fig.add_axes([.935, .25, .014, .43]); fig.colorbar(im, cax=cax, label="实际卷积权重")
        self.save(fig, "03a_fre_filters.png", "FRE卷积核", "8个受约束卷积核来自预训练FRE权重。黑框标记中心位置。统一色标显示真实系数，未将它们解释为注意力或类别。")

    def residuals(self):
        fig = self.figure("03B", "预测误差：每张图片得到8张残差图", "残差去除边缘7像素后导出；红/蓝表示正/负误差。图像内容与预测器的差异形成后续特征。", (18, 10))
        grid = fig.add_gridspec(3, 8, left=.067, right=.93, top=.81, bottom=.14, hspace=.35, wspace=.10)
        v = self.limit(np.concatenate([self.array(case, "residual").ravel() for case in self.cases]))
        for i, case in enumerate(self.cases):
            arr = self.array(case, "residual")
            for k in range(8):
                ax = fig.add_subplot(grid[i, k])
                im = ax.imshow(arr[k], cmap="RdBu_r", vmin=-v, vmax=v)
                self.clean_image_axis(ax)
                ax.set_title(f"{self.letter(case)} · C{k+1}", fontsize=10.5, color=self.color(case))
                if k == 0:
                    ax.set_ylabel(f"样本{self.letter(case)}", fontsize=12)
            y = .14 + (2-i) * (.67/3) + .025
            cax = fig.add_axes([.946, y, .008, .145])
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8)
        fig.text(.067, .086, "展示增强：三个样本全部24张残差图共用±|残差|第99百分位色标，超出范围截色；增强仅用于显示。", fontsize=11, color=MUTED)
        self.save(fig, "03b_residual_channels.png", "8通道残差", "实际FRE残差，缩小用于展示，全部24张共用以0为中心的对称色标，按共同的99%分位截色。亮度不直接代表真假；这些图不是伪造概率或空间定位图。")

    def scales(self):
        fig = self.figure("04", "三尺度残差：全图结构与局部邻域", "残差先按短边缩放并中心裁成1024×1024，再构建1、1/2、1/4尺度；每个尺度均使用8个通道。", (16, 11))
        grid = fig.add_gridspec(3, 3, left=.08, right=.95, top=.81, bottom=.12, hspace=.33, wspace=.18)
        for i, case in enumerate(self.cases):
            all_scales = np.concatenate([self.array(case, f"scale{s}")[0].ravel() for s in range(3)])
            v = self.limit(all_scales)
            for scale in range(3):
                ax = fig.add_subplot(grid[i, scale])
                arr = self.array(case, f"scale{scale}")[0]
                ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v)
                self.clean_image_axis(ax)
                actual = case["scale_shapes"][scale]
                ax.set_title(f"{self.letter(case)} · 尺度{scale+1}  /  {actual[-1]}×{actual[-2]}", fontsize=11.5, color=self.color(case))
                patch = self.array(case, f"patch{scale}")[0]
                inset = ax.inset_axes([.69, .02, .29, .31])
                inset.imshow(patch, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                inset.set_xticks([]); inset.set_yticks([])
                for sp in inset.spines.values(): sp.set_visible(True); sp.set_color("#FFFFFF"); sp.set_linewidth(2)
                inset.set_title("中心11×11", fontsize=7.5, pad=2, color=NAVY, backgroundcolor="white")
        fig.text(.08, .07, "每格仅展示C1通道与真实中心邻域，整幅图为展示缩略。拟合使用全部通道、全图邻域和全部3个尺度。", fontsize=10.5, color=MUTED)
        self.save(fig, "04_multiscale_residuals.png", "三尺度与局部窗口", "实际尺度为1024、512、256；图中主图缩略显示。右下角是从各尺度真实取出的中心11×11窗口，不是缩略图的放大。每个样本跨尺度共用色标。")

    def descriptor(self):
        fig = self.figure("05", "把三尺度信息共同拟合为960维FSD", "11×11窗口去掉中心，共120个邻域系数；8个残差通道 × 120 = 960维。不是3×960拼接。", (16, 10))
        schematic = fig.add_axes([.055, .29, .22, .40])
        mask = np.ones((11, 11)); mask[5, 5] = 0
        schematic.imshow(mask, cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#B9D0EE"]), vmin=0, vmax=1)
        schematic.set_xticks(np.arange(-.5, 11, 1), minor=True); schematic.set_yticks(np.arange(-.5, 11, 1), minor=True)
        schematic.grid(which="minor", color="white", linewidth=1)
        schematic.set_xticks([]); schematic.set_yticks([])
        schematic.text(5, 5, "×", ha="center", va="center", fontsize=23, color=FAKE)
        schematic.set_title("邻域结构示意", fontsize=16, pad=13)
        schematic.text(.5, -.17, "中心是预测目标\n120个位置提供邻域信息", ha="center", transform=schematic.transAxes, fontsize=12, color=MUTED, linespacing=1.7)
        vectors = [self.array(c, "fsd").reshape(8, 120) for c in self.cases]
        v = self.limit(np.stack(vectors))
        for i, (case, array) in enumerate(zip(self.cases, vectors)):
            ax = fig.add_axes([.36, .64-i*.225, .55, .14])
            im = ax.imshow(array, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_title(self.title_case(case), loc="left", fontsize=12, color=self.color(case), pad=7)
            ax.set_yticks([0, 3, 7], ["C1", "C4", "C8"]); ax.tick_params(labelsize=10)
            ax.set_xticks([0, 39, 79, 119], ["1", "40", "80", "120"])
            if i == 2: ax.set_xlabel("每个通道的邻域系数序号（中心已排除）", fontsize=10)
        cax = fig.add_axes([.934, .22, .012, .52]); fig.colorbar(im, cax=cax, label="FSD系数（统一展示色标）")
        fig.text(.055, .085, "左侧为结构示意；右侧为实际拟合系数，统一按|系数|第99百分位截色。该热图不是原图空间，也不是伪造定位图。", fontsize=10.5, color=MUTED)
        self.save(fig, "05_fsd_descriptor.png", "960维FSD", "三尺度通过共同的约束最小二乘求解产生一个960维描述向量。右侧按8个通道×120个邻域系数排列，仅为特征可视化。")

    def transforms(self):
        count = int(self.meta.get("transform_count", self.array(self.cases[0], "transform_steps").shape[0]-1))
        fig = self.figure("06", "学习到的变换：仍是960维，数值发生变化", f"实际执行{count}个预训练特征变换：f(t+1) = f(t) + MLP(t)(f(t))。此步没有重新训练模型。", (16, 10))
        grid = fig.add_gridspec(3, 2, left=.075, right=.95, top=.80, bottom=.18, width_ratios=[3.7, 1], hspace=.55, wspace=.27)
        for i, case in enumerate(self.cases):
            steps = self.array(case, "transform_steps")
            v = self.limit(steps)
            ax = fig.add_subplot(grid[i, 0]); ax.imshow(steps, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_title(self.title_case(case), fontsize=12, loc="left", color=self.color(case), pad=8)
            ticks = list(range(len(steps))) if len(steps) <= 6 else [0, len(steps)//2, len(steps)-1]
            ax.set_yticks(ticks, ["输入FSD" if t == 0 else f"变换{t}" for t in ticks]); ax.tick_params(labelsize=10)
            ax.set_xticks([0, 239, 479, 719, 959], [1, 240, 480, 720, 960])
            if i == 2: ax.set_xlabel("特征维度索引（非像素坐标）", fontsize=10)
            ax = fig.add_subplot(grid[i, 1]); norms = np.linalg.norm(steps, axis=1)
            ax.plot(np.arange(len(steps)), (norms-norms[0])*1e4, "o-", color=self.color(case), lw=2, ms=4)
            ax.set_title("范数变化 / ×10^-4", fontsize=11)
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.text(.98, .05, f"初始范数 {norms[0]:.4f}", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
            ax.set_xticks(ticks); ax.tick_params(labelsize=9); ax.grid(axis="y")
            if i == 2: ax.set_xlabel("变换阶段", fontsize=10)
        fig.text(.075, .078, "热图按样本增强对比度（|值|第99百分位截色）；右侧为相对输入的范数差，以10^-4为单位。特征和范数均不是概率。", fontsize=10.5, color=MUTED)
        self.save(fig, "06_feature_transforms.png", "学习到的特征变换", "每一行是一个实际特征向量：初始FSD及20个预训练变换的输出。右侧显示相对输入的范数变化（按10^-4单位），并标出初始范数；向量始终为960维。")

    def scoring(self):
        scoring = self.meta.get("scoring", self.summary["model_config"]["scoring"])
        mu, sigma = scoring["train_mean"], scoring["train_std"]
        fig = self.figure("07", "GMM打分，再换算为z-score", "5个混合分量共同描述特征分布。分量编号不是生成器类别，也不是“真假概率”。", (16, 10))
        grid = fig.add_gridspec(3, 2, left=.075, right=.95, top=.80, bottom=.14, width_ratios=[1.05, 1.35], hspace=.70, wspace=.35)
        for i, case in enumerate(self.cases):
            ax = fig.add_subplot(grid[i, 0])
            vals = self.array(case, "component_log_prob")
            ax.bar(np.arange(1, 6), vals, color=self.color(case), width=.62)
            ax.set_title(f"{self.letter(case)}  加权分量log值  /  raw={case['raw_score']:.2f}", fontsize=11, loc="left")
            ax.set_xticks(np.arange(1, 6), [f"分量{k}" for k in range(1, 6)]); ax.tick_params(labelsize=9)
            ax.grid(axis="y"); ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax = fig.add_subplot(grid[:, 1])
        case_z = [float(c["z_score"]) for c in self.cases]
        lo, hi = min(case_z)-1.8, max(1.2, max(case_z)+.8)
        ax.axvspan(lo, self.threshold, color=FAKE, alpha=.08)
        ax.axvspan(self.threshold, hi, color=REAL, alpha=.07)
        ax.axvline(self.threshold, color=FAKE, lw=1.7, ls="--")
        for i, case in enumerate(self.cases):
            y = 2-i
            ax.scatter([case["z_score"]], [y], s=155, c=self.color(case), edgecolors="white", linewidths=1.5, zorder=4)
            ax.text(lo+.25, y+.23, f"{self.letter(case)}  z={case['z_score']:.3f}", fontsize=12, weight="bold", color=self.color(case))
        ax.set_yticks([2, 1, 0], ["A · 真实图", "B · 生成图", "C · 误判真图"])
        ax.set_xlim(lo, hi); ax.set_ylim(-.55, 2.65)
        ax.set_xlabel("z-score：越负，越倾向判为生成图", fontsize=11)
        ax.set_title("默认阈值 −2：左侧判生成，右侧判真实", fontsize=12, loc="left", pad=18)
        ax.grid(axis="x"); ax.tick_params(labelsize=11)
        fig.text(.075, .086, f"raw = logsumexp(5个加权分量log值)    →    z = (raw − μ) / σ    |    μ={mu:.3f}，σ={sigma:.3f}", fontsize=12, color=MUTED)
        self.save(fig, "07_gmm_and_threshold.png", "GMM与默认阈值", "左侧为实际log(πk·Nk)；用logsumexp汇总后，按预训练的训练均值和标准差换算z。右侧是三个实际分数和默认阈值，不是概率图。")

    def predictions(self):
        fig = self.figure("08", "最终预测：把分数与默认阈值比较", "判断规则只有一条：z < −2 → AI生成；否则 → 真实。真实标签来自数据集manifest。", (16, 10))
        for j, case in enumerate(self.cases):
            x = .045 + j*.315
            card = FancyBboxPatch((x, .12), .29, .69, boxstyle="round,pad=.012,rounding_size=.012", transform=fig.transFigure, facecolor="white", edgecolor=LINE, zorder=-1)
            fig.patches.append(card)
            ax = fig.add_axes([x+.012, .40, .266, .35]); ax.imshow(self.image(case)); self.clean_image_axis(ax)
            truth = "真实" if int(case["label"]) == 0 else "AI生成"
            fake = float(case["z_score"]) < self.threshold
            pred = "AI生成" if fake else "真实"
            correct = int(fake) == int(case["label"])
            color = self.color(case)
            fig.text(x+.015, .775, f"{self.letter(case)}  {'识别正确' if correct else '误判：假阳性'}", color=color, fontsize=16, weight="bold")
            fig.text(x+.015, .35, f"数据集标签：{truth}", fontsize=13, color=MUTED)
            fig.text(x+.015, .29, f"模型预测：{pred}", fontsize=18, weight="bold", color=FAKE if fake else REAL)
            fig.text(x+.015, .23, f"z = {case['z_score']:.4f}", fontsize=16)
            fig.text(x+.015, .17, "z < −2" if fake else "z ≥ −2", fontsize=12, color=MUTED)
        self.save(fig, "08_prediction_cards.png", "最终预测图卡", "展示相同三个案例的标签、预测和分数。误判图C是数据集标为真实、模型判为生成的假阳性，不能因此改写数据集标签。")

    def evaluation(self):
        m = self.summary["metrics"]
        fig = self.figure("09", "200张图片的整体结果", "200/200成功  ·  Accuracy 99.5%  ·  ROC AUC 1.000  ·  生成图Recall 100%  ·  默认阈值保持−2", (16, 11))
        grid = fig.add_gridspec(2, 2, left=.075, right=.95, top=.80, bottom=.17, hspace=.38, wspace=.28)
        ax = fig.add_subplot(grid[0, 0]); cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=29, weight="bold", color="white" if cm[i,j] > 50 else NAVY)
        ax.set_xticks([0, 1], ["预测真实", "预测生成"]); ax.set_yticks([0, 1], ["实际真实", "实际生成"])
        ax.set_title("混淆矩阵  /  仅1张真实图误判", fontsize=14, loc="left", pad=12)
        ax = fig.add_subplot(grid[0, 1])
        scores = -self.z
        thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
        fpr = [float(np.mean(scores[self.y == 0] >= t)) for t in thresholds]
        tpr = [float(np.mean(scores[self.y == 1] >= t)) for t in thresholds]
        ax.step(fpr, tpr, where="post", color=BLUE, lw=2.4, label=f"AUC = {m['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#A3B2C1", lw=1)
        ax.scatter(cm[0,1]/100, cm[1,1]/100, color=FAKE, s=65, zorder=5, label="默认阈值 −2")
        ax.set(xlim=(-.035, 1.02), ylim=(-.035, 1.06), xlabel="真实图误报率 FPR", ylabel="生成图检出率 TPR")
        ax.set_title("ROC  /  正类=fake，输入分数=−z", fontsize=14, loc="left", pad=12)
        ax.legend(loc="lower right", fontsize=11); ax.grid()
        ax = fig.add_subplot(grid[1, 0]); bins = np.linspace(self.z.min()-1, self.z.max()+1, 35)
        ax.hist(self.z[self.y == 0], bins=bins, color=REAL, alpha=.7, label="真实图 n=100")
        ax.hist(self.z[self.y == 1], bins=bins, color=FAKE, alpha=.65, label="生成图 n=100")
        ax.axvline(self.threshold, color=NAVY, lw=1.7, ls="--", label="阈值 −2")
        ax.set(xlabel="实际z-score", ylabel="图片数")
        ax.set_title("分数分布  /  共用相同分箱", fontsize=14, loc="left", pad=12)
        ax.legend(fontsize=10); ax.grid(axis="y")
        ax = fig.add_subplot(grid[1, 1]); per = self.summary["per_generator"]
        names = list(per); values = [100*per[n]["recall_fake"] for n in names]
        bars = ax.bar(np.arange(len(names)), values, color=FAKE, alpha=.87, width=.62)
        for bar, name in zip(bars, names):
            rec = per[name]
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f"{rec['detected_fake']}/{rec['selected']}", ha="center", fontsize=11)
        ax.set_xticks(np.arange(len(names)), ["SD1.5" if n == "SD15" else n for n in names], rotation=15)
        ax.tick_params(axis="x", labelsize=10); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("生成图检出率 / %")
        ax.set_title("按已知来源分组统计  /  非来源归因任务", fontsize=13, loc="left", pad=12)
        ax.grid(axis="y")
        fig.text(.075, .07, "这是每类100张的便利抽样，用于验证流程；不能把这些指标等同于论文完整测试集表现。", fontsize=11.5, color=MUTED)
        self.save(fig, "09_dataset_evaluation.png", "全量评估", "基于全部200条已核验预测绘制混淆矩阵、ROC、分数分布和按数据集生成器标签分组的检出率。每组样本量直接标在柱顶。")

    def error_analysis(self):
        case = next(c for c in self.cases if c["role"] == "false_positive")
        real_z, fake_z = self.z[self.y == 0], self.z[self.y == 1]
        fig = self.figure("10", "为什么AUC为1，但准确率不是100%？", "AUC衡量分数排序；准确率还取决于所用阈值。本次保留预训练默认阈值，没有用测试标签重新调阈值。", (16, 9))
        ax = fig.add_axes([.055, .25, .32, .49]); ax.imshow(self.image(case)); self.clean_image_axis(ax)
        ax.set_title(f"唯一误判：{Path(case['relative_path']).name}", fontsize=12, color=AMBER)
        fig.text(.055, .15, f"数据标签：真实  →  模型预测：生成\nz = {case['z_score']:.4f} < −2", fontsize=15, color=AMBER, linespacing=1.7)
        ax = fig.add_axes([.49, .24, .45, .52])
        lo = min(-4.4, fake_z.max()-.8); hi = 1.0
        ax.axvspan(lo, self.threshold, color=FAKE, alpha=.08); ax.axvspan(self.threshold, hi, color=REAL, alpha=.07)
        rng = np.random.default_rng(12)
        regular_real_z = np.array([float(row["z_score"]) for row in self.rows
                                   if int(row["label"]) == 0 and row["relative_path"] != case["relative_path"]])
        for values, y, color in [(regular_real_z, 1, REAL), (fake_z, 0, FAKE)]:
            visible = values[(values >= lo) & (values <= hi)]
            jitter = rng.uniform(-.12, .12, len(visible))
            ax.scatter(visible, np.full(len(visible), y)+jitter, s=28, alpha=.72, color=color)
        ax.axvline(self.threshold, color=NAVY, ls="--", lw=1.7)
        ax.scatter([case["z_score"]], [1], marker="X", s=150, color=AMBER, edgecolors="white", zorder=7)
        ax.annotate(f"真图最低分 {real_z.min():.3f}", xy=(real_z.min(), 1), xytext=(lo+.1, 1.45), fontsize=11, color=AMBER, arrowprops={"arrowstyle":"->","color":AMBER})
        ax.annotate(f"生成图最高分 {fake_z.max():.3f}", xy=(fake_z.max(), 0), xytext=(lo+.1, -.50), fontsize=11, color=FAKE, arrowprops={"arrowstyle":"->","color":FAKE})
        ax.set(xlim=(lo, hi), ylim=(-.75, 1.75), xlabel="阈值附近的z-score（局部放大）")
        ax.set_yticks([0, 1], ["生成图", "真实图"])
        ax.set_title("全部生成图分数仍低于全部真实图", fontsize=15, loc="left", pad=14)
        ax.grid(axis="x")
        fig.text(.49, .12, f"局部图未画出更低分的{int(np.sum(fake_z < lo))}张生成图。\n仅凭这张照片无法确定误判原因；此图展示的是可验证的分数关系。", fontsize=11, color=MUTED, linespacing=1.6)
        self.save(fig, "10_false_positive_analysis.png", "唯一误判与AUC解释", "排序完全分离，所以AUC=1；默认阈值−2仍把最低分的那张真实图判成生成图。局部散点明确报告未显示的低分生成图数量。没有凭视觉猜测误判原因。")

    def index(self):
        entries = self.entries
        cards = []
        for entry in entries:
            title, caption, filename = (html.escape(entry[x]) for x in ("title", "caption", "file"))
            cards.append(f'<section id="{Path(filename).stem}"><h2>{title}</h2><p>{caption}</p><a href="{filename}" target="_blank"><img src="{filename}" alt="{title}" loading="lazy"></a><div class="meta">{entry["pixels"][0]} × {entry["pixels"][1]} PNG · 点击图片查看原尺寸</div></section>')
        nav = "".join(f'<a href="#{Path(e["file"]).stem}">{html.escape(e["title"])}</a>' for e in entries)
        doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FSD 推理逐步图解</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f8fc;color:#142c44;font-family:"Microsoft YaHei",system-ui,sans-serif;line-height:1.75}main{max-width:1280px;margin:0 auto;padding:44px 28px 70px}h1{font-size:34px;line-height:1.3;margin:8px 0 18px}h2{font-size:23px;line-height:1.4;margin:0 0 12px}.eyebrow{color:#168b83;font-weight:700;letter-spacing:2px}.lead{font-size:17px;max-width:1000px}.metrics{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0}.metric{background:white;border:1px solid #dfe7f0;border-radius:14px;padding:15px 22px;min-width:180px}.metric strong{display:block;font-size:27px}.metric small{color:#5f7185}nav{display:flex;flex-wrap:wrap;gap:9px;margin:25px 0 35px}a{color:#245a98}nav a{background:#e7eef7;border-radius:20px;text-decoration:none;padding:5px 14px;font-size:14px}section{background:white;border:1px solid #dfe7f0;border-radius:16px;margin:24px 0;padding:26px;scroll-margin-top:18px}section p{margin:0 0 18px;color:#5f7185}img{display:block;width:100%;height:auto;border-radius:8px}.meta{font-size:12px;color:#708195;margin-top:10px}.note{border-left:4px solid #168b83;padding:16px 20px;background:#eaf4f2;margin:24px 0}footer{font-size:14px;color:#5f7185}code{background:#edf1f6;border-radius:4px;padding:2px 5px} @media(max-width:600px){main{padding:25px 12px}section{padding:16px}h1{font-size:27px}}
</style><main><div class="eyebrow">FSD · CVPR 2025 / 实际推理记录</div><h1>从一张图片，到一个真假判断</h1><p class="lead">用真实中间计算值串起每一步：输入、灰度化、FRE残差、多尺度邻域、960维描述向量、预训练变换、GMM打分和阈值判别。三个案例保持一致，包括唯一误判。</p><div class="metrics"><div class="metric"><strong>200 / 200</strong><small>图片全部运行成功</small></div><div class="metric"><strong>99.5%</strong><small>准确率 · 默认阈值 −2</small></div><div class="metric"><strong>1.000</strong><small>ROC AUC · fake 为正类</small></div><div class="metric"><strong>1 张</strong><small>真实图被误判为生成图</small></div></div><div class="note">本页是200张便利抽样的流程验证，不是论文完整测试集复现。残差图只为显示做缩略和对比度增强，原始计算保持float64；特征热图不是空间定位图，z-score不是概率，GMM分量不是生成器类别。</div><nav>''' + nav + '</nav>' + ''.join(cards) + '''<section><h2>数据与复核</h2><p>逐图结果和模型配置来自上一次完成的200张推理。中间值在同一服务器环境、同一权重与配置下重新导出；具体分数核对误差、缩略方式和数组说明见元数据。</p><p><a href="../predictions.csv">逐图CSV</a> · <a href="../summary.json">运行汇总JSON</a> · <a href="../verification.json">独立复核记录</a> · <a href="intermediates/export_metadata.json">中间值导出说明</a> · <a href="visualization_manifest.json">图片清单与SHA-256</a></p><p><a href="https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25">作者代码仓库</a> · <a href="https://huggingface.co/datasets/TheKernel01/Tiny-GenImage">数据来源</a></p></section><footer>服务器实际运行环境：Python 3.10 / PyTorch 2.8 + CUDA 12.8；版本低于仓库声明的最低要求，但本次200张运行及独立结果复核均通过。模型源码和预处理未改动，未微调、重训或调整阈值。</footer></main></html>'''
        (self.output / "index.html").write_text(doc, encoding="utf-8")
        manifest = {"title":"FSD真实推理逐步图解", "source_predictions":"../predictions.csv", "intermediate_metadata":"intermediates/export_metadata.json", "figures":entries,
                    "method":"Matplotlib plots from actual exported arrays and validated predictions; figures explicitly distinguish schematic structure and display-only scaling."}
        (self.output / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description='render fsd visuals')
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--intermediates", type=Path)
    args = parser.parse_args()
    configure_style()
    report = Report(args.root, args.output, args.intermediates)
    for method in (report.overview, report.inputs, report.grayscale, report.filters, report.residuals,
                   report.scales, report.descriptor, report.transforms, report.scoring,
                   report.predictions, report.evaluation, report.error_analysis):
        method()
    report.index()
    print(json.dumps({"figures":len(report.entries), "output":str(report.output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
