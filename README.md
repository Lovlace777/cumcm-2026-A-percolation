# 微构体导电介质的几何渗流建模与低成本填充优化

全国大学生数学建模竞赛风格论文（赛题 A）。完整可编译 LaTeX 源码、图件与提交用 PDF。

## 下载论文 PDF

**发布页（推荐，点击即可下载）：**

https://github.com/Lovlace777/cumcm-2026-A-percolation/releases/download/v1.0/CUMCM_2026_A_percolation.pdf

**仓库内文件：**

https://github.com/Lovlace777/cumcm-2026-A-percolation/raw/main/CUMCM_2026_A_percolation.pdf

## 主要结论

| 问题 | 结论 |
| --- | --- |
| 1 | 组 1 不导通；组 2、组 3 导通 |
| 2 | $P(0.50\%,0.60\%,0.70\%,1.00\%)=10.4\%,22.8\%,52.0\%,100\%$ |
| 3 | $P_{\mathrm{on}}\ge 90\%$ 的最低填充量 $\varphi_A=0.85\%$（$N_A=601$） |
| 4 | 稳健方案：只填充 601 根介质 A，成本 8.92 元 |

## 仓库结构

```
CUMCM_2026_A_percolation.pdf   提交用 PDF（27 页）
main.tex                       论文源码
figures/                       正文全部图件（PDF）
src/                           几何核、并查集、Monte Carlo、出图脚本
results/                       第 1–4 问数值结果
```

编译需要 XeLaTeX / Tectonic，以及宋体、楷体、黑体。`main.tex` 默认从 `fonts/` 读取（本仓库不附带商用字体文件）。
