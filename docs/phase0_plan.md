# When Is More Vision Worth It? — Phase 0 实施计划

> **状态：待执行计划（TO-BE-EXECUTED）**。本文档内所有实验尚未运行，不含任何实测结果，也不对结果做正/负假设；所有数值阈值均为「预注册判定标准（pre-registered criteria）」。

> **并行进行中（由其他子代理负责，非本文产出）**：GitHub 仓库 AlanAkillove/more-vision-worth-it 初始化（SSH 推送）；conda 环境 deepminer（E:\conda\envs\deepminer）校验；FGVC-Aircraft 2013b tarball（raw/FGVC_Aircraft.tar.gz.00，sha256 30dbdbe5...4761e）从 ModelScope 下载解压至 FGVC_Aircraft\raw\fgvc-aircraft-2013b\。

---

## 0. 目的、边界与当前状态

### 0.1 研究问题与定位
- 核心问题：Can we predict the marginal value of additional visual evidence?（能否预测「额外视觉证据」的边际价值？）
- 场景设定：模型已观察一张低分辨率图像后，判断进一步获取更高分辨率视觉信息是否真的会纠正当前预测。
- 与既有做法的区别：不简单预测图像是否「困难」，也不仅依据 softmax confidence / entropy 决定是否增加计算量。
- 最终可能研究 tiny decision model，但当前阶段严格限定为 Phase 0 — Oracle Headroom / Core Phenomenon Audit。
- Phase 0 目标：先验证该研究问题是否实际存在；若 uncertainty 已足够、或额外视觉信息很少能纠正错误，应明确输出 NO-GO，而不是为让项目继续而修改实验。

### 0.2 当前状态（截至本计划编写时）
- 环境：conda env deepminer（E:\conda\envs\deepminer）校验中；GPU 为 NVIDIA RTX 4060 Laptop 8GB。
- 数据集：FGVC-Aircraft 2013b 正在从 ModelScope 下载解压至 FGVC_Aircraft\raw\fgvc-aircraft-2013b\。
- 仓库：GitHub AlanAkillove/more-vision-worth-it 正在初始化（SSH 推送）；当前工作区根目录尚无 .git；docs/ 目录由本计划创建。
- 本计划内所有实验：尚未执行。

### 0.3 硬件约束（全阶段适用）
NVIDIA RTX 4060 Laptop GPU；8GB VRAM；优先 PyTorch；frozen backbone；mixed precision inference；后续实验尽量使用离线 feature cache。必须保证代码在单张 8GB GPU 上运行。遇显存不足时优先降低 batch size，不得擅自改变研究设计。

### 0.4 执行原则
- 先验证现象，再训练模型；先证明 marginal visual value 与 uncertainty 不等价，再谈 decision model。
- 不为得到正结果修改实验。
- 先检查现有仓库与数据环境，再按最小必要修改逐步实现。
- 每完成一个重要阶段即运行验证，不要一次写完整项目后再测试。

## 1. 里程碑总览

| 里程碑 | 名称 | 主要产出 | 覆盖要求 |
| --- | --- | --- | --- |
| M0 | 环境与数据校验 | outputs/dataset_report.json/.md；env 记录 | §1 §2 §17 |
| M1 | Canonical preprocessing | configs/preprocessing.yaml；preproc hash | §3 |
| M2 | Feature extraction | cache/features/**；extraction metrics | §4 §5 |
| M3 | Shared classifier 与 outcomes | classifier artifact；outcomes_val.parquet | §6 §7 |
| M4 | Transition 分析与 Q1 | transition tables；Q1 verdict | §8 §9 §10 |
| M5 | Uncertainty / routing / oracle 与 Q2 | 曲线与表；Q2 verdict | §11 §12 §13 §14 |
| M6 | Predictability probe 与 Q3 | probe A/B 指标；Q3 verdict | §15 |
| M7 | Figures 与 phase0_report | 9 图 + tables；docs/phase0_report.md | §16 §18 §20 |

---

## 2. M0 — 环境与数据校验

### 2.1 目标
确认运行环境可复现；确认 FGVC-Aircraft 2013b 的数据版本、split 与标注完整；产出数据集报告。

### 2.2 输入
- FGVC_Aircraft\raw\fgvc-aircraft-2013b\（tarball 解压结果）
- configs/base.yaml

### 2.3 关键动作
1. 环境采集：Python / PyTorch / CUDA / cuDNN / GPU 名称与显存 / 关键包版本（torch, torchvision, numpy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, pillow, pyarrow）。
2. 数据集版本检查（逐项写入 report）：
   - 图像数量（image count）
   - annotation files 是否齐全：images_variant_{train,val,test}.txt、images_family_*.txt、images_manufacturer_*.txt、variants.txt、families.txt、manufacturers.txt；（可选）images_box.txt
   - variant 数量（K，即分类类别数）
   - train / val / test 各自数量
   - image path（图像目录与命名规则）
   - label mapping（variant 名称 → 连续索引）
3. 图像元信息：原始 width/height 分布；确认图像底部约 20px 的 copyright banner 存在（作为 M1 去除依据）。
4. 记录期望参考值：从官方 2013b 标注文件读出 train/val/test 计数、variant 数、family 数、manufacturer 数，并在 report 中标注其与数据描述文本（例如 10,200 张 / 102 类）之间的任何差异。

### 2.4 输出文件
- outputs/dataset_report.json
- outputs/dataset_report.md
- outputs/logs/env_report.json

### 2.5 关键模块与职责
- src/mvwi/env_report.py：采集并序列化环境版本。
- src/mvwi/data/fgvc_aircraft.py：解析标注文件、split、label mapping，返回样本清单。
- src/mvwi/data/dataset_report.py：汇总数据集统计并写 JSON/MD。
- src/mvwi/config.py：YAML 读取与合并。

### 2.6 验收标准
- 运行 `python scripts/00_env_check.py` → 打印 Python/torch/CUDA/GPU/包版本，并写 outputs/logs/env_report.json。
- 运行 `python scripts/01_dataset_report.py --config configs/base.yaml` → 打印各 split 计数与 variant 数，并写 outputs/dataset_report.json 与 outputs/dataset_report.md；两份报告含 2.3 全部字段。

### 2.7 风险与回退
- raw 尚未解压完成 → 等待下载/解压完成后再运行；脚本给出清晰错误信息。
- 标注文件命名差异 → 依据官方 2013b 目录结构做兼容读取。
- variant 名称含特殊字符（如 F/A-18、F-16A/B）→ 使用单调递增索引，同时保存原始名称字典。

## 3. M1 — Canonical preprocessing

### 3.1 目标
实现唯一 canonical pipeline，保证不同 resolution 观察的是同一视觉区域（same field of view）。

### 3.2 规则（严格）
统一 pipeline：原图 → remove bottom copyright banner（约 20px）→ pad to square while preserving the full image and aspect ratio → 得到单一 canonical image → 将该 SAME canonical image 独立 resize 到 112×112 / 224×224 / 448×448。
- interpolation：bicubic；antialias：开启（True）。
- normalization：DINOv2 官方 normalization（ImageNet mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]）。
- 所有 validation/test preprocessing 必须 deterministic。
- 参数写入 config，严禁散落硬编码。
- 生成 preprocessing hash/version，用于 feature cache 路径。
- 禁止：分别执行 Resize → CenterCrop 导致不同 resolution 的 field of view 不一致。
- 禁止：使用 ground-truth bounding box 作为主实验 preprocessing（避免提供额外的人工定位信息）。

### 3.3 可配置决策（写入 config，不改变研究设计）
- banner_remove_px：默认 20。
- pad_mode：square padding（保留完整图像与宽高比）；pad 填充值可配置。
- canonical 基准：定义 canonical image 的生成方式（如以去 banner 后 pad-to-square 结果作为 canonical，再各自下采样到目标尺寸），须保证 448 目标信息不被人为损失。
- target_sizes：[112, 224, 448]。

### 3.4 输出文件
- configs/preprocessing.yaml
- outputs/phase0/preproc_check/（少量可视化，可选，用于人工核验 FOV 一致）
- preproc hash 写入 outputs/phase0/metrics.json（preprocessing.preproc_hash）

### 3.5 关键模块与职责
- src/mvwi/preprocess/canonical.py：去 banner、pad to square、resize、normalize 的可复用函数。
- src/mvwi/preprocess/version.py：由 config 计算稳定的 preproc hash。

### 3.6 验收标准
- 运行 `python scripts/02_make_canonical.py --config configs/preprocessing.yaml --check` → 生成若干样本的 112/224/448 派生图并打印 preproc hash。
- 脚本/人工核验：三种 resolution 覆盖同一内容区域（如叠加对比 3 张派生图，确认无 CenterCrop 导致的 FOV 漂移）。

### 3.7 风险与回退
- banner 高度并非恰好 20px → 记录实际值并参数化，不阻断流程。
- 方形 pad 的填充边界影响结果 → 填充色写入 config 并固定。

---

## 4. M2 — Feature extraction

### 4.1 目标
对 112/224/448 分别离线提取 frozen DINOv2 global embeddings，仅 train/val；缓存可重复读取而无需再次运行 backbone。

### 4.2 Backbone 规格（严格）
- 主 backbone：DINOv2 ViT-S/14（官方 dinov2_vits14）。
- backbone completely frozen；不 fine-tune；eval mode；inference_mode；CUDA mixed precision。
- 提取 global image embedding；不缓存全部 patch tokens。
- patch size=14；embedding dim=384；112→8×8 patch grid、224→16×16、448→32×32。
- 这些 resolution 不得擅自修改（除非程序因技术原因无法运行）。448 OOM 时只减少 batch size。

### 4.3 范围
- 首先只使用 train/val。不要在 Phase 0 研究设计阶段反复查看 test result。

### 4.4 cache 字段（至少包含）
sample_id；split；ground-truth label；resolution；embedding；original width/height；preprocessing version；backbone identifier。
- 格式：.npz / .pt 或 memory-mappable numpy。
- 必须保证 cache 可重复读取而无需再次运行 backbone。
- 不得缓存 augmentation 后的随机版本。

### 4.5 输出指标（112/224/448 分别记录）
extraction runtime；peak GPU VRAM；images/sec；cache size；batch size。

### 4.6 输出文件
- cache/features/{preproc_hash}/dinov2_vits14/{res}/{split}.npz
- outputs/phase0/metrics.json（extraction 段，按 resolution 分列）

### 4.7 关键模块与职责
- src/mvwi/backbone/dinov2.py：加载 dinov2_vits14、eval()、inference_mode、AMP 前向，返回 global embedding。
- src/mvwi/features/extract.py：遍历 dataset、批处理、写缓存与 metrics。

### 4.8 验收标准
- 运行 `python scripts/03_extract_features.py --config configs/extract.yaml`（112/224/448）→ 产出三份缓存，并打印每 resolution 的 runtime / VRAM / images-per-sec / cache size / batch size。
- 再次运行加 `--cache-only` → 直接读取缓存、不触发 backbone 前向。

### 4.9 风险与回退
- 448 OOM → 仅降低 batch size（如 10 → 8 → 4 …），不改 resolution、不改研究设计。
- 下载/解压未完成 → 等待完成。
- 磁盘空间 → 预估三 resolution 缓存规模（约 N×384 维 float，多 GB 量级）并在 metrics 记录实际 cache size。

## 5. M3 — Shared classifier 与 outcomes

### 5.1 目标
训练一个 shared linear classifier h: R^384 → R^K（K = 实际 variant class 数），并生成 validation outcome 表。

### 5.2 Shared classifier 规则（严格）
- 不为三个 resolution 各训练一个独立 classifier。
- 训练数据为 train split 三个 resolution embeddings 的联合 {z_112, z_224, z_448}。
- 同一个 classifier head 必须处理三个视觉预算。
- backbone 始终 frozen。
- linear probe 可用 PyTorch linear layer 或 sklearn logistic regression；选更稳定、易复现的方法（默认 sklearn LogisticRegression，记录 solver / C / max_iter / 多分类设置）。
- 保存 classifier.pt 或等价 artifact。
- hyperparameters 必须写入 config 与实验日志。

### 5.3 Validation outcome 生成
- 仅在 validation split 上进行 Phase 0 hypothesis audit。
- 对每个 sample、每个 resolution 保存：ground truth y；predicted class；logits；probabilities；top-1 confidence；entropy；top1-top2 margin；energy score；correctness。
- 保存为统一 outcome table：cache/outcomes/{preproc_hash}/outcomes_val.parquet。
- 要求一行唯一对应 sample × resolution。

### 5.4 输出文件
- cache/classifier/{preproc_hash}/classifier.joblib（或 classifier.pt）
- cache/outcomes/{preproc_hash}/outcomes_val.parquet
- outputs/phase0/metrics.json（classifier 段：train/val accuracy per resolution 等）

### 5.5 关键模块与职责
- src/mvwi/models/linear_probe.py：拼接/洗牌联合特征、训练、保存 artifact、推理。
- src/mvwi/audit/outcomes.py：计算 per-sample 分数并写 outcome 表。
- src/mvwi/eval/metrics.py：accuracy、entropy、margin、energy 等计算。

### 5.6 验收标准
- 运行 `python scripts/04_train_classifier.py --config configs/classifier.yaml` → 保存 artifact，并把 hyperparameters 写入 metrics/日志。
- 运行 `python scripts/05_generate_outcomes.py --config configs/classifier.yaml` → 生成 parquet；校验 (sample_id, resolution) 唯一、每 resolution 行数 = val 样本数。

### 5.7 风险与回退
- 类别不平衡 → 记录 class distribution；可选 class_weight（写入 config）。
- 收敛/正则 → 记录并固定超参；不做大规模搜索。
- 防泄漏：不得在 val 上 fit 任何预处理器（如标准化器须仅在 train 上拟合）。

---

## 6. M4 — Transition 分析与 Q1

### 6.1 目标
分析三个 transition：112→224、224→448、112→448；主分析为 112→448；给出 Q1 预注册判定。

### 6.2 Transition 定义与分类（每个 transition，low → high）
- CC = low correct, high correct
- WC = low wrong, high correct（recoverable error）
- WW = low wrong, high wrong
- CW = low correct, high wrong（harmful escalation）

### 6.3 需要报告的统计（每个 transition）
- count（CC / WC / WW / CW 计数）
- fraction of all samples
- fraction conditional on low-resolution correctness / error
- 特别记录 recoverable error（WC）与 harmful escalation（CW）

### 6.4 Q1（主 transition = 112→448）
定义：r_rec = N(WC) / N；r_rec_error = N(WC) / N(low wrong)。
Q1 GO 建议标准（需同时满足）：
1. WC ≥ 150 validation samples
2. WC / N ≥ 4.5%
3. WC / N(low wrong) ≥ 15%
4. high-resolution improvement 有实际意义，例如 Acc_448 − Acc_112 ≥ 2 percentage points，或净修正 WC − CW 明显为正
- BORDERLINE：WC = 75–149；或 recoverable fraction = 2–4.5%
- NO-GO：WC < 75；或 fraction < 2%

### 6.5 额外约束
- 不要为了通过 gate 自动搜索大量 resolution combinations。
- 如果 112 accuracy 极高导致几乎没有 low-resolution errors：只允许记录这一事实，并建议未来单独测试 98 resolution；不要当前自动改变实验。

### 6.6 输出文件
- outputs/phase0/tables/transitions_112-224.csv、transitions_224-448.csv、transitions_112-448.csv
- outputs/phase0/metrics.json（q1: {r_rec, r_rec_error, WC, CW, Acc_112, Acc_448, verdict}）

### 6.7 关键模块与职责
- src/mvwi/audit/transitions.py：由 outcomes 表构建 transition 计数与比例。
- src/mvwi/audit/q1.py：应用 Q1 预注册阈值，输出 PASS/BORDERLINE/FAIL。

### 6.8 验收标准
- 运行 `python scripts/06_transition_analysis.py --config configs/analysis.yaml` → 打印三 transition 的 CC/WC/WW/CW 计数与各类比例，写 CSV，并把 Q1 verdict（PASS/BORDERLINE/FAIL）写入 metrics.json。

### 6.9 风险与回退
- 低分辨率错误极少（N(low wrong) 很小）→ 按 6.5 记录事实；分母为 0 时返回明确值而非报错。
- test split 不参与 → 仅 val。

## 7. M4 补充 — High-confidence recoverable errors

### 7.1 目标
分析 low wrong → high correct，但 low-resolution 下 max probability 很高的情形。

### 7.2 定义与报告
- 重点：low wrong → high correct，且 max probability at low resolution ≥ 0.8；并单独报告 ≥ 0.9。
- 统计：count；fraction among all recoverable errors；fraction among all samples。
- 核心问题：是否存在显著数量的 high-confidence recoverable errors？
- 建议强信号（预注册工程标准，非统计定理）：count ≥ 30 且占 recoverable errors ≥ 10%。

### 7.3 输出文件
- outputs/phase0/tables/high_conf_recoverable.csv
- outputs/phase0/metrics.json（high_conf 段）

### 7.4 关键模块与职责
- src/mvwi/audit/high_conf.py。

### 7.5 验收标准
- 运行 `python scripts/06_transition_analysis.py --config configs/analysis.yaml`（含 high-conf 段）→ 打印 count / fraction（≥0.8 与 ≥0.9 分列）并写 CSV。

### 7.6 风险与回退
- 阈值 0.8 / 0.9 及 count/fraction 判据必须写入 config，而非硬编码。

---

## 8. M5 — Uncertainty baselines / routing / oracle 与 Q2

### 8.1 Uncertainty baselines
对 112 resolution 计算：max softmax probability；entropy；top1-top2 margin；energy。
- 目标不是预测 current correctness，而是预测 WC = 1（otherwise = 0）。
- 报告：AUROC；AUPRC；ROC curve；precision-recall curve。
- positive class 可能不平衡，AUPRC 比 AUROC 更重要。
- 同时将样本按 confidence/entropy 分 bin，估计 P(WC | uncertainty bin) 并绘图。

### 8.2 Budget-matched routing
不要只比较 AUROC。对 high-resolution invocation rate = 20% / 40% / 60%，分别计算：如果根据每种 uncertainty score 将最「需要继续」的 top ρ samples 路由到 448，最终 accuracy 是多少？
记录：final accuracy；WC recall；unnecessary escalation rate；CW exposure；average visual cost proxy。
对每个 invocation rate 同时计算 oracle。

### 8.3 Sequential Oracle
- 遵守真实执行流程：所有样本首先承担 112 inference cost。
- oracle 已知每个样本在 112 和 448 下的真实 outcome。
- 在固定 escalation rate ρ 下，oracle 优先把额外视觉预算给予能够产生最大真实 utility improvement 的样本。
- 主要比较：best scalar uncertainty router vs oracle。
- 目标：衡量 dataset 上到底存在多少可以被学习策略利用的 headroom。
- 真实 sequential cost：C_policy = C_112 + I(continue) * C_448 + C_router（Phase 0 中 router cost 可视为近似零，但公式和代码接口应保留）。
- 另可计算 clairvoyant lower bound，但必须明确与 sequential oracle 分开。

### 8.4 Q2
问题：uncertainty 是否已经近似等价于 marginal visual value？比较 best scalar baseline 与 oracle。
- NO-GO for learned value router：如果在 20% / 40% / 60% 主要 operating points，oracle 和 best uncertainty accuracy gap 始终 < 0.5 percentage point，且 recoverable recall gap 很小。
- GO 信号：至少一个有意义 operating point：oracle accuracy − best scalar accuracy ≥ 0.75 percentage point，或 oracle recoverable recall − best scalar recoverable recall ≥ 10 percentage points。

### 8.5 输出文件
- outputs/phase0/tables/uncertainty_scores_val.csv
- outputs/phase0/tables/routing_curves.csv
- outputs/phase0/tables/oracle_curves.csv
- outputs/phase0/metrics.json（q2: {operating_points, gaps, verdict}）

### 8.6 关键模块与职责
- src/mvwi/audit/uncertainty.py：四类 scalar score、AUROC/AUPRC、分箱估计。
- src/mvwi/audit/routing.py：budget-matched routing（含 unnecessary escalation、CW exposure、visual cost proxy）。
- src/mvwi/audit/oracle.py：sequential oracle + clairvoyant lower bound（两者分开）。
- src/mvwi/audit/q2.py：Q2 预注册判定。

### 8.7 验收标准
- 运行 `python scripts/07_uncertainty_baselines.py --config configs/analysis.yaml` → 输出四 score 的 AUROC/AUPRC、ROC/PR 数据与分箱 P(WC|bin)。
- 运行 `python scripts/08_routing_oracle.py --config configs/analysis.yaml` → 在 20/40/60% 输出 routing 指标、oracle 指标与 Q2 verdict。

### 8.8 风险与回退
- 类别不平衡（WC 稀疏）→ 以 AUPRC 为主；bootstrap 估计波动。
- utility / cost 定义必须写入 config（util 定义为「由错到对」的改进量），不得隐式假设。

---

## 9. M6 — Phase-0 lightweight predictability probe 与 Q3

### 9.1 触发条件
只有 Q1 / Q2 有初步研究空间后，才运行这一部分。不要训练正式 neural decision model。

### 9.2 Probe 定义
- Probe A（Logistic Regression）：input = logits、confidence、entropy、margin、energy；target = WC。
- Probe B（Logistic Regression）：input = z_112、logits、confidence、entropy、margin、energy；target = WC。

### 9.3 数据与防泄漏
- training data 只能来自 development data，需避免数据泄漏。
- 如果需要从官方 training data 构建 router-training labels，则这些 labels 必须由 frozen classifier 对 train sample 的 out-of-sample prediction 或适当 split 产生，不能简单使用训练该 classifier 时同一数据上的过拟合预测。
- 如果第一轮实现过度复杂，请先在 validation 内使用明确的 temporary split 做 exploratory predictability audit，并在报告里标记其用途，不要把它当 final result。

### 9.4 报告指标
AUROC；AUPRC；recoverable recall at 20 / 40 / 60% escalation；routing regret；bootstrap confidence interval。

### 9.5 Q3
Q3 GO 信号（相比 best scalar baseline）：
- 至少两个 operating points 上 recoverable recall +5 pp；
- routing regret 平均下降 ≥ 10%；
- AUPRC improvement 大致 ≥ 0.03；
- 结果不是明显位于 bootstrap noise 内。
如果没有这些信号，应明确输出 NO-GO / weak predictability evidence。不要继续设计 Transformer。

### 9.6 输出文件
- outputs/phase0/tables/probe_metrics.csv、probe_curves.csv
- outputs/phase0/metrics.json（q3: {probe_a, probe_b, verdict}）

### 9.7 关键模块与职责
- src/mvwi/audit/probe.py：Probe A/B 训练、评估、bootstrap、routing regret。

### 9.8 验收标准
- 运行 `python scripts/09_predictability_probe.py --config configs/probe.yaml` → 输出 A/B 的 AUROC / AUPRC / recoverable recall@20/40/60% / routing regret / bootstrap CI 与 Q3 verdict。

### 9.9 风险与回退
- 数据泄漏 → 严格遵守 out-of-sample / split 策略；否则标记 exploratory。
- 样本量小 → bootstrap CI；不得据噪声下结论。

## 10. M7 — Figures 与 phase0_report

### 10.1 必做图（9 张，输出到 outputs/phase0/figures/）
1. Accuracy vs resolution
2. transition matrix / Sankey-like summary
3. WC / WW / CC / CW count
4. confidence distribution（recoverable / unrecoverable）
5. P(recoverable | confidence bin)
6. Precision–Recall curves for scalar baselines
7. Accuracy vs high-resolution invocation rate
8. Recoverable recall vs invocation rate
9. scalar baselines vs oracle

### 10.2 数据保全要求
所有 figure 的 underlying table 同时保存 CSV / Parquet。不允许只保存 PNG 而丢失原始数据。

### 10.3 报告（docs/phase0_report.md）必须包含的小节
Environment / Dataset validation / Preprocessing / Backbone and cache / Linear classifier / Accuracy by resolution / Transition analysis / High-confidence recoverable errors / Uncertainty baselines / Oracle headroom / Lightweight predictability probe / Runtime / VRAM / storage / GO / NO-GO。
最后必须明确给出：Q1: PASS/BORDERLINE/FAIL；Q2: PASS/BORDERLINE/FAIL；Q3: PASS/BORDERLINE/FAIL；OVERALL: GO 或 NO-GO 或 NEEDS ONE PREDEFINED FOLLOW-UP TEST。
不要因为已经写了代码就默认项目必须继续；如果结果是否定的，请忠实报告。

### 10.4 输出文件
- outputs/phase0/figures/*.png（9 张以上）+ 对应 outputs/phase0/tables/*.csv|.parquet
- docs/phase0_report.md

### 10.5 关键模块与职责
- src/mvwi/report/figures.py：读取 tables 生成 9 图。
- src/mvwi/report/report.py：汇总 metrics.json 生成报告骨架。

### 10.6 验收标准
- 运行 `python scripts/10_make_figures.py --config configs/analysis.yaml` → 生成 9 张图 + 对应 tables。
- 核验 docs/phase0_report.md 存在，且含 10.3 全部小节与四项 verdict 字段。

### 10.7 风险与回退
- matplotlib 中文字体缺失 → 使用英文标签或指定字体。
- 图与表数值不一致 → 图统一从 tables 读取。

---

## 附录 A — 建议 repo 结构

```
more-vision-worth-it/
├─ README.md
├─ configs/
│  ├─ base.yaml
│  ├─ preprocessing.yaml
│  ├─ extract.yaml
│  ├─ classifier.yaml
│  ├─ analysis.yaml
│  └─ probe.yaml
├─ src/mvwi/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ env_report.py
│  ├─ data/       {__init__.py, fgvc_aircraft.py, dataset_report.py}
│  ├─ preprocess/ {__init__.py, canonical.py, version.py}
│  ├─ backbone/   {__init__.py, dinov2.py}
│  ├─ features/   {__init__.py, extract.py}
│  ├─ models/     {__init__.py, linear_probe.py}
│  ├─ audit/      {__init__.py, outcomes.py, transitions.py, q1.py, high_conf.py,
│  │               uncertainty.py, routing.py, oracle.py, q2.py, probe.py}
│  ├─ eval/       {__init__.py, metrics.py}
│  └─ report/     {__init__.py, figures.py, report.py}
├─ scripts/       (见附录 C)
├─ cache/
│  ├─ features/{preproc_hash}/dinov2_vits14/{112,224,448}/{train,val}.npz
│  ├─ classifier/{preproc_hash}/classifier.joblib
│  └─ outcomes/{preproc_hash}/outcomes_val.parquet
├─ outputs/
│  ├─ dataset_report.json / dataset_report.md
│  ├─ logs/{env_report.json, run_*.log}
│  └─ phase0/{figures/, tables/, metrics.json, run_config.yaml}
└─ docs/
   ├─ phase0_plan.md   (本文件)
   └─ phase0_report.md (M7 产出)
```

## 附录 B — YAML config 字段草案

### B.1 configs/base.yaml
- project: {name, repo, seed}
- paths: {raw_dir, cache_dir, outputs_dir, docs_dir}
- device: {name: cuda, allow_tf32, amp_dtype: fp16|bf16}
- logging: {level, log_dir}
- git: {capture_commit: true, warn_if_dirty: true}

### B.2 configs/preprocessing.yaml
- banner_remove_px: 20
- pad_mode: square
- pad_fill: <int>
- canonical: {mode: pad_square_from_debannered, size: null|int}
- target_sizes: [112, 224, 448]
- interpolation: bicubic
- antialias: true
- normalize: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
- deterministic: true

### B.3 configs/extract.yaml
- backbone: {name: dinov2_vits14, source: torch_hub|local, checkpoint, frozen: true, eval: true}
- patch_size: 14 ; embed_dim: 384
- resolutions: [112, 224, 448]
- splits: [train, val]
- batch_size: {112: 64, 224: 32, 448: 8}   # 448 OOM 时仅下调此值
- num_workers: <int>
- cache_dir_template: "cache/features/{preproc_hash}/dinov2_vits14/{res}/{split}.npz"

### B.4 configs/classifier.yaml
- model: logistic_regression
- solver: lbfgs|saga ; C: <float> ; max_iter: <int> ; class_weight: null|balanced
- multi_class: multinomial
- standardize: {enable: <bool>, fit_on: train}
- seed: <int>
- train_splits: [train] ; eval_split: val

### B.5 configs/analysis.yaml
- transitions: [[112,224], [224,448], [112,448]]
- main_transition: [112, 448]
- q1: {wc_go_min: 150, wc_border_min: 75, wc_frac_go: 0.045, wc_frac_nogo: 0.02, rec_err_go: 0.15, acc_gain_pp: 2.0}
- high_conf: {conf_hi: 0.8, conf_higher: 0.9, count_strong: 30, frac_strong: 0.10}
- uncertainty_scores: [max_prob, entropy, margin, energy]
- escalation_rates: [0.20, 0.40, 0.60]
- utility: {type: error_to_correct, weight: 1.0}
- cost: {C_112: 1.0, C_448: 4.0, C_router: 0.0}
- q2: {nogo_acc_gap: 0.005, go_acc_gap: 0.0075, go_recall_gap: 0.10}
- figures_dir: outputs/phase0/figures ; tables_dir: outputs/phase0/tables

### B.6 configs/probe.yaml
- probe_a_features: [logits, confidence, entropy, margin, energy]
- probe_b_features: [z_112, logits, confidence, entropy, margin, energy]
- target: WC
- model: logistic_regression
- split_strategy: {mode: out_of_sample|temporary_val_split, label_source: frozen_classifier_oos}
- bootstrap: {n: <int>, seed: <int>}
- q3: {recall_gain_pp: 5, regret_reduction: 0.10, auprc_gain: 0.03, min_operating_points: 2}

### B.7 preprocessing hash 机制
- 将 B.2 中影响像素结果的字段（banner_remove_px, pad_mode, pad_fill, canonical, target_sizes, interpolation, antialias, normalize）规整为稳定 JSON（键排序）→ sha256 → 取前 12 位 hex 作为 preproc_hash。
- preproc_hash 用于所有 cache 路径：features / classifier / outcomes。
- 更改任一 preprocessing 参数 → hash 变化 → 自动落到新缓存目录，避免脏读旧缓存。
- hash 与参数快照写入 outputs/phase0/metrics.json 与 run_config.yaml。

---

## 附录 C — CLI 脚本清单（scripts/）

| 脚本 | 用途 | 主要参数 |
| --- | --- | --- |
| scripts/00_env_check.py | 环境采集（包装现有 _env_check.py） | --out outputs/logs/env_report.json |
| scripts/01_dataset_report.py | 数据集校验报告 | --config configs/base.yaml |
| scripts/02_make_canonical.py | canonical preprocessing 生成与校验 | --config configs/preprocessing.yaml --check |
| scripts/03_extract_features.py | 离线特征提取 | --config configs/extract.yaml --res 112/224/448 --cache-only |
| scripts/04_train_classifier.py | 训练 shared linear classifier | --config configs/classifier.yaml |
| scripts/05_generate_outcomes.py | 生成 val outcome 表 | --config configs/classifier.yaml |
| scripts/06_transition_analysis.py | transition + high-conf + Q1 | --config configs/analysis.yaml |
| scripts/07_uncertainty_baselines.py | uncertainty scores + AUROC/AUPRC + bins | --config configs/analysis.yaml |
| scripts/08_routing_oracle.py | budget-matched routing + oracle + Q2 | --config configs/analysis.yaml |
| scripts/09_predictability_probe.py | Probe A/B + bootstrap + Q3 | --config configs/probe.yaml |
| scripts/10_make_figures.py | 9 图 + tables | --config configs/analysis.yaml |

约定：
- 每个脚本接受 --config；运行时输出 run_config.yaml 与 metrics.json（增量合并）。
- 所有脚本必须可通过 CLI 重跑；notebook 仅用于分析，不作为唯一实验实现。
- 正式 extraction / audit 不得只在 notebook 中完成。

## 附录 D — Figures / Tables 清单

### D.1 图（outputs/phase0/figures/）
- f01_accuracy_vs_resolution.png
- f02_transition_matrix.png（Sankey-like summary）
- f03_transition_counts.png（WC / WW / CC / CW）
- f04_confidence_distribution.png（recoverable / unrecoverable）
- f05_p_recoverable_by_conf_bin.png
- f06_pr_curves_scalar_baselines.png
- f07_accuracy_vs_invocation_rate.png
- f08_recoverable_recall_vs_invocation_rate.png
- f09_scalar_vs_oracle.png

### D.2 表（outputs/phase0/tables/）
- accuracy_by_resolution.csv
- transitions_112-224.csv / transitions_224-448.csv / transitions_112-448.csv
- high_conf_recoverable.csv
- uncertainty_scores_val.csv
- uncertainty_bins.csv
- pr_curves_scalar.csv / roc_curves_scalar.csv
- routing_curves.csv
- oracle_curves.csv
- probe_metrics.csv / probe_curves.csv
- 所有图对应的 underlying table（CSV / Parquet）必须存在，禁止只保存 PNG。

---

## 附录 E — Runtime / VRAM / Storage 记录要求
- Feature extraction 完成后必须输出（112/224/448 分别）：extraction runtime；peak GPU VRAM；images/sec；cache size；batch size。写入 outputs/phase0/metrics.json 的 extraction 段，并在 docs/phase0_report.md 的 Runtime / VRAM / storage 小节汇总。
- 记录设备信息：GPU 名称、总显存、CUDA / driver、AMP dtype。
- 所有长任务的 stdout / stderr 落盘到 outputs/logs/run_*.log。

## 附录 F — Reproducibility 清单（§17）
固定并记录：
- Python version
- PyTorch version
- CUDA version（含 cuDNN、driver）
- GPU
- random seed
- DINOv2 source / checkpoint
- dataset version
- preprocessing hash
- git commit

要求：
- 所有实验从 YAML config 读取。
- 任何运行都输出 run_config.yaml 与 metrics.json。
- 不要使用 notebook 作为唯一实验实现；正式 extraction / audit 必须可通过 CLI 重跑。

### F.1 git commit 记录要求
- 每次运行采集 `git rev-parse HEAD` 与 `git status --porcelain`（是否 dirty）。
- 将 commit hash、dirty 标记、分支写入 run_config.yaml 与 metrics.json。
- 结果表格与图关联到对应 commit；若工作区 dirty，在报告中显式标注。

## 附录 G — README 第一版要求（§18）
README 第一版需包括：
- 标题：# When Is More Vision Worth It?
- 一句话：Predicting the marginal value of additional visual evidence for budget-aware visual recognition.
- 说明核心区别：
  - uncertainty asks "How uncertain is the current prediction?"
  - 本项目问 "Will additional visual evidence actually improve it?"
- 明确说明 Phase 0 first tests whether this distinction exists empirically。
- 不要声称：our method outperforms / novel / state-of-the-art / high-confidence errors are definitely predictable（必须等实验结果）。

---

## 附录 H — 明确禁止提前实现（§19）
- Transformer controller
- RL
- adaptive crop
- active vision
- visual token pruning
- MVTec
- Food-101
- ConvNeXt experiments
- corruption OOD
- extensive hyperparameter search
- end-to-end backbone fine-tuning

全部在 Phase 0 GO 后再决定。

## 附录 I — Final deliverable 清单（§20，本轮）
- docs/phase0_report.md（M7 产出），内容含：Environment / Dataset validation / Preprocessing / Backbone and cache / Linear classifier / Accuracy by resolution / Transition analysis / High-confidence recoverable errors / Uncertainty baselines / Oracle headroom / Lightweight predictability probe / Runtime / VRAM / storage / GO / NO-GO。
- 末尾必须明确给出：Q1: PASS/BORDERLINE/FAIL；Q2: PASS/BORDERLINE/FAIL；Q3: PASS/BORDERLINE/FAIL；OVERALL: GO / NO-GO / NEEDS ONE PREDEFINED FOLLOW-UP TEST。
- 不得因已写代码而默认项目继续；若结果是否定的，忠实报告。
- 同时产出：outputs/dataset_report.json / .md；outputs/phase0/figures/（9 图）；outputs/phase0/tables/（全部 underlying tables）；outputs/phase0/metrics.json；outputs/phase0/run_config.yaml。

## 附录 J — 覆盖度自检（20 项要求 → 计划落点）
| 原始要求 | 计划落点 |
| --- | --- |
| §1 Hardware | 0.3、M0（2）、M2（4） |
| §2 Dataset | M0（2.3 dataset_report 校验项） |
| §3 Canonical preprocessing | M1（3） |
| §4 Backbone | M2（4.2） |
| §5 Feature extraction | M2（4.4 cache 字段 / 4.5 指标） |
| §6 Shared linear classifier | M3（5.2） |
| §7 Validation outcome | M3（5.3） |
| §8 Transition analysis | M4（6.2–6.3） |
| §9 GO / NO-GO Q1 | M4（6.4–6.5） |
| §10 High-confidence recoverable errors | 7 |
| §11 Uncertainty baselines | M5（8.1） |
| §12 Budget-matched routing | M5（8.2） |
| §13 Sequential Oracle | M5（8.3） |
| §14 GO / NO-GO Q2 | M5（8.4） |
| §15 Predictability probe | M6（9） |
| §16 Required figures | M7（10.1）+ 附录 D |
| §17 Reproducibility | 附录 F |
| §18 Repository README | 附录 G |
| §19 What NOT to implement | 附录 H |
| §20 Final deliverable | M7（10.3–10.4）+ 附录 I |

（计划结束）
