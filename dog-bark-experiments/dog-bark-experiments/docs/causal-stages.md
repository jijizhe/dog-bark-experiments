# 狗叫实验：选脚本、运行和查看结果

这里的实验主要研究两件事：模型听到狗叫时，内部是否出现了与“狗”有关的信息；改变这些信息，是否会改变模型的回答。本页说明需要准备什么、如何选择实验，以及怎样阅读输出。

## 1. 先准备好数据和模型

按照 [README](../README.md) 安装环境，并准备以下内容：

- **模型**：本地的 Qwen2.5-Omni-7B 完整模型。运行模型需要支持 CUDA 的显卡。
- **音频**：狗叫、猫叫、其他非狗叫声音。把它们的目录填入 `config.local.toml`。
- **J-lens 文件**：前面的拟合实验生成的 `jlens_main.pt`。默认位置是 `outputs/fit_jacobian_lens/jlens_main.pt`，准备方法见 [前期实验说明](early-stages.md)。
- **J-lens 的配套说明文件**：保存拟合设置的 `frozen_fit_config.json`。复制或移动 J-lens 时，请连同这个文件一起保留。

**Lens 可以理解为观察模型内部状态的一种工具。** 本项目的 J-lens 用较前面网络层的状态来估计后面层的状态，帮助研究与“狗”等概念有关的方向。它是否适用于音频，还需要单独验证。

接下来生成一张“音频清单”。这是一份 CSV 表，记录每段音频的位置和类别；实验根据这张表找文件。以下命令生成每类 35 条、共 105 条的清单，音频目录中需要有足够的文件。

2. 脚本

这里的**干预**是指主动改变模型内部的一部分活动，例如减弱或增强与“狗”有关的方向，然后比较回答是否发生变化。实验会配合其他方向和随机方向作为对照，帮助判断变化是否与目标概念有关。

| 想研究的问题 | 使用的脚本 |
| --- | --- |
| 减弱、增强或交换“狗/猫”方向后，模型对这些答案的分数会变吗？ | `intervene_bark_concepts.py` |
| 改变内部方向后，模型实际回答“狗”“动物”等类别的表现会变吗？ | `evaluate_hierarchical_behavior.py` |
| 换一种提问方式，甚至问与狗无关的问题，模型内部还能读到“狗”的信息吗？ | `evaluate_prompt_invariance.py` |
| 在模型的哪一层、音频的哪个位置进行改变，最能影响回答？ | `localize_causal_effects.py` |
| 用文本拟合的 J-lens，能否较好地估计模型处理音频时的内部状态？ | `validate_audio_jacobian_lens.py` |
| 想运行早期采用单个文本 token 评分的行为实验，或与它比较？ | `evaluate_single_token_behavior.py` |

前五个入口分别对应概念干预、行为评估、提示词比较、位置定位和工具验证。最后一个保留了较早的评分方法，与新的行为评估方法有实质区别；具体差异放在本页末尾。

这些脚本共用音频清单和 J-lens。文件名不区分 35 条或 200 条数据，样本量在运行命令中设置。

 3. 先试跑，再扩大实验

命令中的 `--phase` 表示“运行哪一个阶段”。最适合先尝试的是 `smoke`：它用少量音频检查模型、文件路径和实验流程能否正常工作。**Smoke 成功只说明小样本流程跑通，不代表科学假设已经得到验证。**

例如，先试跑行为评估：

```bash
python scripts/evaluate_hierarchical_behavior.py --phase smoke
```

默认使用狗叫、猫叫、其他声音各 1 条，共 3 条。需要试跑其他实验时，选择对应的一条命令即可：

```bash
python scripts/intervene_bark_concepts.py --phase smoke
python scripts/evaluate_prompt_invariance.py --phase smoke
python scripts/localize_causal_effects.py --phase smoke
python scripts/validate_audio_jacobian_lens.py --phase smoke
python scripts/evaluate_single_token_behavior.py --phase smoke
```

`--per-label` 表示**每一类使用多少条音频**。它在 smoke 阶段也有效，例如每类 2 条，共 6 条：

```bash
python scripts/evaluate_hierarchical_behavior.py --phase smoke --per-label 2
```

不需要把所有脚本依次执行。根据研究问题，查看试跑和基线结果后再选择下一步。

| 实验 | 接下来通常做什么 |
| --- | --- |
| 概念干预 | 用 `dose` 比较不同改变强度，再用 `layer` 比较不同网络层 |
| 行为评估 | 用 `annotate` 检查音频类别细分，再用 `d0` 查看未干预的基线表现，之后运行 `primary` |
| 提示词比较 | 运行 `primary`，扩大样本量比较不同提问方式 |
| 位置定位 | 先用 `screen` 做较小规模筛查，再运行 `primary` |
| J-lens 音频验证 | 运行 `primary`，注意它默认每类只有 3 条 |
| 早期单 token 行为评估 | 先运行 `d0` 查看基线，再运行 `primary` |

例如，选择新的行为评估时，可以分别运行以下阶段，并在进入下一阶段前检查输出：

```bash
python scripts/evaluate_hierarchical_behavior.py --phase annotate
python scripts/evaluate_hierarchical_behavior.py --phase d0
python scripts/evaluate_hierarchical_behavior.py --phase primary
```

其中 `annotate` 会把“其他声音”进一步区分为其他动物发声、人声、非发声等类型。它根据清单信息和文件名推断，结果需要检查，不能自动当作已经人工确认的标签。

## 4. 运行后先看什么

结果默认放在 `outputs/脚本名/阶段名/`。例如，行为评估的正式阶段输出在：

```text
outputs/evaluate_hierarchical_behavior/primary/
```

建议按以下顺序阅读：

1. **先看 `*_technical_check.csv`**：检查输入、模型运行和记录是否符合程序要求。出现 `FAIL` 时，先处理对应问题；`PENDING` 表示尚不能按该检查给出通过结论。
2. **再看汇总表**：了解整体趋势、基线和对照表现。CSV 文件可以用表格软件打开。
3. **最后查看逐条结果**：回到具体音频、模型回答和条件，检查异常情况。
