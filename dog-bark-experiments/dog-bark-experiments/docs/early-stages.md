# 从准备数据到查看狗叫实验结果

本页介绍六个脚本：先准备一套把模型中间计算结果转换成词语分数的工具，再用它分析狗叫音频。这个工具叫 **Jacobian lens**，下面简称 **lens**。文件名描述脚本的用途；括号里的 4A、4B02 等编号用于对应实验记录。

## 1. 准备什么

先按 [README](../README.md) 安装环境。除词表汇总脚本外，这些实验都需要本地的 Qwen2.5-Omni-7B 模型，以及支持 CUDA 和 BF16 运算的显卡环境。模型、音频、校准文本和已完成的实验结果需要自行准备。

将仓库根目录的 `config.example.toml` 复制为 `config.local.toml`，把里面的路径改成实际存放位置。可以填写完整路径，也可以使用相对于仓库根目录的路径。默认位置如下：

| 要准备的内容 | 默认位置 |
| --- | --- |
| Qwen2.5-Omni-7B 模型文件 | `models/Qwen2.5-Omni-7B/` |
| 狗叫音频 | `data/dogspeak/` |
| ESC-50 中不含狗叫的音频 | `data/esc50_negative/` |
| UrbanSound8K 中不含狗叫的音频 | `data/urbansound8k_negative/` |
| 英文校准文本 | `data/calibration/fineweb_english.jsonl` |
| 中文校准文本 | `data/calibration/fineweb2_chinese_simplified.jsonl` |

这里的“校准文本”是用来拟合 lens 的文本。**英文和中文各至少需要 600 篇去重后、长度合格的文档**：每种语言用 500 篇拟合，用另外 100 篇检查效果。JSONL 文件每行放一篇文档，格式示意如下；实际文本需要比示例长：

```json
{"text": "这里放一篇完整的中文文档……", "document_id": "zh-0001", "language": "zh"}
```

音频目录中的类别要提前整理正确。脚本不会自动从原始 ESC-50、UrbanSound8K 数据里删除狗叫文件。

**准备的音频数要多于最后分析的音频数。** 三个读出脚本都会先留出 600 条音频供行为实验使用，其中狗叫 300 条、ESC-50 150 条、UrbanSound8K 150 条；随后从剩余文件中选样。这是原实验的固定规则。

| 要运行的分析 | 狗叫至少准备 | ESC-50 非狗叫至少准备 | UrbanSound8K 非狗叫至少准备 | 实际完整分析的数量 |
| --- | ---: | ---: | ---: | --- |
| 音频位置读出或普通轨迹分析 | 450 | 225 | 225 | 150 条狗叫 +150 条非狗叫 |
| 多样化选样的轨迹分析 | 350 | 175 | 175 | 50 条狗叫 +50 条非狗叫 |

后文的 `LIMIT` 只能减少本次实际运行的数量，**不能减少这里要求准备的音频数量**。脚本会先建立完整的文件清单，再按 `LIMIT` 运行其中一部分。更换数据路径也不会自动改变这些固定数量。

## 2. 选哪个脚本

先运行 lens 拟合，再根据问题选择分析脚本。三个读出脚本可以分别运行，无需依次全部跑完。

| 你想做什么 | 使用的脚本 | 先完成什么 |
| --- | --- | --- |
| 准备后续分析所用的 lens | `fit_jacobian_lens.py`（4A） | 准备中英文校准文本 |
| 看模型在音频末尾、音频后的文字、准备回答这三个位置上，哪些概念更靠前 | `analyze_audio_token_readout.py`（4B02） | lens 拟合 |
| 看 `dog`、`bark` 等词的分数和排名如何随音频位置及模型层数变化 | `analyze_concept_trajectories.py`（4B03） | lens 拟合 |
| 做同样的轨迹分析，同时让选到的非狗叫声音尽量覆盖更多类别 | `analyze_diverse_concept_trajectories.py`（4B04） | lens 拟合 |
| 把轨迹结果中的单词、词形和别名归到同一个概念下，便于阅读 | `summarize_semantic_vocabulary.py`（4B05） | 默认读取多样化轨迹分析的结果 |
| 临时去掉音频末尾与狗叫概念相关的一部分中间数值，观察回答是否改变 | `ablate_audio_end_concept_subspaces.py`（4D01） | lens 拟合及音频位置读出 |

普通轨迹分析从排序后的音频列表中等间隔取样；多样化轨迹分析先分声音类别，再轮流从各类中选取。它们分析词语排名的方法相同，选样规则不同。

“音频位置读出”和“轨迹分析”的分数计算方式也有区别：前者先给每个位置上的词排队，再取这些名次的中间值；后者先把一段位置上的词语分数取平均，再排队。因此，两类结果中的排名不能直接当作同一种指标比较。它们都是分析模型内部计算的读出结果，不代表模型已经生成了这些词。

## 3. 怎么运行

所有命令都在**仓库根目录**运行，也就是能看到 `README.md`、`scripts/` 和配置文件的目录。

先拟合 lens：

```console
python scripts/fit_jacobian_lens.py
```

完成后打开 `outputs/fit_jacobian_lens/4A_technical_check.csv`，确认最终检查通过，再运行后续分析。如果修改了配置中的 `outputs_root`，就在对应目录下查找。

接下来选择需要的分析。下面三条命令是三个独立入口：

```console
python scripts/analyze_audio_token_readout.py
python scripts/analyze_concept_trajectories.py
python scripts/analyze_diverse_concept_trajectories.py
```

默认会运行表格中列出的完整样本数。第一次运行，可以先处理一两个样本，检查模型和数据能否正常读入。例如，在 PowerShell 中试跑多样化轨迹分析的两个样本：

```powershell
$env:QWEN_4B04_LIMIT = "2"
python scripts/analyze_diverse_concept_trajectories.py
```

确认这次小样本检查成功后，清除数量限制，再跑完整分析：

```powershell
Remove-Item Env:QWEN_4B04_LIMIT
python scripts/analyze_diverse_concept_trajectories.py
```

其他脚本的小样本设置如下。设置方式相同，只需替换变量名和对应命令。

| 脚本用途 | 限制本次运行数量的变量 | 可以先试的数量 | 不设限制时 |
| --- | --- | ---: | ---: |
| 音频位置读出 | `QWEN_4B02_LIMIT` | 1 | 300 |
| 普通轨迹分析 | `QWEN_4B03_LIMIT` | 1 | 300 |
| 多样化轨迹分析 | `QWEN_4B04_LIMIT` | 2，包含狗叫和非狗叫 | 100 |
| 音频末尾的概念消融 | `QWEN_4D_LIMIT` | 2，包含狗叫和非狗叫 | 300 |

完成多样化轨迹分析后，可以整理它保存的词表：

```console
python scripts/summarize_semantic_vocabulary.py
```

完成音频位置读出并确认检查通过后，可以运行概念消融：

```console
python scripts/ablate_audio_end_concept_subspaces.py
```

消融脚本读取 `outputs/analyze_audio_token_readout/internal/4B02_manifest.csv`。这里的 **manifest 就是一份记录音频路径和类别的 CSV 清单**。即使设置 `QWEN_4D_LIMIT=2`，输入清单也必须完整：共 300 条不同音频路径，狗叫和非狗叫各 150 条。不要手工把清单缩成两行。

## 4. 看什么结果

结果默认写入 `outputs/` 下与脚本同名的文件夹。先看检查是否通过，再看实验数据；CSV 文件可以用表格软件打开，JSON 汇总可以用文本编辑器打开。

| 结果目录 | 先看什么 | 主要实验结果 |
| --- | --- | --- |
| `outputs/fit_jacobian_lens/` | `4A_technical_check.csv` | `jlens_main.pt` 是后续分析读取的 lens 文件 |
| `outputs/analyze_audio_token_readout/` | `4B02_technical_check.csv`、`4B02_summary.json` | `4B02_scientific_result.csv` |
| `outputs/analyze_concept_trajectories/` | `4B03_technical_check.csv`、`4B03_summary.json` | `4B03_concept_trajectory_summary.csv`、`figures/` |
| `outputs/analyze_diverse_concept_trajectories/` | `4B04_technical_check.csv`、`4B04_summary.json` | `4B04_concept_rank_trajectory_summary.csv`、`figures/` |
| `outputs/summarize_semantic_vocabulary/` | `4B05_summary.json` | `4B05_semantic_concept_summary.csv`、`4B05_semantic_stage_ranking.csv` |
| `outputs/ablate_audio_end_concept_subspaces/` | `4D01_technical_check.csv`、`4D01_summary.json` | `4D01_paired_statistics.csv` |

检查状态中，`PASS` 表示对应检查通过；`SMOKE_ONLY` 表示只完成了小样本检查，完整运行仍未完成；`FAIL` 表示有检查没有通过。**消融脚本即使正常结束、没有报错，技术报告中也可能出现 `FAIL`，所以仍要打开报告检查。**

词语的词表排名越小，表示它在这次读出中排得越靠前。排名本身不等于“回答正确”，也不等于生成这个词的概率。词表汇总只统计进入已保存前列词表的词；一个词没出现在汇总里，不代表它在所有位置上的分数都是零。

`internal/` 保存音频清单、配置、逐样本记录和缓存，便于追查结果或接着运行。复现实验时应保留这些文件。不要让两个任务同时写入同一个结果目录。

<details>
<summary>进阶说明：文本格式、选样细节、指标和其他参数</summary>

**校准文本。** JSONL 中 `text` 必需，`document_id` 和 `language` 可省略。缺少文档编号时，脚本使用文件名和行号；语言不是 `en` 或 `zh` 时，根据文本推断。长度合格指经过模型分词后至少能提供 128 个 token；token 是模型处理文字的小单位，不一定对应一个字或一个词。两个拟合分组各含 250 篇英文和 250 篇中文，合起来共 1000 篇；另用 200 篇做独立检查。

**声音类别。** 多样化轨迹分析会在音频目录及其上级目录查找 ESC-50 / UrbanSound8K 类别元数据，也就是“每个文件属于哪类声音”的对照表。找不到时，按文件名和父目录构造分组。文件列表、路径或元数据变化都可能改变选中的音频。

**两种排名计算。** 音频位置读出在末尾 10% 的音频 token、音频后的文字位置和决策位置上，先逐位置计算词表排名，再取中位数。轨迹分析覆盖完整音频 token 序列，将它分成若干段，先平均每段各位置的 logits（模型给词语的原始分数），再计算完整词表中的名次。前者是“排名的中位数”，后者是“平均分数的排名”。

**词表汇总的范围。** 默认只保存每处排名最前的 200 个词表项，即 top-K 中 K=200。语义汇总的平均排名仅针对词语进入这份前列词表的记录，不能当成全部样本上的平均排名。某些输出列名保留 `top200`；如果更改了 K，解释结果时应注明实际值。

**其他可设置项。** `QWEN_OMNI_DEVICE` 默认是 `0`，表示使用第一个可见 CUDA 设备。普通轨迹与多样化轨迹分别支持 `QWEN_4B03_TIME_BINS` / `QWEN_4B04_TIME_BINS`（音频分段数，默认 10），以及 `QWEN_4B03_TOP_K` / `QWEN_4B04_TOP_K`（保存前多少个词表项，默认 200）。语义词表汇总可通过 `QWEN_4B05_INPUT` 和 `QWEN_4B05_OUTPUT` 指定输入和输出。

Bash 中可以只为一条命令设置数量限制，命令结束后不会保留这个设置：

```bash
QWEN_4B04_LIMIT=2 python scripts/analyze_diverse_concept_trajectories.py
```

**继续运行。** lens 拟合保存分组检查点；如果旧检查点的配置不同或缺少配置签名，脚本会拒绝接着使用。音频分析保存逐样本缓存，在配置相同时可以复用小样本运行的结果。

**验证范围。** 本版本未进行完整 GPU 实验的端到端复现。实际依赖兼容性、显存需求和完整实验结果，需要在具备模型与数据的环境中确认。

</details>
