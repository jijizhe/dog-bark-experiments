# 每个脚本是做什么的

`scripts/` 中有 18 个实验脚本，文件名直接说明用途，不再带日期、实验序号或数据量。另有 `project_paths.py`，负责读取模型、数据和结果的保存位置，无需单独运行。

下表中的 J-lens 是分析模型内部信号的工具，帮助把模型计算中的数字转成与词语有关的分数。

## 按用途找脚本

| 文件名 | 
| --- | --- |
| `train_bark_classifier.py` | 用狗叫和其他声音训练一个小型分类模型。 |
| `compare_bark_classifiers.py` | 用同一批音频比较两个小型分类模型的识别结果。 |
| `evaluate_bark_detection_qwen.py` | 让 Qwen 判断音频是否属于狗叫，并统计结果。 |
| `build_audio_manifest.py` | 从狗叫、猫叫和其他声音中选样本，生成记录路径和类别的 CSV 表格。 |
| `inspect_qwen_architecture.py` | 查看 Qwen 内部有哪些模块和层，确认后续分析的位置。 |
| `probe_qwen_jacobian_adapter.py` | 用短文本检查 Qwen 能否接入 J-lens 分析工具。 |
| `fit_jacobian_lens.py` | 用中英文文本生成 J-lens，供后续分析模型内部信号时使用。 |
| `analyze_audio_token_readout.py` | 查看模型处理音频时，不同位置的内部表示对应哪些词语。 |
| `analyze_concept_trajectories.py` | 观察“狗”“猫”等词语的分数和排名，如何随音频时间和模型层数变化。 |
| `analyze_diverse_concept_trajectories.py` | 先从不同声音类别中选样本，再观察词语分数和排名的变化。 |
| `summarize_semantic_vocabulary.py` | 把上一步得到的词语归类汇总，方便查看常出现的概念。 |
| `ablate_audio_end_concept_subspaces.py` | 在音频末端对应的模型位置削弱部分概念信息，观察输出是否变化。 |
| `intervene_bark_concepts.py` | 在模型作答的位置削弱、加入或交换狗叫与猫叫相关的信息，观察判断如何变化。 |
| `evaluate_single_token_behavior.py` | 按指定的单个词元评分，检查模型对声音类别的判断；词元是模型切分文字的基本单位。 |
| `evaluate_hierarchical_behavior.py` | 检查模型能否区分不同声音类别，并考虑同一答案的不同说法。 |
| `evaluate_prompt_invariance.py` | 改变提问方式，检查模型判断和干预效果是否稳定。 |
| `localize_causal_effects.py` | 改变干预发生的层和位置，寻找哪些位置对判断影响较大。 |
| `validate_audio_jacobian_lens.py` | 检查用文本生成的 J-lens，在模型处理音频时还能否准确估计内部信号。 |

训练和基础评测的运行方法见 [README](../README.md)。模型内部分析的步骤见[内部信号分析说明](early-stages.md)和[修改内部信号的实验说明](causal-stages.md)。

