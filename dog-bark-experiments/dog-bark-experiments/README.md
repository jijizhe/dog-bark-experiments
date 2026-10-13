# 狗叫识别实验

本仓库包含轻量狗叫分类器训练、Qwen2.5-Omni 狗叫识别评测，以及模型内部信号分析和干预实验。音频数据、模型权重和历史实验结果需另行准备。

## 运行环境

| 项目 | 要求 |
| --- | --- |
| Python | 3.11 或 3.12 |
| 轻量分类器 | 训练支持 CPU 或 CUDA；INT8 模型比较使用 CPU |
| Qwen 相关实验 | 本地 Qwen2.5-Omni-7B 完整模型、支持 CUDA 的 NVIDIA 显卡；部分实验需要 BF16 支持 |
| 主要工具包 | PyTorch、NumPy、Librosa、SoundFile、scikit-learn |
| Qwen 推理 | Transformers、Accelerate、qwen-omni-utils |
| 内部信号分析 | Jacobian Lens、pandas、Matplotlib；安装 Jacobian Lens 需要 Git |
| 音频处理 | 可用的音频解码环境，按需安装 FFmpeg |

依赖按实验类型分别列在以下文件中：

- [requirements/lightweight.txt](requirements/lightweight.txt)：轻量分类器训练与比较。
- [requirements/qwen.txt](requirements/qwen.txt)：Qwen 基础评测。
- [requirements/interpretability.txt](requirements/interpretability.txt)：Qwen 内部信号分析与干预。

PyTorch 需与本机 CUDA 环境匹配。依赖清单未锁定完整环境，版本兼容情况及模型、数据路径设置见[配置说明](docs/configuration.md)。

## 实验内容

| 脚本 | 用途 |
| --- | --- |
| [train_bark_classifier.py](scripts/train_bark_classifier.py) | 训练狗叫与非狗叫的二分类模型，保存原始模型参数和 INT8 压缩模型。 |
| [compare_bark_classifiers.py](scripts/compare_bark_classifiers.py) | 在同一批测试音频上比较两个轻量分类器的识别结果。 |
| [evaluate_bark_detection_qwen.py](scripts/evaluate_bark_detection_qwen.py) | 测试 Qwen 对狗叫及其他环境声音的判断，统计识别准确率。 |
| [build_audio_manifest.py](scripts/build_audio_manifest.py) | 从狗叫、猫叫和其他声音中抽样，生成记录音频路径及类别的 CSV 清单。 |
| [inspect_qwen_architecture.py](scripts/inspect_qwen_architecture.py) | 查看 Qwen 的模型结构和网络层，确定后续分析的位置。 |
| [probe_qwen_jacobian_adapter.py](scripts/probe_qwen_jacobian_adapter.py) | 检查 Qwen 与 Jacobian Lens 分析工具的接口是否可用。 |
| [fit_jacobian_lens.py](scripts/fit_jacobian_lens.py) | 用中英文文本拟合 Jacobian Lens，供后续分析模型内部信号。 |
| [analyze_audio_token_readout.py](scripts/analyze_audio_token_readout.py) | 读取音频末尾、音频后的文字及回答位置上，狗叫等概念的分数和排名。 |
| [analyze_concept_trajectories.py](scripts/analyze_concept_trajectories.py) | 分析概念分数和排名随音频位置、模型层数的变化。 |
| [analyze_diverse_concept_trajectories.py](scripts/analyze_diverse_concept_trajectories.py) | 从不同声音类别中均衡选样，再分析概念分数和排名的变化。 |
| [summarize_semantic_vocabulary.py](scripts/summarize_semantic_vocabulary.py) | 将分析结果中的词形和别名归并为概念，汇总排名及出现情况。 |
| [ablate_audio_end_concept_subspaces.py](scripts/ablate_audio_end_concept_subspaces.py) | 移除音频末端位置上与狗叫有关的部分信号，观察输出变化。 |
| [intervene_bark_concepts.py](scripts/intervene_bark_concepts.py) | 在回答位置削弱、增强或交换狗与猫的概念信号，比较输出分数。 |
| [evaluate_single_token_behavior.py](scripts/evaluate_single_token_behavior.py) | 使用固定的单个词元（模型切分文字的单位）评分，评估模型对声音类别的判断。 |
| [evaluate_hierarchical_behavior.py](scripts/evaluate_hierarchical_behavior.py) | 评估狗、动物、发声等类别的判断，考虑答案的不同写法，并分析自由回答。 |
| [evaluate_prompt_invariance.py](scripts/evaluate_prompt_invariance.py) | 改变提问方式，比较内部概念信号和干预效果是否稳定。 |
| [localize_causal_effects.py](scripts/localize_causal_effects.py) | 比较不同网络层和音频位置的干预效果，定位影响判断的位置。 |
| [validate_audio_jacobian_lens.py](scripts/validate_audio_jacobian_lens.py) | 检查用文本拟合的 Jacobian Lens 能否准确估计模型处理音频时的内部信号。 |

具体数据要求和运行参数见[数据说明](docs/data.md)、[内部信号分析](docs/early-stages.md)及[干预实验](docs/causal-stages.md)。
