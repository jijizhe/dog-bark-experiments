# 告诉程序：模型和音频放在哪里

程序不会自动知道电脑里的文件位置。运行 Qwen 相关实验前，需要把这些位置填到一个配置文件里。

**只训练或比较小模型时，可以先跳过这份配置**：音频位置直接写在运行命令的 `--data-dir`、`--dog-dir` 等参数后面。

## 第一次怎么填写

1. 在项目根目录找到 `config.example.toml`。根目录就是有 `README.md` 的那一层。
2. 复制这个文件，将副本命名为 `config.local.toml`，与原文件放在一起。注意不要变成 `config.local.toml.txt`。
3. 用文本编辑器打开副本，只改等号右边、引号里面的路径。不要改左边的英文名称，也不要删除 `[paths]`。
4. 保存后运行脚本，程序会自动读取这个文件。

例如，狗叫音频放在 E 盘的 `datasets/dogspeak` 文件夹，就写：

```toml
dog_audio = "E:/datasets/dogspeak"
```

Windows 路径也建议用 `/`，不要在这个文件里直接写 `E:\datasets\dogspeak`。

`#` 开头的是说明文字，不会被程序执行。暂时用不到的项目保持原样即可；需要运行对应实验时再填写。

## 每一项是什么意思

| 名称 | 应该填什么 | 什么时候用 |
| --- | --- | --- |
| `qwen_model` | Qwen2.5-Omni-7B 完整模型文件夹 | 所有 Qwen 实验 |
| `dog_audio` | 狗叫音频文件夹 | Qwen 评测、内部信号分析、生成音频清单 |
| `cat_audio` | 猫叫音频文件夹 | 生成包含狗叫、猫叫、其他声音的清单 |
| `esc50_audio` | 已去掉狗叫的 ESC-50 音频文件夹 | Qwen 评测、内部信号分析、生成清单 |
| `urbansound8k_audio` | 已去掉狗叫的 UrbanSound8K 音频文件夹 | 同上 |
| `calibration_en` | 英文文本的 `.jsonl` 文件 | 生成 Lens |
| `calibration_zh` | 简体中文文本的 `.jsonl` 文件 | 生成 Lens |
| `outputs_root` | 保存实验结果的总文件夹 | 各脚本的默认结果位置 |
| `lens`（选填） | 已生成的 `jlens_main.pt` 文件 | 读取 Lens 做后续分析 |
| `manifest`（选填） | 音频清单的 `.csv` 文件 | 使用清单的后续实验 |

三类实验要把猫叫单独归类，因此“其他声音”的两个文件夹还需要去掉猫叫。音频怎么分目录见[数据说明](data.md)。

`outputs_root = "outputs"` 通常不用改。没有填写 `lens` 和 `manifest` 时，程序会在这个结果总文件夹下寻找：

```text
outputs/
├── fit_jacobian_lens/jlens_main.pt     生成的 Lens
└── manifests/audio_manifest.csv       音频清单
```

只有这两行前面的 `#` 被去掉后，选填值才会生效。填写 `lens` 只是告诉后续程序去哪里读取它；生成 Lens 的脚本仍把新文件保存到 `outputs_root` 下的 `fit_jacobian_lens/`。

## 路径怎么写

- `E:/datasets/dogspeak` 是完整路径，可以指向项目外的文件夹，不用把大量音频搬进项目。
- `data/dogspeak` 是相对路径，配置文件里这样写，表示项目根目录下的 `data/dogspeak/`。
- 命令中的 `--output-dir outputs/test01` 表示把这次结果放到 `outputs/test01/`。所有示例命令都应在项目根目录执行。

准备比较不同模型或数据时，为每次实验使用不同的结果文件夹，便于分清结果，也避免继续读取上次保存的中间记录。

## 想临时换数据，不想修改配置文件

支持相应参数的脚本可以在命令中指定新路径，例如：

```powershell
python scripts/evaluate_bark_detection_qwen.py --dog-dir E:/datasets/dogspeak --output-dir outputs/qwen_test02
```

这个例子只改变狗叫目录和本次输出目录，其他配置照常读取。它使用默认的 300 条狗叫、150 条 ESC-50 和 150 条 UrbanSound8K，因此要准备足够音频。

后期实验常见的两个参数是：

- `--manifest`：选择哪一张音频清单。
- `--per-label`：每个类别使用多少条音频。

生成清单时，数量参数叫 `--per-class`。具体命令见[数据说明](data.md)和[修改内部信号的实验](causal-stages.md)。

