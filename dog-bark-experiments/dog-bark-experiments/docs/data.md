 音频和模型怎么准备


| 测试 Qwen 能否认出狗叫 | Qwen2.5-Omni-7B 完整模型、狗叫、过滤后的 ESC-50 和 UrbanSound8K 音频 |
| 分析 Qwen 内部信号 | Qwen 模型、音频，以及用于生成 Lens 的中英文文本 |
| 做狗叫、猫叫、其他声音的三类实验 | 上一行材料，再加猫叫音频和三类音频清单 |

模型应提前完整下载到本地。配置里的 `qwen_model` 要指向整个模型文件夹，包括模型参数和配套处理文件；只放一个权重文件不够。这些脚本不会替代完整的模型下载步骤。

 1. 按类别放好音频

只训练小模型时，用下面两类目录：

```text
data/student/
├── dog/          狗叫
└── non_dog/      不是狗叫的声音
```

做 Qwen 实验时，可以按下面的方式整理（以下结构仅举例）。
data/
├── dogspeak/                  狗叫
│   ├── dog_001/               第一只狗的录音
│   └── dog_002/               第二只狗的录音
├── cat/                       猫叫
├── esc50_negative/            从 ESC-50 中挑出的其他声音
│   ├── rain/                  雨声
│   └── engine/                引擎声
├── urbansound8k_negative/      从 UrbanSound8K 中挑出的其他声音
│   ├── siren/                 警笛声
│   └── drilling/              钻孔声
└── calibration/               生成 Lens 时用的文本
```

雨声、引擎声等只是目录示例，不要求一定包含这些类别。原实验使用 DogSpeak 狗叫数据；也可以填写自己的狗叫音频位置。

文档中“正样本”指狗叫，“负样本”指用来与狗叫区分的声音，不是在评价音频好坏。

先筛选声音，再运行代码：

 狗叫不能混进“其他声音”中。
 做三类实验时，猫叫单独放在猫叫目录，也不能混进“其他声音”中。
 清单生成器不会根据 ESC-50 / UrbanSound8K 原始标签文件自动完成这些筛选。
 尽量把不同声音分到对应的类别子目录。程序根据音频所在的直接上级目录识别类别；全部平铺在一个目录里，会失去按声音种类均衡选取的依据。

如果已知录音属于哪只狗，可使用 `dog_001`、`dog_002` 这类目录名，生成清单时会优先覆盖不同犬只。训练集和测试集也应事先分开；代码不会自动检查它们是否包含相同录音。

 2. 需要三类实验时，生成音频清单

清单是一张 CSV 表，记录“这条音频在哪里、正确类别是什么”。** 后面的实验按这张表取数据，所以更换数据集不需要再复制脚本。

```powershell
python scripts/build_audio_manifest.py --per-class 50 --output outputs/manifests/custom.csv
```

`--per-class 50` 表示每类 50 条，共 150 条：

| 清单里的标签 | 意思 | 本例数量 |
| --- | --- | --- |
| `dog_bark` | 狗叫 | 50 |
| `cat_vocalization` | 猫叫 | 50 |
| `non_dog_bark` | 除狗叫、猫叫以外的其他声音 | 50 |

本例的“其他声音”从 ESC-50 和 UrbanSound8K 各选 25 条。各文件夹都必须有足够音频。希望每类 200 条，就把 `50` 改成 `200`。

命令完成后，检查 `outputs/manifests/custom.csv`。后续实验要使用这张表，就在对应命令中写 `--manifest outputs/manifests/custom.csv`。只生成清单还没有运行模型，也不会生成 Lens。

如果换成自己的音频，还要检查清单里的 `source`（数据来源）列。生成器沿用了原实验的来源名称，例如狗叫写为 `dogspeak`、猫叫写为 `bark_translate_cat`，不会自动识别真实来源。使用其他数据时，请将这些名称改为实际来源后再做实验。

不写参数时，生成器默认每类选 200 条，保存到 `outputs/manifests/audio_manifest.csv`；若在配置中设置了其他清单位置，则使用该位置。

3. 想自己制作 CSV，也可以

至少保留下面四列，列名按原样填写：

| 列名 | 填写内容 |
| --- | --- |
| `sample_id` | 每条音频的唯一编号，不要重复 |
| `audio_path` | 音频的完整路径 |
| `label` | 上表中的三个英文标签之一 |
| `source` | 真实的数据来源名称，例如自己的数据集名称 |

格式示例，实际使用时要换成存在的音频路径：

```csv
sample_id,audio_path,label,source
dog_001,E:/audio/dog_001.wav,dog_bark,my_dataset
cat_001,E:/audio/cat_001.wav,cat_vocalization,my_dataset
rain_001,E:/audio/rain_001.wav,non_dog_bark,my_dataset
```

还可以添加 `selection_stratum`（用于分组抽样的类别或犬只）和 `semantic_subtype`（更细的声音类别）。有些后续分析会进一步区分动物声、人声和环境声，运行前要核对自动分类结果。

生成器会检查是否重复列出同一路径，但不会听音频，也不会发现“同一段录音换了文件名”。移动音频文件后，清单里的旧路径会失效，需要重新生成或更新。

 4. 只有生成 Lens 时才需要文本

Lens 是后续分析模型内部信号的工具。生成它时需要英文和简体中文文本，分别保存为 JSONL 文件。“JSONL”就是每一行各写一条 JSON 记录。

```text
data/calibration/
├── fineweb_english.jsonl
└── fineweb2_chinese_simplified.jsonl
```

每行必须有非空的 `text`，内容是实际文本。可以额外填写 `document_id` 和 `language`。例如：

```json
{"text": "This is an example of the file format.", "document_id": "en_001", "language": "en"}
```

上面只演示格式，不能用一两句示例文本完成实验。**每种语言至少需要 600 条满足长度等筛选条件的文本**，建议留出余量。

