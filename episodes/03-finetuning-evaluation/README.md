# 第三期：微调与评测

[English](README.en.md)

这套代码对应第三期文章，用公开的 BANKING77 数据复现训练、选模型和对照评测。数据、模型和依赖都固定到具体版本；训练包只放训练集与验证集，测试集留到权重选定后再使用。

仓库不附带数据、模型权重和训练产物。`RESULTS.md` 记录的是文章实验已经得到的结果，不代表每台机器重新运行都会得到完全相同的数字。发布这套示例时也没有另起一次完整 GPU 训练。

## 1. 准备环境

使用 Python 3.10+。先按 PyTorch 官方说明安装与 CUDA 匹配的版本，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

文章中的完整训练使用 NVIDIA T4 16GB。RTX 3060 Ti 8GB 也做过早期试验，但当时训练范围和参数不同，不能把两次结果只归因于显存差异。

## 2. 下载并重建数据

```bash
python prepare.py --output data/banking77
```

脚本下载固定提交中的三个官方文件，核对 SHA-256，清理规范化重复项、冲突标签和训练/测试重合文本，再按类别划分训练集和验证集。成功后应得到：

- 训练集：7,957 条
- 验证集：1,989 条
- 测试集：3,075 条
- 三份数据之间的规范化文本重合数：0

文件哈希写在 `settings.py` 中。只要上游内容或处理方式变了，脚本就会报错，而不是悄悄生成另一份数据。

## 3. 生成训练专用压缩包

```bash
python build_training_bundle.py \
  --data data/banking77 \
  --output outputs/episode03-training.zip
```

压缩包内有训练代码、训练集、验证集、协议和清单，**没有 `test.json`**。可以把它上传到 Colab，解压后安装依赖并运行：

```bash
python train.py --root . --output outputs/run
```

训练沿用 Laya 原生决策结构：编码器、决策头和打分路径参与更新，action head 冻结；FP32 参数配合 FP16 autocast 与 GradScaler。每轮验证后保存完整推理快照，按验证准确率选择模型；准确率相同时再比较 Macro-F1，仍相同就保留较早的轮次。

免费 Colab 会话可能中断。代码每轮先写验证记录，再导出权重；需要主动停止时，在输出目录创建名为 `STOP` 的文件。会话中断后，可以对完整的运行目录重新生成选择记录：

```bash
python select_checkpoint.py outputs/run
```

它不会恢复优化器状态，也不会自动把训练集和验证集合并重训。

## 4. 评测原始模型与微调模型

把选中的 `export` 目录下载到本机。测试集不在训练压缩包里，评测时从第 2 步生成的数据目录读取：

```bash
python evaluate.py \
  --data data/banking77 \
  --trained outputs/run/epoch-07/export \
  --output outputs/evaluation

python verify_results.py \
  --data data/banking77 \
  --result outputs/evaluation
```

两个模型使用同一标签顺序、同一问题、同一批测试文本和相同编码结果。`verify_results.py` 不加载模型，只根据保存的逐条概率重新计算准确率、Macro-F1、NLL 和成对变化，用来检查汇总文件有没有算错或被改动。

这份测试集在前期探索中用过一部分，因此后续评测属于历史复测，不写成全新的盲测。正式项目还应保留一份从未用于调参的最终验收集。

## 5. 不下载模型的测试

```bash
python -m unittest discover -p "test_*.py" -v
```

这些测试覆盖拆分与清理规则、选模型规则、指标复算、训练包隔离和文件分块，不启动 GPU 训练。

## 文件说明

- `prepare.py`：下载、清理并重建三份数据。
- `build_training_bundle.py`：生成不含测试集的训练压缩包。
- `modeling.py`：加载 Laya、编码 77 个选项、保存完整推理权重。
- `training_core.py` / `train.py`：运行验证集选权重阶段。
- `select_checkpoint.py`：根据已完成轮次重建选择记录。
- `evaluate.py`：在同一测试输入上比较原始模型和微调模型。
- `verify_results.py`：CPU 独立复算保存的结果。
- `RESULTS.md`：文章实验的历史结果和适用边界。
