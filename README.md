# Laya 随文实战示例

[English](README.en.md)

本仓库收录系列文章的运行示例，与日常开发和训练资料分开，不继承原仓库历史。目前包含第二期的本地推理与服务，以及第三期的公开数据准备、微调、选模型和对照评测。

## 目录

- `episodes/02-local-service`：中文工单推理、业务门控、FastAPI 服务、性能记录和通用数据审计。
- `episodes/03-finetuning-evaluation`：固定版本的 BANKING77 数据重建、训练专用压缩包、Laya 原生微调和结果复算。

## 公开范围

仅包含示例代码、公开数据的下载与处理规则、演示策略、模拟接口测试、合成测试用例和说明文档。不提交真实工单、数据集文件、权重、微调产物、内部日志、凭据、本机配置、文章草稿、配图生成记录或原仓库历史。

## 第二期：安装与运行

需要 Python 3.10+ 和 Git，建议使用虚拟环境。先按 PyTorch 官方安装说明安装适合本机的构建，再从仓库根目录执行：

```bash
python -m pip install -r episodes/02-local-service/requirements.txt
cd episodes/02-local-service
python minimal.py
python quickstart.py
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

首次加载会获取所需模型文件，仓库不打包权重。离线使用前应准备并验证固定版本的完整缓存。服务启动后，打开 `http://127.0.0.1:8000/docs`，在 `POST /triage` 中点击 `Try it out`，提交带 `text` 字段的中文工单，再点击 `Execute`。

服务启动时加载并预热，忙时返回 503；`/healthz` 检查存活，`/readyz` 检查就绪。这是教学骨架，不是完整生产部署。门控阈值是演示值，使用前需要业务验证。

## 运行与部署补充

- **对外开放**：补充鉴权、请求大小限制、限流、超时与取消处理；日志应脱敏，并准备监控、灰度发布和回滚方案。示例里的忙时 503 只处理单进程推理占用，不代替这些措施。
- **输入长度**：除接口的字符长度限制外，还应按模型的 tokenizer 检查 token 数，避免超出上下文后被截断。
- **进程与设备**：示例使用一个 worker。多个 worker 通常会各自加载模型，需要分别核算内存或显存；MPS 环境还应确认实际用到的算子兼容。
- **离线部署**：准备固定 revision 的模型、配置和 tokenizer 缓存，完成后断网启动一次，验证所需文件齐全。
- **双语预加载**：只需要英文和多语言检查点时，用 `router.preload(["english", "multilingual"])` 明确指定。本文固定版本默认注册三个检查点，`preload=True` 会加载全部已注册检查点；配置 `max_loaded=2` 时不要把它当作“只预加载两个”。预热应准备中英文两组输入和问题，分别指定 `model="english"`、`model="multilingual"` 调用。

`threatens_cancellation.noul` 判断的是输入文本是否表达取消订阅的意向，不是客户未来流失率。分类置信度未达到门槛时先转人工；达到门槛后，再用取消意向判断决定是否优先跟进。

## 第二期文件与测试

本期目录中，`minimal.py` 是最小分类示例，`quickstart.py` 定义三个问题与门控策略，`server.py` 提供 API，`benchmark.py` 记录热推理，`audit_dataset.py` 审计输入数据。两个 `test_*.py` 文件分别覆盖门控与模拟 HTTP、数据审计。

在仓库根目录运行不下载模型的测试：

```bash
python -m pip install "fastapi>=0.115,<1" "httpx>=0.27,<1"
python -m unittest discover -s episodes/02-local-service -p "test_*.py" -v
```

HTTP 测试使用 FakeAgent，数据审计测试使用合成记录；两者都不验证真实模型准确率或性能。

## 第三期：微调与评测

第三期目录会从固定提交下载 BANKING77，并核对原始文件与重建后拆分的 SHA-256。生成的训练压缩包只有训练集和验证集，测试集留到权重选定后再读。

```bash
cd episodes/03-finetuning-evaluation
python prepare.py --output data/banking77
python build_training_bundle.py --data data/banking77 --output outputs/episode03-training.zip
python -m unittest discover -p "test_*.py" -v
```

完整训练需要 CUDA GPU，不会在普通测试中启动。训练、选权重、评测命令和文章历史结果见该目录的 `README.md` 与 `RESULTS.md`。

在本期目录运行真实模型测量：

```bash
python benchmark.py --warmup 5 --runs 100 --output benchmark_result.json
```

它使用固定输入、batch=1、串行热推理，记录 P50、P95、逐次耗时和环境信息，不是 HTTP 并发压测。仓库没有预置延迟、显存或准确率成绩，生成记录默认不提交。

测量时将首次下载、模型加载和预热后的推理分开记录。模型版本、语言、输入长度、问题数量、dtype 和设备都会影响耗时，跨设备比较应保持模型和输入条件一致。结果文件记录设备、模型 revision、dtype、依赖版本，以及 CUDA 下 PyTorch 分配器的显存峰值；GPU 计时前后做同步。该显存口径不是 `nvidia-smi` 显示的整进程占用。服务容量另做 HTTP 压测，计入排队、网络、序列化、输入长度变化和忙时拒绝率。

## 数据审计

自行准备 JSON 数组，每条记录包含 `state` 和 `gold.triage_category.probabilities`。脚本按 gold 概率最大项统计，不加载模型，也不输出原始工单。

```bash
python audit_dataset.py --train data/train.json --test data/test.json --output outputs/audit.json
```

仓库没有附带文章中的原始数据。脚本统计类别数量、多数类基线、训练集缺失类别和精确文本重合；示例本身复现不了文章中的具体数量，也不计算模型准确率。精确文本不重合不等于没有语义泄漏。

## 版本与发布检查

- Laya 源码提交：`4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`。
- 模型：`convaiinnovations/laya` 的 `multilingual` 子目录。
- 权重 revision：`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`。
- 其他依赖使用兼容范围，复现实验时需另存实际安装环境。
- 本仓库的发布许可证尚待维护者确定，依赖和模型的许可声明应分别确认。

在根目录执行 `python tools/check_public_content.py`，检查文件清单和常见凭据模式。它不替代人工审阅。新增文件须明确加入 `PUBLIC_FILES.txt`，不要整体复制开发目录或原仓库历史。
