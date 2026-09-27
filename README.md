# Laya 随文实战示例

[English](README.en.md)

本仓库收录系列文章的运行示例，与日常开发和训练资料分开，不继承原仓库历史。目前包含第二期：中文工单推理、业务门控、FastAPI 服务、性能记录和通用数据审计。

## 公开范围

仅包含示例代码、演示策略、模拟接口测试、合成测试用例和说明文档。不包含真实工单、业务或训练数据、权重、微调产物、内部日志、凭据、本机配置、文章草稿、配图生成记录或原仓库历史。

## 安装与运行

需要 Python 3.10+ 和 Git，建议使用虚拟环境。先按 PyTorch 官方安装说明安装适合本机的构建，再从仓库根目录执行：

```bash
python -m pip install -r episodes/02-local-service/requirements.txt
cd episodes/02-local-service
python minimal.py
python quickstart.py
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

首次加载会获取所需模型文件，仓库不打包权重。离线使用前应准备并验证固定版本的完整缓存。服务启动后，进入本机端口 8000 的 `/docs` 页面，向 `/triage` 提交带 `text` 字段的中文工单。

服务启动时加载并预热，忙时返回 503；`/healthz` 检查存活，`/readyz` 检查就绪。这是教学骨架，不是完整生产部署。门控阈值是演示值，使用前需要业务验证。
## 文件与测试

本期目录中，`minimal.py` 是最小分类示例，`quickstart.py` 定义三个问题与门控策略，`server.py` 提供 API，`benchmark.py` 记录热推理，`audit_dataset.py` 审计输入数据。两个 `test_*.py` 文件分别覆盖门控与模拟 HTTP、数据审计。

在仓库根目录运行不下载模型的测试：

```bash
python -m pip install "fastapi>=0.115,<1" "httpx>=0.27,<1"
python -m unittest discover -s episodes/02-local-service -p "test_*.py" -v
```

HTTP 测试使用 FakeAgent，数据审计测试使用合成记录；两者都不验证真实模型准确率或性能。

在本期目录运行真实模型测量：

```bash
python benchmark.py --warmup 5 --runs 100 --output benchmark_result.json
```

它使用固定输入、batch=1、串行热推理，记录 P50、P95、逐次耗时和环境信息，不是 HTTP 并发压测。仓库没有预置延迟、显存或准确率成绩，生成记录默认不提交。

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