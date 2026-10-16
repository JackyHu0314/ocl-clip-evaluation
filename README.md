# OCL 上的 CLIP 零样本评测

使用冻结的 OpenAI CLIP ViT-B/32，在 OCL 测试集中样本最多的前 10 个类别上评测物体类别、属性和可供性。技术栈为 Python、PyTorch、OpenAI CLIP 和 scikit-learn。

流程：读取物体标注 → 按框裁剪 → CLIP 编码图片和候选文本 → 计算余弦相似度 → 计算 Top-k 和 mAP。

## 实验结果

使用 8 张 RTX 5090，每卡 batch size 为 4，共评测 11,234 个物体。

| 指标 | 结果 |
| --- | ---: |
| 类别 Top-1 | 88.07% |
| 类别 Top-3 | 97.56% |
| 类别 Top-5 | 99.05% |
| 属性 mAP | 13.51% |
| 可供性 mAP | 32.57% |

原始目标为 11,388 个物体，83 张不可用图片对应的 154 个物体已跳过。部分 ImageNet 标注框采用 256×256 参考坐标换算，依据与限制见 [report.md](report.md)。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `dataset.py` | 选择前 10 类、读取标注、裁剪物体、构造多标签向量 |
| `evaluate.py` | 调用冻结的 CLIP，支持单卡与多卡推理，保存评测结果 |
| `metrics.py` | 计算类别 Top-k、属性和可供性 mAP |
| `scripts/` | 准备数据、重试缺图、核验 ImageNet 框坐标 |
| `tests/` | 检查裁剪与标签对应、指标计算和多卡合并 |
| `results/` | 本次评测的汇总指标、逐标签 AP 和核验摘要 |
| `report.md` | 具体实现、实验分析、问题与限制 |

## 环境与数据

在 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

本次实验使用 Python 3.12.3、PyTorch 2.11.0+cu128 和 torchvision 0.26.0+cu128；相应主要依赖版本记录在 `requirements-server.lock.txt`。CLIP 源码固定到指定提交，权重在首次调用时下载至 `model_cache/`。

数据放在 `data/`，其 `resources/` 子目录需要 `OCL_annot_test.pkl`（或 `OCL_annot.tgz`）、`OCL_class_object.json`、`OCL_class_attribute.json` 和 `OCL_class_affordance.json`。图片按标注中 `name` 字段的相对路径存放。

如需从官方来源准备数据，安装数据工具依赖并执行：

```bash
python -m pip install -r requirements-data.lock.txt
python scripts/prepare_ocl_data.py --data-dir data --split test --top-k 10 --download --fetch-web-archive
python scripts/fetch_imagenet_subset.py --data-dir data
python scripts/fetch_imagenet_validation.py --data-dir data
python scripts/prepare_ocl_data.py --data-dir data --verify-only
```

准备脚本在有缺图时以退出码 2 提醒数据不完整；各下载步骤完成后再次核验。原链接可用性会变化，以生成的 `data/manifests/test_top10.summary.json` 为准。也可以通过 `--imagenet-root` 使用已有的 ImageNet 图片目录。

## 运行评测

单卡运行：

```bash
python evaluate.py --data-dir data --device cuda --batch-size 4 \
  --missing-policy skip --imagenet-box-mode scale256 \
  --model-cache model_cache --output-dir results/clip_test
```

复现本次 8 卡配置：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo \
python -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
  --data-dir data --device cuda --batch-size 4 \
  --missing-policy skip --imagenet-box-mode scale256 \
  --model-cache model_cache --output-dir results/clip_test_8gpu_final --save-scores
```

`--missing-policy skip` 允许在缺图时评测可用部分；`--imagenet-box-mode scale256` 显式启用本报告采用的坐标换算。默认模式遇到缺图或未明确的 ImageNet 坐标会报错，避免静默改变评测范围。

检查实现：

```bash
python -m unittest discover -s tests -v
```

结果中的 AP、mAP 和 accuracy 均为 0～1。属性和可供性逐标签计算 AP，再取宏平均；没有正样本的标签不参与平均。仓库只包含代码、文档和精简结果，数据、模型权重及虚拟环境由 `.gitignore` 排除。

## 参考

- [OCL 官方数据与代码](https://github.com/silicx/ObjectConceptLearning)
- [OpenAI CLIP](https://github.com/openai/CLIP)
