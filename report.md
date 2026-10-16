# OCL 数据集上的 CLIP 零样本推理评测

日期：2026-10-17。模型为冻结的 CLIP ViT-B/32，使用 8 张 RTX 5090 完成评测。**本次结果覆盖前 10 类中有可读原图的 11,234 个物体（目标物体的 98.65%），并非包含全部缺失图片的完整官方测试集。**

## 具体实现

### 数据处理

- 按[任务说明](https://notes.sjtu.edu.cn/s/wqHA3_Uux)使用 OCL。[官方数据与代码](https://github.com/silicx/ObjectConceptLearning/tree/4ad86d929513961ff018fc53db5c7b40910ca88a)固定在 `4ad86d929513961ff018fc53db5c7b40910ca88a`。
- 读取官方 `OCL_annot_test.pkl` 和三份标签名称表。官方 test 有 9,101 张图片、25,617 个物体、221 个出现过的类别；完整类别字典含 381 类，属性和可供性词表分别为 114、170 项。
- 以物体实例数排序，**在检查缺图之前**固定样本最多的前 10 类；同数时按官方类别 ID 排序。目标为 4,151 张图片、11,388 个物体。重试和核验后有 4,068 张可读图片、11,234 个物体；83 张图片不可用，对应 154 个物体。
- 原图来自 COCO、ImageNet 和网页。保留原始标注，使用官方脚本所指的数据来源按需取图，结合官方网页补充包、原 URL 和先前历史快照恢复；ImageNet 原图采用按文件名读取原始 tar 的方式获取，避免完整大包落盘。图片完整解码并记录哈希；已识别的占位图隔离，不作为有效样本。
- `dataset.py` 继承 PyTorch `Dataset`，一个物体对应一条样本。读取现成物体框，自动裁剪并截断到图像边界，再执行 `clip.load` 返回的 RGB 图像预处理。属性和可供性编号转为 multi-hot 向量。ImageNet 本轮采用 256×256 参考坐标换算：横坐标乘 `原图宽/256`，纵坐标乘 `原图高/256`；依据和限制见后文。

### 模型调用与推理

- 使用[官方 OpenAI CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)，源码固定为 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。通过 `clip.load("ViT-B/32")` 加载权重，调用 `eval()`，逐参数设置 `requires_grad=False`，全部前向在 `torch.inference_mode()` 中运行。
- 数据集统一 tokenize 候选文本，文本特征只编码一次。类别提示为 `a photo of a {label}.`，属性提示为 `a photo of a {label} object.`，可供性提示为 `an object associated with the action: {label}.`。可供性将首个动词和词义定义拼接，保留同词不同义的 170 个标签。未用测试结果搜索提示词或调整模型。
- CLIP 的 CUDA 计算 dtype 为 `torch.float16`，参数中保留混合 dtype；图像和文本输出特征转 float32 后 L2 归一化，以矩阵乘法计算余弦相似度。属性、可供性分数不做跨标签 softmax。
- `evaluate.py` 通过 `torchrun` 启动 8 个进程，各自绑定不同的 GPU，每个 DataLoader 的 `batch_size=4`，合计每轮最多处理 32 个物体。按 `rank::world_size` 分片，保留末批且不补齐样本；每卡独立前向，在 rank 0 按原样本次序合并预测，检查索引无重复、无遗漏，再计算全局指标。
- 实验环境为 Python 3.12.3、torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.2、scikit-learn 1.9.1。主要依赖版本见 `requirements-server.lock.txt`，通用依赖见 `requirements.txt`。

### 指标计算

- **Top-k accuracy**：正确类别出现在前 k 名的物体数除以实际评测物体总数。本轮报告 Top-1/3/5，按物体数加权，不能把不同大小 batch 的准确率等权平均。
- **属性/可供性 mAP**：收集全部物体的标签分数，逐标签调用 sklearn `average_precision_score`，再对标签取宏平均。沿用 [OCL 官方 evaluator](https://github.com/silicx/ObjectConceptLearning/blob/4ad86d929513961ff018fc53db5c7b40910ca88a/utils/evaluator.py) 的口径，无正例标签的 AP 为未定义，不参与平均；本轮实际参与平均的是 94 个属性、161 个可供性标签，其余 20、9 个标签逐项列在结果中。不能把各卡/各 batch 的 mAP 直接平均。
- AP 使用非插值计算，相同分数作为同一阈值处理。JSON 和 CSV 中指标范围为 0～1；本报告转换成百分数。

### 运行与输出

在合适的 CUDA Python 环境安装依赖，执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo \
python -m torch.distributed.run --standalone --nproc_per_node=8 evaluate.py \
  --data-dir data \
  --model ViT-B/32 --device cuda --batch-size 4 \
  --missing-policy skip --imagenet-box-mode scale256 \
  --model-cache model_cache --output-dir results/clip_test_8gpu_final --save-scores
```

`metrics.json` 保存评测配置、候选文本、数据覆盖率、汇总指标和源码哈希；`per_label_ap.csv` 保存逐标签 AP 与正例数。运行时还会生成样本清单，可通过 `--save-scores` 保存分数和真实标签以重算指标。数据目录结构和准备步骤见 [README.md](README.md)。

## 实验结果

结果位于 `results/clip_test_8gpu_final/`。

| 指标 | 本次结果 |
| --- | ---: |
| 类别 Top-1 accuracy | 88.07% |
| 类别 Top-3 accuracy | 97.56% |
| 类别 Top-5 accuracy | 99.05% |
| 属性 mAP | 13.51% |
| 可供性 mAP | 32.57% |

| 类别 | 目标物体数 | 实际评测数 | Top-1 |
| --- | ---: | ---: | ---: |
| car | 1599 | 1591 | 75.49% |
| broccoli | 1282 | 1282 | 94.54% |
| boat | 1271 | 1230 | 88.21% |
| horse | 1229 | 1196 | 89.97% |
| dog | 1056 | 1031 | 93.02% |
| banana | 1050 | 1040 | 94.23% |
| truck | 1042 | 1042 | 80.71% |
| motorcycle | 1011 | 1006 | 91.65% |
| zebra | 928 | 899 | 94.55% |
| sheep | 920 | 917 | 83.75% |

**简要分析：**

- 类别识别表现较好，但 car 和 truck 混淆突出：239 个 car 预测成 truck，128 个 truck 预测成 car。Top-1 到 Top-3 的提升说明部分错误仍将正确类别排在较前位置。
- 属性与可供性的直接图文匹配仍有提升空间。为辅助理解 mAP，给每个物体相同分数的常量预测基线，在相同有效标签上分别得到属性 8.66%、可供性 27.89%。CLIP 分别高出约 4.85、4.68 个百分点；该基线不使用图像信息。
- 类别 accuracy 与多标签 mAP 衡量对象不同，不能直接把它们当作同一种“正确率”横向比较。提示词、视觉可辨性和标签定义都可能影响属性/可供性结果，本轮没有消融实验，不能量化归因。
- 最终一轮推理及合并耗时约 10.40 秒；包含进程启动、模型加载、文本编码、指标计算和保存的启动器总耗时约 29.53 秒。

## 遇到的问题、解决方法与限制

1. **缺图与占位图。** 重试原链接和网页存档后，仍有 82 张图片未取得、1 张为已隔离的占位图，共影响 154 个物体。跳过这些图片，保持前 10 类的选择不变，并记录覆盖率。因此，结果适用于当前可用的 98.65% 目标物体。
2. **ImageNet 框坐标尺度。** 原坐标直接用于部分原图会裁偏。从 [ImageNet 官方页面](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) 获取定位标注并校验 MD5，匹配到 361 个物体中的 200 个；按 256×256 参考坐标换算后，框重合度中位数从约 0.23 升至 0.92，197/200 个更吻合。采用该换算并保留原框，但这仍是有交叉核验支持的假设，其余 161 个物体没有匹配到定位框，OCL 与 ILSVRC 也可能标注不同实例或边界。脚本见 `scripts/audit_imagenet_boxes.py`，核验摘要见 `results/imagenet_box_audit.summary.json`。
3. **多卡结果汇总。** 各卡样本数可能不同，且常规分布式采样器可能补齐样本。采用不补齐的索引分片，合并后检查样本索引完整性，再计算全局指标。13 项测试通过；保存分数的 Top-k/mAP 已独立重算一致，核验记录见 `results/clip_test_8gpu_final/validation.json`。
