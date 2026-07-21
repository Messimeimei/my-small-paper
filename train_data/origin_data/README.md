# 数据查看说明

## 数据介绍

该数据是直接从《《Reward Modeling for Scientific Writing Evaluation》》论文作者要的官方原始数据中按照 train 字段提取出来的原始分任务、分维度训练数据。后续会对数据根据任务需要做进一步的清洗、蒸馏等，会在其他文件夹下说明。[当前文件夹](train_data/origin_data/README.md)仅用来保存原始官方数据。

## Cursor 中查看

不要直接打开本目录下数十 MB 的完整训练文件。请打开 `preview/` 中的预览文件：

- 二分类任务每个文件包含标签 0、1 各一条；
- 五级评分任务每个文件包含标签 1-5 各一条；
- 预览样本保留了完整的原始字段和 prompt。

## 终端中查看完整文件

流式浏览文件，不让 Cursor 加载全文：

```bash
less -S 'train_data/origi_data/rw_gen__coherence__n4890.json'
```

在 `less` 中按 `/` 搜索，按 `q` 退出。

查看某个完整文件中的指定样本，可以使用 Python。下面的 `0` 是样本索引：

```bash
python -c "import json; p='train_data/origi_data/rw_gen__coherence__n4890.json'; d=json.load(open(p, encoding='utf-8')); print(json.dumps(d['train'][0], ensure_ascii=False, indent=2))" | less -S
```

完整文件供训练程序读取，`preview/` 仅用于人工浏览，不能代替完整训练数据。
