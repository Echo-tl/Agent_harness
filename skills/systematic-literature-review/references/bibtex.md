# BibTeX 引用格式规则

## 文献条目
- 报告末尾（参考资料节）以 BibTeX 条目形式列出全部来源，每条一个 `@entry{...}` 块。
- **arXiv 论文一律用 `@misc`，不用 `@article`**（arXiv 不是正式期刊）。
- 网页/报告用 `@misc` 或 `@techreport`。

## 模板

```bibtex
@misc{author2026shorttitle,
  title        = {完整标题},
  author       = {作者1 and 作者2 and ...},
  year         = {2026},
  howpublished = {arXiv preprint arXiv:XXXX.XXXXX},
  url          = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## 规则
- cite key：`首作者姓氏+年份+短标题`（小写、无空格）。
- `author` 用 `and` 分隔多个作者；超过 6 个作者可写 `et al.`。
- `url` 必须真实存在，禁止编造。
- 正文用 `\cite{key}` 引用；无 LaTeX 时在报告中给出每条条目即可。
