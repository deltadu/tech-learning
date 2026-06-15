# Biochemistry & Gene Editing: mRNA Cancer Vaccines

本目录仿照 `tech-learning` 的学习笔记风格，提供一套面向工程/技术背景学习者的**生物化学、基因编辑与 mRNA 癌症疫苗**入门路径。内容强调概念模型、技术管线、论文阅读方法和案例分析。

> **学习边界**：本目录仅用于教育与文献理解，不是医疗建议，不提供临床决策或湿实验操作方案。mRNA 癌症疫苗、CRISPR 和肿瘤免疫治疗都属于高风险医学与生物技术领域，实际应用必须遵守伦理、监管与专业医疗要求。

---

## Learning Path

| 顺序 | 文件 | 你会学到什么 |
|---:|---|---|
| 1 | [`01_FUNDAMENTALS.md`](./01_FUNDAMENTALS.md) | 中心法则、mRNA、免疫识别、新抗原、LNP、CRISPR 与临床大图景 |
| 2 | [`02_PRACTICAL.md`](./02_PRACTICAL.md) | 如何读懂 mRNA 癌症疫苗论文、案例拆解、检查清单与练习 |
| 3 | [`CHEATSHEET.md`](./CHEATSHEET.md) | 高频概念、流程速查、常见误区和论文阅读模板 |

---

## What This Topic Covers

mRNA 癌症疫苗的技术链条可以理解为一个从**肿瘤信息提取**到**免疫系统训练**的端到端系统。个体化新抗原疫苗通常会使用患者肿瘤测序数据，预测可能被 T 细胞识别的新抗原，并将这些候选抗原编码进合成 mRNA，再通过脂质纳米颗粒或 lipoplex 等递送系统进入细胞，最终诱导抗肿瘤 T 细胞反应。[1] [2]

```
Tumor sample
  └── sequencing
       └── neoantigen prediction
            └── synthetic mRNA
                 └── LNP / lipoplex delivery
                      └── antigen presentation
                           └── T-cell response
                                └── tumor surveillance
```

---

## Why mRNA Vaccines Matter in Cancer

mRNA 疫苗平台的优势在于设计速度快、可编码多个抗原、无需将外源 DNA 整合进基因组，并且可与检查点抑制剂等免疫疗法组合。MedlinePlus Genetics 指出，疫苗 mRNA 不进入细胞核，也不会改变 DNA；细胞完成蛋白生产后会较快降解 mRNA。[3]

| 技术问题 | mRNA 平台的回答 |
|---|---|
| 如何快速定制患者特异性抗原？ | 将多个候选新抗原编码进合成 mRNA |
| 如何让免疫系统看见肿瘤突变？ | 通过细胞翻译、抗原加工与 MHC 呈递激活 T 细胞 |
| 如何避免永久改变基因组？ | mRNA 通常短暂表达，不进入细胞核，不改写 DNA |
| 如何增强疗效？ | 与 PD-1/PD-L1 抑制剂、化疗或其他免疫调节策略组合 |

---

## Representative Clinical Examples

| 项目 | 癌种 | 关键学习点 |
|---|---|---|
| Autogene cevumeran | 胰腺导管腺癌 | Nature 2023 研究显示，16 名接种患者中 8 名出现高强度新抗原特异性 T 细胞反应，responders 的无复发生存信号更好。[1] [2] |
| mRNA-4157/V940 | 高危黑色素瘤 | Merck/Moderna 2026 五年随访新闻稿称，联合 pembrolizumab 相比单药将复发或死亡风险降低 49%。[4] |

---

## Suggested Study Routine

| 天数 | 任务 | 输出 |
|---:|---|---|
| Day 1 | 阅读 `01_FUNDAMENTALS.md` 的中心法则、mRNA 和免疫识别部分 | 用自己的话解释 mRNA 不等于基因编辑 |
| Day 2 | 阅读新抗原、递送系统与 CRISPR 部分 | 画出个体化疫苗技术管线 |
| Day 3 | 阅读 `02_PRACTICAL.md` 的两个案例 | 完成案例对比表 |
| Day 4 | 使用 `CHEATSHEET.md` 阅读一篇新论文摘要 | 写一张 paper card |
| Day 5 | 回答实践问题 | 总结三个仍不理解的问题，继续查文献 |

---

## References

[1]: https://www.nature.com/articles/s41586-023-06063-y "Rojas et al. Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer. Nature, 2023"
[2]: https://pubmed.ncbi.nlm.nih.gov/37165196/ "PubMed: Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer"
[3]: https://medlineplus.gov/genetics/understanding/therapy/mrnavaccines/ "MedlinePlus Genetics: What are mRNA vaccines and how do they work?"
[4]: https://www.merck.com/news/moderna-merck-announce-5-year-data-for-intismeran-autogene-in-combination-with-keytruda-pembrolizumab-demonstrated-sustained-improvement-in-the-primary-endpoint-of-recurrence-free-survival-i/ "Merck/Moderna: 5-year KEYNOTE-942 data for intismeran autogene plus pembrolizumab"
