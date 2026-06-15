# mRNA 癌症疫苗与基因编辑速查表

这份速查表用于复习 `01_FUNDAMENTALS.md` 与 `02_PRACTICAL.md`。它采用工程笔记风格，把术语、流程、判断标准和常见误区压缩到一页式参考。

---

## Core Concepts

| 概念 | 一句话解释 | 易混点 |
|---|---|---|
| mRNA | DNA 信息的临时工作副本，可被核糖体翻译为蛋白质；NHGRI 将其定义为参与蛋白合成的单链 RNA。[1] | mRNA 不是 DNA，也不是永久指令 |
| mRNA 疫苗 | 把编码抗原的 mRNA 递送进细胞，让细胞短暂表达抗原并诱导免疫反应 | 通常不进入细胞核，不改变 DNA。[2] |
| 新抗原 | 肿瘤突变产生、正常组织通常不存在的抗原 | 预测结合强不等于真实免疫原性强 |
| MHC/HLA | 细胞展示肽段给 T 细胞的分子平台 | 不同 HLA 决定可展示肽段不同 |
| CD8+ T cell | 识别 MHC I 上肽段并杀伤目标细胞 | 需要足够抗原呈递和共刺激 |
| PD-1/PD-L1 | T 细胞免疫刹车通路 | 抑制剂不是疫苗，但可增强 T 细胞效应 |
| LNP | 保护并递送 RNA 的脂质纳米颗粒 | 递送不仅是包装，还涉及靶向和内体逃逸 |
| CRISPR-Cas9 | guide RNA 引导 Cas9 到特定 DNA 位置进行切割或编辑。[3] | CRISPR 是基因编辑，mRNA 疫苗通常不是 |

---

## One Diagram to Remember

```
Tumor mutation
  │
  ▼
Neoantigen candidate
  │
  ▼
Synthetic mRNA
  │
  ▼
LNP / lipoplex
  │
  ▼
APC uptake and translation
  │
  ▼
Peptide-MHC presentation
  │
  ▼
T-cell activation
  │
  ▼
Tumor recognition
```

---

## mRNA Vaccine vs CRISPR

| 维度 | mRNA 癌症疫苗 | CRISPR 基因编辑 |
|---|---|---|
| 主要目标 | 表达抗原，训练免疫系统 | 改变 DNA、RNA 或基因表达状态 |
| 是否永久 | 通常短暂 | 可能永久，取决于编辑方式 |
| 常见载荷 | 抗原编码 mRNA | Cas 蛋白、Cas mRNA、sgRNA、编辑器 |
| 癌症应用 | 新抗原疫苗、联合检查点抑制剂 | 工程化 T 细胞、靶点筛选、免疫逃逸研究 |
| 主要风险 | 免疫反应不足或炎症过强 | 脱靶、大片段变异、递送和长期安全性 |

---

## Clinical Signals to Know

| 项目 | 关键数据 | 如何解读 |
|---|---|---|
| Autogene cevumeran | Nature 2023 胰腺癌研究中，16 名接种患者有 8 名诱导高强度新抗原特异性 T 细胞反应；responders 的无复发生存信号更好。[4] [5] | 早期试验证明可行性和免疫原性，不等于已成为标准治疗 |
| mRNA-4157/V940 | Merck/Moderna 2026 称，KEYNOTE-942 五年随访中联合 pembrolizumab 将复发或死亡风险降低 49%。[6] | 代表个体化 mRNA 新抗原疗法向后期临床推进 |

---

## Paper Card Template

```yaml
paper_card:
  title: ""
  disease: ""
  setting: "adjuvant / neoadjuvant / metastatic / preventive"
  intervention: "mRNA vaccine / CRISPR / cell therapy / combination"
  antigen_strategy: "shared antigen / neoantigen / viral antigen"
  personalization: "yes/no; number of antigens"
  delivery: "LNP / lipoplex / viral / ex vivo / other"
  comparator: "control arm or historical baseline"
  immune_readout: "T-cell response / TCR expansion / cytokines / MHC"
  clinical_endpoint: "RFS / DMFS / OS / ORR / safety"
  key_result: ""
  limitation: ""
  next_question: ""
```

---

## Readiness Checklist

| 问题 | Good Sign | Red Flag |
|---|---|---|
| 抗原是否可信？ | 有突变、表达、HLA、T cell 验证 | 只有算法预测 |
| 递送是否可信？ | 有细胞摄取、表达和安全性证据 | 只展示配方但无功能读数 |
| 免疫是否可信？ | 多方法证明抗原特异 T 细胞反应 | 只有总炎症指标 |
| 临床是否可信？ | 随机对照、预设终点、长期随访 | 小样本、无对照、探索性结论 |
| 转化是否可信？ | 制造时间窗和质控明确 | 个体化制造链不可解释 |

---

## Common Mistakes

| Mistake | Correct Model |
|---|---|
| mRNA 疫苗会改写 DNA | 疫苗 mRNA 通常不进入细胞核，也不改变 DNA。[2] |
| 新抗原预测越多越好 | 质量、呈递、表达和 TCR 识别比数量更重要 |
| 有 T 细胞反应就一定有疗效 | 免疫反应需要转化为 RFS、OS 或其他临床终点 |
| CRISPR 可以直接解决所有癌症 | 癌症多基因、多克隆、微环境复杂，递送和安全性仍是难题 |
| LNP 只是包装 | LNP 决定稳定性、组织分布、细胞摄取和内体逃逸 |

---

## References

[1]: https://www.genome.gov/genetics-glossary/Messenger-RNA-mRNA "NHGRI: Messenger RNA (mRNA)"
[2]: https://medlineplus.gov/genetics/understanding/therapy/mrnavaccines/ "MedlinePlus Genetics: What are mRNA vaccines and how do they work?"
[3]: https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/questions-and-answers-about-crispr "Broad Institute: Questions and Answers about CRISPR"
[4]: https://www.nature.com/articles/s41586-023-06063-y "Rojas et al. Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer. Nature, 2023"
[5]: https://pubmed.ncbi.nlm.nih.gov/37165196/ "PubMed: Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer"
[6]: https://www.merck.com/news/moderna-merck-announce-5-year-data-for-intismeran-autogene-in-combination-with-keytruda-pembrolizumab-demonstrated-sustained-improvement-in-the-primary-endpoint-of-recurrence-free-survival-i/ "Merck/Moderna: 5-year KEYNOTE-942 data for intismeran autogene plus pembrolizumab"
