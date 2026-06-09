# 生物化学与基因编辑基础：mRNA 癌症疫苗入门

面向工程背景学习者的生命科学笔记：先建立**分子生物学中心法则、免疫识别、mRNA 递送、肿瘤新抗原、CRISPR 基因编辑**的共同语言，再理解为什么 mRNA 疫苗可以被用于癌症治疗。

> **学习定位**：本文是学习材料，不是医疗建议，也不提供实验操作方案。癌症治疗与基因编辑属于高风险医学领域，具体诊疗必须由合格医生与监管合规的临床试验决定。

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Central Dogma: DNA → RNA → Protein](#central-dogma-dna--rna--protein)
3. [mRNA: 临时指令，而不是永久改写](#mrna-临时指令而不是永久改写)
4. [免疫系统如何识别癌细胞](#免疫系统如何识别癌细胞)
5. [癌症疫苗的核心：抗原与新抗原](#癌症疫苗的核心抗原与新抗原)
6. [mRNA 癌症疫苗如何工作](#mrna-癌症疫苗如何工作)
7. [递送系统：LNP、lipoplex 与 APC 靶向](#递送系统lnplipoplex-与-apc-靶向)
8. [CRISPR 基因编辑基础](#crispr-基因编辑基础)
9. [mRNA 疫苗与基因编辑的交叉点](#mrna-疫苗与基因编辑的交叉点)
10. [Clinical Landscape: 目前学到哪里](#clinical-landscape-目前学到哪里)
11. [Algorithm Cheat Sheet](#algorithm-cheat-sheet)
12. [References](#references)

---

## The Big Picture

mRNA 癌症疫苗的核心思想不是“直接杀死癌细胞”，而是**把肿瘤特异性信息编码成 mRNA，让患者体内细胞短暂产生抗原片段，从而训练 T 细胞识别并攻击携带这些抗原的癌细胞**。这条路径把分子生物学、免疫学、肿瘤基因组学、纳米递送与临床肿瘤学连接在一起。

| 层级 | 你要理解的问题 | 对 mRNA 癌症疫苗的意义 |
|---|---|---|
| 生物化学 | DNA、RNA、蛋白质如何相互转换 | mRNA 是蛋白表达的临时模板 |
| 细胞生物学 | 核糖体、内吞、胞质递送如何工作 | 决定 mRNA 能否进入细胞并被翻译 |
| 免疫学 | 抗原如何被呈递给 T 细胞 | 决定疫苗能否激活抗肿瘤免疫 |
| 肿瘤基因组学 | 肿瘤突变如何产生新抗原 | 决定“个体化”疫苗编码什么 |
| 递送工程 | LNP/lipoplex 如何保护并递送 RNA | 决定稳定性、靶向性和副作用 |
| 临床医学 | 疗效、安全性与联合治疗如何评估 | 决定技术能否成为标准治疗 |

```
┌─────────────────────────────────────────────────────────────────┐
│                    mRNA CANCER VACCINE LANDSCAPE                │
│                                                                 │
│  Tumor Biopsy/Resection                                          │
│        │                                                        │
│        ▼                                                        │
│  DNA/RNA Sequencing ──► Neoantigen Prediction ──► mRNA Design    │
│        │                                      │                 │
│        ▼                                      ▼                 │
│  Patient-specific tumor mutations         Synthetic mRNA         │
│                                               │                 │
│                                               ▼                 │
│                                    LNP / Lipoplex Delivery       │
│                                               │                 │
│                                               ▼                 │
│                                  Antigen-presenting cells        │
│                                               │                 │
│                                               ▼                 │
│                                      CD8+/CD4+ T-cell response   │
│                                               │                 │
│                                               ▼                 │
│                                 Tumor cell recognition/attack    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Central Dogma: DNA → RNA → Protein

生命系统里最重要的信息流通常被称为中心法则：**DNA 保存长期遗传信息，RNA 是可读的工作副本，蛋白质执行结构、催化、信号与免疫识别等功能**。NHGRI 对 mRNA 的定义是：mRNA 是一种参与蛋白合成的单链 RNA，由 DNA 模板转录而来，并把蛋白质信息从细胞核带到细胞质，核糖体读取每三个碱基组成的密码子并翻译为氨基酸链。[1]

```
DNA in nucleus
  │ transcription
  ▼
mRNA transcript
  │ export to cytoplasm
  ▼
Ribosome reads codons
  │ translation
  ▼
Protein / peptide fragments
  │ processing + presentation
  ▼
Immune recognition
```

| 分子 | 类比 | 稳定性 | 在疫苗技术中的角色 |
|---|---|---:|---|
| DNA | 长期硬盘 | 高 | 储存基因组信息，肿瘤突变从这里读取 |
| mRNA | 临时指令文件 | 低到中 | 指导细胞短暂表达目标抗原 |
| 蛋白质/肽段 | 执行程序或输出结果 | 取决于蛋白 | 被加工并呈递给免疫系统 |

---

## mRNA: 临时指令，而不是永久改写

mRNA 疫苗经常被误解为会改变 DNA。对学习者来说，最关键的机制区别是：**mRNA 在细胞质中被核糖体读取，通常不进入细胞核，也不整合进基因组**。MedlinePlus Genetics 明确指出，疫苗 mRNA 不进入细胞核，也不会改变 DNA；细胞完成蛋白生产后会较快降解 mRNA。[2]

```
Correct mental model:

mRNA vaccine
   │
   ▼
Cytoplasm ──► ribosome ──► antigen protein/peptide
   │
   └── degraded over time

Not the model:

mRNA vaccine ──X──► nucleus ──X──► DNA rewrite
```

mRNA 分子本身需要工程化设计。常见设计变量包括 5' cap、UTR、开放阅读框、密码子优化、poly(A) tail、核苷修饰与纯化质量。这些设计会影响 mRNA 的稳定性、翻译效率和先天免疫刺激强度。对癌症疫苗而言，目标不是无限表达，而是**在足够时间内表达足量抗原，同时避免不可控炎症**。

| 设计部件 | 作用 | 学习重点 |
|---|---|---|
| 5' cap | 帮助核糖体识别并启动翻译 | 影响翻译效率和 RNA 稳定性 |
| UTR | 调节翻译与半衰期 | 不是编码区，但非常重要 |
| ORF | 编码抗原序列 | 个体化癌症疫苗的核心内容 |
| poly(A) tail | 提升稳定性和翻译 | 太短或太长都会影响性能 |
| 核苷修饰 | 调节免疫感知与稳定性 | 需要在表达与免疫刺激之间平衡 |

---

## 免疫系统如何识别癌细胞

癌细胞来自自身组织，因此它们不像病毒那样天然“外来”。免疫系统要识别癌细胞，通常依赖两个条件：第一，癌细胞表达了足够独特的抗原；第二，这些抗原被抗原呈递细胞或肿瘤细胞通过 MHC 分子展示给 T 细胞。CD8+ T 细胞主要识别 MHC I 上的肽段，CD4+ T 细胞主要识别 MHC II 上的肽段。

```
Protein inside cell
  │ proteasome cuts into peptides
  ▼
Peptides loaded on MHC I
  │
  ▼
Displayed on cell surface
  │
  ▼
CD8+ T cell scans peptide-MHC
  │
  ├── match: activation / killing program
  └── no match: ignore
```

| 免疫对象 | 功能 | 癌症疫苗中的角色 |
|---|---|---|
| APC（树突状细胞等） | 摄取、处理并呈递抗原 | 启动 T 细胞反应的关键细胞 |
| CD8+ T cell | 杀伤携带目标抗原的细胞 | 直接抗肿瘤效应核心 |
| CD4+ T cell | 帮助、调节、维持免疫反应 | 支持 CD8+ 和记忆反应 |
| checkpoint | 抑制过强免疫反应 | PD-1/PD-L1 抑制剂可解除刹车 |

---

## 癌症疫苗的核心：抗原与新抗原

癌症疫苗的难点是选择“该让免疫系统看见什么”。传统肿瘤相关抗原可能也在正常组织中表达，因此存在耐受性或毒性问题。**新抗原（neoantigen）**来自肿瘤突变，理论上更接近“只属于肿瘤”的标记，因此成为个体化癌症疫苗的重要方向。

| 抗原类型 | 来源 | 优点 | 风险或限制 |
|---|---|---|---|
| 肿瘤相关抗原 | 正常组织也可能低表达 | 可用于多患者通用设计 | 免疫耐受、脱靶毒性 |
| 病毒相关抗原 | HPV、EBV 等病毒相关癌症 | 外源性强，免疫原性好 | 仅适合病毒驱动癌症 |
| 突变新抗原 | 患者肿瘤特异突变 | 个体化、特异性高 | 预测难、制造链复杂 |

个体化新抗原疫苗通常需要肿瘤测序、正常组织对照、突变识别、HLA 分型、MHC 结合预测、表达证据、免疫原性排序以及最终 mRNA 序列设计。这个流程与软件工程中的“数据管线”很相似：输入是患者样本与测序数据，输出是编码候选新抗原的个体化 mRNA。

---

## mRNA 癌症疫苗如何工作

mRNA 癌症疫苗与预防性传染病疫苗不同。它多数是**治疗性疫苗**：目标是在患者已经患癌或术后存在复发风险时，提高免疫系统对肿瘤的识别能力。个体化 mRNA 疫苗通常把多个候选新抗原串联编码在同一或多条 mRNA 中，递送后由患者细胞表达、加工并呈递。

```
Patient tumor sample
  │
  ├── DNA/RNA sequencing
  │
  ├── mutation calling
  │
  ├── HLA typing
  │
  ├── neoantigen ranking
  │
  └── mRNA construct design
          │
          ▼
      mRNA vaccine
          │
          ▼
      APC uptake + translation
          │
          ▼
      peptide-MHC presentation
          │
          ▼
      T-cell priming / expansion
          │
          ▼
      tumor surveillance
```

| 步骤 | 输入 | 输出 | 主要失败模式 |
|---|---|---|---|
| 测序 | 肿瘤与正常样本 | 突变列表 | 样本质量差、肿瘤纯度低 |
| 新抗原预测 | 突变、HLA、表达 | 候选肽段 | 预测不等于真实免疫原性 |
| mRNA 设计 | 候选新抗原 | 合成 mRNA | 表达不足或免疫刺激不合适 |
| 递送 | mRNA 与纳米颗粒 | 细胞摄取 | 靶向不足、降解、炎症 |
| 免疫激活 | 抗原呈递 | T 细胞扩增 | 肿瘤微环境抑制 |
| 临床疗效 | 免疫反应 | RFS/OS 等终点 | 免疫逃逸、复发、异质性 |

---

## 递送系统：LNP、lipoplex 与 APC 靶向

裸露 mRNA 很容易被核酸酶降解，也很难高效穿过细胞膜。递送系统要解决三个问题：**保护 RNA、把 RNA 带到合适组织或细胞、帮助 RNA 从内体逃逸到胞质中被翻译**。LNP 和 RNA-lipoplex 是 mRNA 疫苗与核酸药物中最常见的递送方向之一。

2022 年 Molecular Pharmaceutics 综述指出，CRISPR/Cas9 的治疗潜力受到递送挑战限制，而 LNP 因为低免疫原性与应用灵活性，成为 CRISPR 介导基因编辑有吸引力的非病毒递送平台。[9] 同一综述还指出，非病毒载体通常更易组装、可与核酸形成稳定复合物、保护核酸免受血清核酸酶降解，并更易规模化。[9]

| 递送方式 | 常见载荷 | 优点 | 限制 |
|---|---|---|---|
| LNP | mRNA、siRNA、sgRNA/Cas mRNA | 可规模化、非病毒、保护 RNA | 组织靶向和内体逃逸仍是难点 |
| RNA-lipoplex | mRNA 疫苗 | 可用于免疫细胞靶向和免疫刺激 | 配方、剂量和反应窗口复杂 |
| 病毒载体 | DNA 或编辑器表达盒 | 转导效率高 | 免疫原性、载荷大小、长期表达风险 |
| ex vivo 细胞递送 | T cell/CAR-T 编辑 | 可质控后回输 | 制造成本高，流程复杂 |

---

## CRISPR 基因编辑基础

CRISPR 原本是细菌和古菌的免疫系统。Broad Institute 解释说，CRISPR-Cas9 可被编程以定位特定遗传代码片段并在精确位置编辑 DNA；研究者使用 guide RNA 将 Cas9 引导到目标 DNA 序列，Cas9 结合并切割 DNA。[6] Stanford 的解释强调，CRISPR 的靶向识别主要由 RNA 编码，因此更换靶点比重新设计蛋白工具容易得多，类似在 GPS 中更改目的地。[7]

```
CRISPR-Cas9 basic model:

Guide RNA:  "go to this sequence"
      │
      ▼
Cas9 protein + guide RNA complex
      │ scans DNA
      ▼
Target DNA + PAM found
      │
      ▼
Cut / nick / edit / regulate
      │
      ▼
Cell repair or expression change
```

| CRISPR 工具 | 做什么 | 与癌症治疗的关系 |
|---|---|---|
| Cas9 nuclease | 切断 DNA，诱导 NHEJ/HDR 修复 | 敲除免疫抑制基因、工程化 T 细胞 |
| Base editor | 不产生双链断裂地改写单碱基 | 精准改写突变或调控位点 |
| Prime editor | 更灵活的小片段改写 | 理论上可修复更广泛变异 |
| CRISPRi/a | 不切 DNA，抑制或激活转录 | 功能筛选、调控免疫通路 |
| Cas13 | 靶向 RNA | RNA 调控、诊断或抗病毒方向 |

---

## mRNA 疫苗与基因编辑的交叉点

mRNA 癌症疫苗与 CRISPR 基因编辑不是同一种技术，但它们共享许多底层能力：核酸设计、递送系统、免疫调控和个体化制造。mRNA 疫苗通常是**表达抗原**；CRISPR 通常是**改变基因或基因表达状态**。二者可能在癌症免疫治疗中形成组合，例如用 mRNA 疫苗训练 T 细胞识别肿瘤，再用检查点抑制剂或基因编辑细胞疗法增强 T 细胞效应。

| 技术 | 主要目的 | 是否改变基因组 | 癌症应用示例 |
|---|---|---:|---|
| mRNA 新抗原疫苗 | 表达肿瘤抗原并激活 T 细胞 | 否 | 个体化术后辅助免疫治疗 |
| CRISPR 编辑 T 细胞 | 改造免疫细胞功能 | 是，通常 ex vivo | 敲除 PD-1、增强 CAR-T/TCR-T |
| CRISPR 筛选 | 找免疫逃逸或敏感性基因 | 研究系统中是 | 发现新靶点和联合治疗机制 |
| LNP-CRISPR | 体内递送编辑器 | 可能 | 仍需解决组织靶向和安全性 |

2024 年 Exp Hematol Oncol 综述认为，CRISPR/Cas9 可通过准确修改肿瘤中的免疫细胞和肿瘤细胞，提高免疫治疗效果；应用方向包括靶向免疫检查点分子、免疫调节基因、增强抗原呈递和调控免疫细胞功能，但临床应用仍面临编辑准确性、安全性、过度免疫反应和体内递送挑战。[8]

---

## Clinical Landscape: 目前学到哪里

个体化 mRNA 癌症疫苗最值得关注的证据来自早期临床试验和正在推进的随机试验。2023 年 Nature 胰腺导管腺癌研究使用 autogene cevumeran，这是一种基于 uridine mRNA-lipoplex nanoparticles 的个体化新抗原疫苗。研究者从手术切除肿瘤实时合成 mRNA 疫苗，每位患者最多编码 20 个新抗原，并与 atezolizumab 和 mFOLFIRINOX 序贯使用；16 名患者中 8 名诱导出高强度新抗原特异性 T 细胞，18 个月中位随访时 responders 的中位无复发生存未达到，而 non-responders 为 13.4 个月。[3] [4]

Merck 与 Moderna 在 2026 年公布的 KEYNOTE-942/mRNA-4157-P201 五年随访数据中称，intismeran autogene（mRNA-4157/V940）联合 pembrolizumab 在高危 III/IV 期黑色素瘤完全切除后患者中，与 pembrolizumab 单药相比，将复发或死亡风险降低 49%（HR=0.510，95% CI 0.294–0.887）。该候选疗法由合成 mRNA 编码最多 34 个基于患者肿瘤突变特征预测的新抗原。[5]

| 项目 | 癌种 | 疫苗设计 | 联合治疗 | 关键信号 |
|---|---|---|---|---|
| Autogene cevumeran | 胰腺导管腺癌 | 最多 20 个新抗原，mRNA-lipoplex | atezolizumab + mFOLFIRINOX | 8/16 诱导高强度新抗原 T 细胞反应；responders RFS 更长 |
| mRNA-4157/V940 | 高危黑色素瘤 | 最多 34 个个体化新抗原 | pembrolizumab | 2026 五年随访称复发或死亡风险降低 49% |

这些结果令人鼓舞，但学习时要保持科学谨慎。早期试验主要证明可行性、免疫原性和安全信号；是否成为标准治疗，需要更大规模、随机、长期随访和不同癌种的验证。

---

## Algorithm Cheat Sheet

```
Personalized mRNA cancer vaccine pipeline:

1. Collect tumor + normal sample
2. Sequence DNA/RNA
3. Identify somatic mutations
4. Type patient HLA alleles
5. Predict peptide-MHC binding
6. Filter by tumor expression and clonality
7. Rank neoantigen candidates
8. Encode selected candidates into mRNA
9. Formulate with LNP/lipoplex
10. Administer with rational combination therapy
11. Measure immune response: TCR expansion, ELISpot, flow, cytokines
12. Track clinical endpoints: RFS, DMFS, OS, safety
```

| Concept | One-line Definition |
|---|---|
| Neoantigen | 肿瘤突变产生、正常组织通常不存在的免疫识别标记 |
| HLA/MHC | 把肽段展示给 T 细胞的分子展示平台 |
| APC | 负责摄取、处理和呈递抗原的免疫细胞 |
| Checkpoint inhibitor | 解除 T 细胞免疫刹车的抗体药物 |
| LNP | 保护并递送核酸的脂质纳米颗粒 |
| CRISPR | 由 guide RNA 引导 Cas 蛋白定位核酸序列的可编程系统 |

---

## References

[1]: https://www.genome.gov/genetics-glossary/Messenger-RNA-mRNA "NHGRI: Messenger RNA (mRNA)"
[2]: https://medlineplus.gov/genetics/understanding/therapy/mrnavaccines/ "MedlinePlus Genetics: What are mRNA vaccines and how do they work?"
[3]: https://www.nature.com/articles/s41586-023-06063-y "Rojas et al. Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer. Nature, 2023"
[4]: https://pubmed.ncbi.nlm.nih.gov/37165196/ "PubMed: Personalized RNA neoantigen vaccines stimulate T cells in pancreatic cancer"
[5]: https://www.merck.com/news/moderna-merck-announce-5-year-data-for-intismeran-autogene-in-combination-with-keytruda-pembrolizumab-demonstrated-sustained-improvement-in-the-primary-endpoint-of-recurrence-free-survival-i/ "Merck/Moderna: 5-year KEYNOTE-942 data for intismeran autogene plus pembrolizumab"
[6]: https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/questions-and-answers-about-crispr "Broad Institute: Questions and Answers about CRISPR"
[7]: https://news.stanford.edu/stories/2024/06/stanford-explainer-crispr-gene-editing-and-beyond "Stanford Report: What is CRISPR? A bioengineer explains"
[8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11490091/ "Feng et al. CRISPR/Cas9 technology for advancements in cancer immunotherapy. Exp Hematol Oncol, 2024"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9176214/ "Kazemian et al. Lipid-Nanoparticle-Based Delivery of CRISPR/Cas9 Genome-Editing Components. Molecular Pharmaceutics, 2022"
