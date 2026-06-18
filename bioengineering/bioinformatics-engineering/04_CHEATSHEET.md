# 生物信息学速查表 (Bioinformatics Cheatsheet)

**作者**: Manus AI

这份速查表汇总了生物信息学工程师日常工作中最常遇到的文件格式、核心工具命令以及关键术语。

---

## 1. 核心文件格式速查

在生物信息学中，数据在不同的 Pipeline 阶段以特定的格式流动。

| 格式后缀 | 全称 | 描述 | 类似 IT 概念 |
| :--- | :--- | :--- | :--- |
| **.fasta / .fa** | FASTA | 纯文本格式，存储核酸 (DNA/RNA) 或蛋白质序列。第一行以 `>` 开头为序列名称。 | 纯文本配置文件 |
| **.fastq / .fq** | FASTQ | 纯文本格式，存储测序仪下机的原始序列及其对应的**质量分数**。每条序列占 4 行。 | 带权重的日志文件 |
| **.sam** | Sequence Alignment Map | 纯文本格式，记录 Reads 比对到参考基因组上的位置信息。非常庞大。 | 详细的比对日志 |
| **.bam** | Binary Alignment Map | `.sam` 的二进制压缩版本。日常分析中只使用 BAM，几乎不直接处理 SAM。 | 压缩后的二进制日志 |
| **.vcf** | Variant Call Format | 纯文本格式，记录基因组上的变异位点 (SNP, Indel 等) 及其注释信息。 | Diff / Patch 文件 |
| **.bed** | Browser Extensible Data | 纯文本格式，定义基因组上的特定区域 (如染色体、起始位置、终止位置)。 | 区域索引文件 |
| **.gtf / .gff** | Gene Transfer Format | 纯文本格式，记录基因组特征的注释信息（如哪里是外显子，哪里是内含子）。 | 数据库 Schema 注释 |

---

## 2. 常用命令行工具速查 (CLI Tools)

### 序列处理
*   **查看 FASTQ 文件前几行**: `zcat sample.fastq.gz | head -n 8`
*   **统计 FASTA 文件中的序列数**: `grep -c "^>" reference.fasta`

### Samtools (BAM 文件处理瑞士军刀)
*   **将 SAM 转换为 BAM**: `samtools view -S -b sample.sam > sample.bam`
*   **对 BAM 文件进行排序**: `samtools sort sample.bam -o sample_sorted.bam`
*   **为排序后的 BAM 建立索引** (生成 `.bai` 文件，必须步骤): `samtools index sample_sorted.bam`
*   **查看 BAM 文件头信息**: `samtools view -H sample.bam`

### BCFtools / VCFtools (VCF 文件处理)
*   **统计 VCF 文件中的变异数量**: `bcftools stats sample.vcf`
*   **提取特定区域的变异**: `bcftools view -r chr1:1000000-2000000 sample.vcf.gz`

---

## 3. 高频术语对照表 (Jargon Buster)

| 术语 | 英文全称 | 解释 |
| :--- | :--- | :--- |
| **NGS** | Next-Generation Sequencing | 下一代测序技术（高通量测序），一次能对数百万条 DNA 分子进行测序。 |
| **WGS** | Whole Genome Sequencing | 全基因组测序，对个体的全部 DNA 进行测序。成本高，信息最全。 |
| **WES** | Whole Exome Sequencing | 全外显子组测序，仅对编码蛋白质的区域（约占基因组 1-2%）测序。性价比高。 |
| **RNA-Seq** | RNA Sequencing | 转录组测序，用于量化基因在特定条件下的表达水平。 |
| **Read** | Read | 测序仪读取出的一条短序列片段（通常 150bp 左右）。 |
| **Coverage/Depth** | Sequencing Depth | 测序深度，指基因组上某个碱基被 Reads 覆盖的平均次数（如 30X 表示平均被读取 30 次）。 |
| **Exon / Intron** | Exon / Intron | 外显子 (Exon) 是基因中编码蛋白质的部分；内含子 (Intron) 是非编码部分，在 RNA 处理中会被剪接掉。 |
| **Pipeline** | Bioinformatics Pipeline | 数据分析流水线，将多个命令行工具串联起来处理数据的自动化流程。 |

---

## 4. R 语言 / Bioconductor 快速起步

R 语言在下游统计分析和可视化中不可或缺。

```R
# 安装 Bioconductor 核心管理器
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# 安装常用的生物信息学包 (例如用于差异表达分析的 DESeq2)
BiocManager::install("DESeq2")

# 安装强大的可视化包
install.packages("ggplot2")
```
