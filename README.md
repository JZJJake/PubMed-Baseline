# PubMed Intelligent Literature Assistant

PubMed 智能文献助手 (PubMed-Baseline) 是一个基于 AI 和语义检索的本地化文献知识库系统。它集成了从数据采集、解析、向量化索引到智能问答的全流程功能，旨在帮助研究人员高效地利用海量生物医学文献。

## 核心功能

### 1. 数据同步 (Sync)
- **功能**: 自动从 NCBI FTP 服务器下载最新的 PubMed XML 数据文件 (`.xml.gz`)。
- **特点**: 支持断点续传、并发下载和进度显示。

### 2. 数据解析 (Parse)
- **功能**: 将下载的复杂 XML 结构解析为高效的 JSONL 格式 (`metadata.jsonl`)。
- **提取字段**: PMID、标题、摘要、作者列表、期刊名称、发表年份、DOI 等。

### 3. 语义索引 (Index)
- **功能**: 构建本地向量数据库，支持基于语义的文献检索。
- **技术栈**:
    - **Embeddings**: 使用 `SentenceTransformers` (`all-MiniLM-L6-v2`) 将文献标题和摘要转化为高维向量。
    - **Vector DB**: 使用 `ChromaDB` 进行存储和相似度搜索。
- **特点**:
    - **GPU 加速**: 自动检测 CUDA 环境，大幅提升向量生成速度。
    - **增量更新**: 支持从上次中断处继续索引，无需重新开始。
    - **数据重置**: 提供 `--reset` 选项以应对数据库损坏的情况。

### 4. 智能检索 (Search)
- **功能**: 提供两种检索模式：
    - **关键词匹配**: 传统的精确文本匹配。
    - **语义检索 (`-v`)**: 基于向量相似度，能够发现用词不同但语义相关的文献（例如搜索 "heart attack" 可匹配 "myocardial infarction"）。

### 5. AI 问答 (Ask / RAG)
- **功能**: 基于检索增强生成 (RAG) 技术，利用 DeepSeek AI 回答专业问题。
- **流程**:
    1. 用户提问。
    2. 系统在本地向量库中检索相关文献摘要。
    3. 将文献作为上下文提供给 AI。
    4. AI 生成基于证据的回答，并标注引用来源。

## 安装与配置

### 环境要求
- Python 3.8+
- 推荐使用 NVIDIA GPU 以加速索引过程。

### 安装依赖
```bash
pip install -r requirements.txt
```

### GPU 加速设置 (重要)
默认安装的 `torch` 可能是 CPU 版本。若要启用 GPU 加速，请先卸载当前版本，并根据您的 CUDA 版本安装对应的 PyTorch。

**检查当前状态**:
```bash
python src/check_gpu.py
```

**重新安装 PyTorch (示例: CUDA 11.8)**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您环境的安装命令)*

### 配置 API Key
在项目根目录创建 `.env` 文件或使用命令行配置 DeepSeek API Key：
```bash
config DEEPSEEK_API_KEY sk-your-api-key
```

## 使用指南

启动命令行工具：
```bash
python main.py
```

常用命令：
- `sync 5`: 下载前 5 个数据文件。
- `parse`: 解析已下载的文件。
- `index`: 构建向量索引。
- `search "lung cancer" -v`: 语义搜索肺癌相关文献。
- `ask "最新的肺癌免疫疗法有哪些进展？"`: 向 AI 提问。
- `help`: 查看更多命令帮助。
