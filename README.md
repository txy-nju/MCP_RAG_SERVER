# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG（检索增强生成）系统，通过 MCP（Model Context Protocol）协议对外暴露工具接口，支持 GitHub Copilot / Claude Desktop 等 AI 助手直接调用私有知识库。

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [MCP 工具接口](#mcp-工具接口)
- [Dashboard 功能页](#dashboard-功能页)
- [可支持的 Provider](#可支持的-provider)
- [脚本使用](#脚本使用)
- [测试](#测试)
- [Agent Skills](#agent-skills)
- [分支说明](#分支说明)

---

## 项目简介

本项目围绕 RAG（Retrieval-Augmented Generation）核心链路构建，将检索、生成、评估、可观测性等关键环节串联成一个完整的、可运行的工程系统。

**核心理念：全链路可插拔。** LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每个环节均定义了抽象接口，通过 `config/settings.yaml` 一键切换后端，零代码修改。

> 详细架构设计与任务排期请参阅 [DEV_SPEC.md](DEV_SPEC.md)

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      MCP Clients                          │
│         (GitHub Copilot / Claude Desktop / Agent)         │
└────────────────────────┬─────────────────────────────────┘
                         │  MCP Protocol (stdio)
┌────────────────────────▼─────────────────────────────────┐
│                    MCP Server                             │
│  query_knowledge_hub | list_collections | get_doc_summary │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                   Query Engine                            │
│  Query Processor → Hybrid Search → RRF Fusion → Rerank   │
│                       │                  │                │
│              Dense Retrieval      Sparse (BM25)           │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                 Storage Layer                             │
│      ChromaDB (向量)  │  BM25 Index  │  SQLite (历史)    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│               Ingestion Pipeline                          │
│  PDF Loader → Splitter → Transform → Embedding → Upsert  │
│        (Chunk Refiner / Metadata Enricher / Image Cap.)   │
└──────────────────────────────────────────────────────────┘
```

---

## 核心功能

### 数据摄取（Ingestion Pipeline）

| 阶段 | 说明 |
|------|------|
| **PDF Loader** | 解析 PDF 为规范化 Markdown，提取页码、标题、图片引用等 metadata |
| **Splitter** | 基于 Markdown 结构语义分块（RecursiveCharacterTextSplitter），保留上下文 |
| **Chunk Refiner** | 可选 LLM 对 Chunk 进行语义润色与去噪 |
| **Metadata Enricher** | 可选 LLM 自动生成标签、摘要等 metadata 字段 |
| **Image Captioner** | 可选 Vision LLM 描述图片并缝合进 Chunk 文本（Image-to-Text 策略，实现"搜文字出图"） |
| **Embedding & Upsert** | 批量向量化后幂等 upsert 至 ChromaDB，同步构建 BM25 索引 |
| **File Dedup** | SHA-256 文件指纹去重，相同文件跳过重复摄取 |

### 混合检索（Hybrid Search）

- **Dense Retrieval**：基于 Embedding 向量的语义检索（ChromaDB）
- **Sparse Retrieval**：基于 BM25 的关键词精确匹配
- **RRF Fusion**：Reciprocal Rank Fusion 融合两路召回结果
- **Rerank**：Cross-Encoder 或 LLM Reranker 对候选集精排（可选）

### MCP 服务

遵循 Model Context Protocol 标准，以 stdio 方式暴露 3 个工具，可直接被 Copilot / Claude 调用。

### Streamlit Dashboard

6 个功能页面，提供全链路可视化管理与追踪能力，更换 Provider 后自动适配，无需修改代码。

### 评估体系

集成 Ragas + Custom Evaluator，支持 Golden Test Set 回归测试，量化检索与生成质量。

### 全链路可观测

Ingestion 与 Query 两条链路的每个中间状态均记录至 `logs/traces.jsonl`，Dashboard 同步展示。

---

## 项目结构

```
├── config/
│   ├── settings.yaml          # 主配置文件（Provider / 参数 / 路径）
│   └── prompts/               # LLM Prompt 模板
├── data/
│   ├── db/chroma/             # ChromaDB 向量存储
│   ├── db/bm25/               # BM25 索引
│   └── documents/             # 待摄取文档目录（放置 PDF）
├── logs/
│   └── traces.jsonl           # 全链路追踪日志
├── scripts/
│   ├── ingest.py              # 文档摄取脚本
│   ├── query.py               # 命令行查询脚本
│   ├── evaluate.py            # 评估脚本
│   └── start_dashboard.py     # Dashboard 启动脚本
├── src/
│   ├── core/                  # 核心类型与设置
│   │   ├── settings.py
│   │   ├── types.py
│   │   ├── query_engine/      # 查询引擎（混合检索 + 响应构建）
│   │   └── trace/             # 追踪记录
│   ├── ingestion/             # 摄取流水线
│   │   ├── pipeline.py
│   │   ├── document_manager.py
│   │   └── transform/         # Chunk Refiner / Metadata Enricher / Image Captioner
│   ├── libs/                  # 可插拔组件库（工厂模式）
│   │   ├── llm/               # OpenAI / Azure / DeepSeek / Ollama + Vision LLM
│   │   ├── embedding/         # OpenAI / Azure / Ollama
│   │   ├── reranker/          # Cross-Encoder / LLM Reranker
│   │   ├── splitter/          # Recursive Splitter
│   │   ├── vector_store/      # ChromaDB
│   │   ├── loader/            # PDF Loader
│   │   └── evaluator/         # Ragas / Custom Evaluator
│   ├── mcp_server/            # MCP 服务器
│   │   ├── server.py
│   │   ├── protocol_handler.py
│   │   └── tools/             # query_knowledge_hub / list_collections / get_document_summary
│   └── observability/
│       ├── logger.py
│       ├── dashboard/         # Streamlit Dashboard（6 页面）
│       └── evaluation/
├── tests/
│   ├── unit/                  # 单元测试（无外部依赖）
│   ├── integration/           # 集成测试
│   └── e2e/                   # 端到端测试
├── main.py                    # MCP Server 入口
├── pyproject.toml
└── requirements.txt
```

---

## 快速开始

### 方式一：Setup Skill（推荐）

在 VS Code 中打开项目，通过 Copilot / Claude 对话框输入：

```
setup
```

Agent 会自动引导完成：Provider 选择 → API Key 配置 → 依赖安装 → 配置文件生成 → Dashboard 启动。

### 方式二：手动配置

**1. 创建虚拟环境并安装依赖**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

**2. 配置 `config/settings.yaml`**（填入 API Key 并选择 Provider）

**3. 摄取文档**

将 PDF 文件放入 `data/documents/`，然后运行：

```bash
python scripts/ingest.py
```

**4. 启动 Dashboard**

```bash
python scripts/start_dashboard.py
# 访问 http://localhost:8501
```

**5. 启动 MCP Server**

```bash
python main.py
```

---

## 配置说明

`config/settings.yaml` 完整示例：

```yaml
llm:
  provider: openai          # openai | azure | deepseek | ollama
  model: gpt-4o-mini
  api_key: your-api-key
  api_url: your-proxy-url   # 可选，代理 / 自定义端点

embedding:
  provider: openai          # openai | azure | ollama
  model: text-embedding-3-small
  api_key: your-api-key

splitter:
  provider: recursive
  chunk_size: 1000
  chunk_overlap: 200

vector_store:
  provider: chroma
  collection: default
  persist_path: data/db/chroma

retrieval:
  top_k: 5

rerank:
  provider: none            # none | cross_encoder | llm
  max_candidates: 20

evaluation:
  backend: custom           # custom | ragas

ingestion:
  chunk_refiner:
    use_llm: false          # true：调用 LLM 润色 Chunk
  metadata_enricher:
    use_llm: false          # true：调用 LLM 生成 metadata 标签
    max_tags: 5
  image_captioner:
    use_vision_llm: false   # true：调用 Vision LLM 描述图片
```

---

## MCP 工具接口

遵循 [Model Context Protocol](https://modelcontextprotocol.io) 标准，暴露以下 3 个工具：

| 工具名 | 说明 |
|--------|------|
| `query_knowledge_hub` | 对知识库进行混合检索，返回最相关的文档片段 |
| `list_collections` | 列出当前所有已索引的文档集合 |
| `get_document_summary` | 获取指定文档的摘要与元数据 |

**在 VS Code 中注册**（`settings.json`）：

```json
{
  "github.copilot.chat.mcp.servers": {
    "modular-rag": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/path/to/MODULAR-RAG-MCP-SERVER"
    }
  }
}
```

---

## Dashboard 功能页

访问 `http://localhost:8501`：

| 页面 | 功能 |
|------|------|
| **系统总览** | 查看当前可插拔组件配置（LLM / Embedding / Splitter / Reranker）与数据资产统计 |
| **数据浏览器** | 浏览已索引文档列表、Chunk 详情（原文 / metadata / 关联图片），支持搜索过滤 |
| **Ingestion 管理** | 通过界面上传文件触发摄取、展示各阶段实时进度、删除已摄取文档 |
| **Ingestion 追踪** | 摄取历史列表，各阶段耗时与处理详情 |
| **Query 追踪** | 查询历史，耗时瀑布图，Dense/Sparse 召回对比，Rerank 前后排名变化 |
| **评估面板** | 运行评估任务、查看 Faithfulness / Relevancy / Hit Rate 等指标，历史趋势对比 |

---

## 可支持的 Provider

修改 `config/settings.yaml` 中对应的 `provider` 字段即可切换，无需修改代码。

**LLM**

| Provider | 说明 |
|----------|------|
| `openai` | OpenAI API（含兼容格式的第三方代理） |
| `azure` | Azure OpenAI Service |
| `deepseek` | DeepSeek API |
| `ollama` | 本地 Ollama 部署 |

**Embedding**

| Provider | 说明 |
|----------|------|
| `openai` | OpenAI Embedding API |
| `azure` | Azure OpenAI Embedding |
| `ollama` | 本地 Ollama 模型 |

**Reranker**

| Provider | 说明 |
|----------|------|
| `none` | 跳过重排，直接使用 RRF Fusion 结果 |
| `cross_encoder` | Sentence-Transformers Cross-Encoder 本地模型 |
| `llm` | 调用已配置的 LLM 后端进行重排 |

> **扩展新 Provider**：在 `src/libs/<模块>/` 下新建继承基类的 Provider 类 → 在对应 Factory 中注册 → 更新 `settings.yaml`。可直接让 AI（`setup` 或直接对话）完成这三步。

---

## 脚本使用

```bash
# 摄取 data/documents/ 下的所有 PDF
python scripts/ingest.py

# 命令行查询
python scripts/query.py --query "你的问题"
python scripts/query.py --query "你的问题" --collection my_collection

# 运行评估
python scripts/evaluate.py

# 启动 Dashboard（http://localhost:8501）
python scripts/start_dashboard.py

# 启动 MCP Server
python main.py
```

---

## 测试

```bash
# 运行所有测试
pytest

# 按层级运行
pytest -m unit          # 单元测试（无外部依赖，运行最快）
pytest -m integration   # 集成测试（需要配置有效的 Provider）
pytest -m e2e           # 端到端测试
```

| 层级 | 路径 | 说明 |
|------|------|------|
| **Unit** | `tests/unit/` | 独立模块逻辑，无外部依赖 |
| **Integration** | `tests/integration/` | 模块间交互、存储读写 |
| **E2E** | `tests/e2e/` | 完整摄取链路与查询链路 |

---

## Agent Skills

在 VS Code Copilot / Claude 对话框输入对应关键词即可激活：

| Skill | 触发词 | 功能 |
|-------|--------|------|
| **setup** | `setup` / `初始化` | 交互式环境配置向导：Provider 选择 → API Key → 安装依赖 → 启动 |
| **auto-coder** | `auto code` / `自动开发` | 读取 DEV_SPEC，识别下一个待开发任务，自动生成代码（含测试修复） |
| **qa-tester** | `QA test` / `跑测试` | 全自动执行 QA 测试计划，失败自动修复（最多 3 轮），记录结果 |
| **package** | `package` / `打包` | 清理缓存、日志、API Key，生成可分发代码包 |
| **resume-writer** | `写简历` / `resume` | 采集岗位与背景信息，生成四段式定制化中英文简历项目经历 |
| **interview-prep** | `模拟面试` / `mock interview` | 围绕项目进行最多 3 轮深度技术追问，生成面试报告与参考答案 |
| **project-learner** | `学习项目` / `learn project` | 问答驱动的项目学习，覆盖 10 个知识域 × 45 个知识点 |
| **project-review** | `复习项目` / `review project` | 按章节带领系统复习，记录掌握进度，支持断点续学 |

---

## 分支说明

| 分支 | 说明 | 适用场景 |
|------|------|---------|
| `main` | 最新完整代码，仅 1 个 commit | 快速体验 / 直接使用 / 二次扩展 |
| `dev` | 与 main 代码一致，保留完整 commit 历史 | 回溯开发思路，了解项目构建过程 |
| `clean-start` | 仅含 DEV_SPEC + Skills，代码清零 | 从零体验完整工作流（强烈推荐） |
