# Unity RAG Agent

> 面向 Unity 工程的本地 AI 编程助手 — An AI coding assistant for Unity projects

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-orange)](https://github.com/langchain-ai/langgraph)

> 该项目含有大量关于vibe coding的测试，难免疏漏。请注意保护个人信息！（KEY/工程信息）

对 Unity 工程建立多域语义索引，通过对话式 Agent 辅助代码检索、问答与文件修改。

**🆕 v2.0 新特性：**
- ✨ **流式输出**：实时显示 LLM 生成和工具执行进度
- 🛡️ **人工审批**：危险操作前确认，支持 YOLO 模式
- ⏮️ **状态回溯**：检查点系统，支持对话回滚
- 🧠 **ReAct 模式**：显式推理 → 行动 → 观察循环

---

## 功能特性

### 核心功能
- **多域 RAG 索引**：`code / scene / prefab / asset / audio / image` 六域分别建立向量索引，检索时可按域过滤
- **C# 语法感知分块**：tree-sitter 解析 AST，以类 / 方法为边界切块，保证代码片段语义完整
- **Unity YAML 解析**：Scene、Prefab、Asset 按逻辑对象摘要切块，而非裸 YAML 文本
- **双 LLM Provider**：支持 Claude（含 Prompt Caching）和 OpenAI / 兼容接口，运行时可切换
- **完全离线 Embedding**：使用本地 sentence-transformers 模型，不依赖 embedding API，代码不出本地

### LangGraph 增强功能 (v2.0)
- **流式输出**：
  - Token 级别实时显示
  - 工具执行进度提示
  - CLI 和 Dashboard 双端支持
  
- **人工审批系统**：
  - 可配置审批规则（per-project）
  - 三种模式：auto / batch / yolo
  - 危险操作默认需要确认（run_command, rag_rebuild）
  - 支持"批量审批"和"一次批准后自动执行"
  
- **状态回溯**：
  - SQLite 检查点持久化
  - `/rollback [steps]` 命令回滚对话
  - `/history` 查看检查点历史
  - 支持从任意断点恢复
  
- **ReAct 模式**：
  - 显式推理过程
  - 工具调用追踪
  - 观察结果整合

### 其他功能
- **Tool Use Agent**：LLM 可自主调用 12 个沙盒工具（读写文件、搜索代码、执行命令、RAG 索引管理）
- **多工程管理**：同时注册多个 Unity 工程，向量数据库按工程隔离
- **交互式 CLI**：Rich 终端界面，内置检索、对话、文件操作等 30+ 指令
- **Web Dashboard**：Flask 驱动的浏览器界面，支持聊天、检索面板、文件编辑、AI Draft
- **对话持久化**：多对话框管理，按工程隔离存储，记录 token 消耗统计
- **文件监听**：`watch` 模式监听工程文件变更，自动增量更新索引

---

## 目录

- [前置要求](#前置要求)
- [安装](#安装)
- [下载 Embedding 模型](#下载-embedding-模型)
- [快速开始](#快速开始)
- [使用界面](#使用界面)
  - [CLI](#cli-用法)
  - [Web Dashboard](#dashboard-用法)
- [索引管理](#索引管理)
- [配置参考](#配置参考)
- [目录结构](#目录结构)
- [技术架构](#技术架构)
- [已知限制](#已知限制)
- [许可证](#许可证)

---

## 前置要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | **3.10+** | 必须 |
| pip | 最新版 | 用于安装依赖 |
| Claude API Key | — | 使用 Claude 时必须；[申请地址](https://console.anthropic.com/) |
| OpenAI API Key | — | 使用 OpenAI 或兼容接口时必须 |
| 磁盘空间 | **约 3 GB** | Embedding 模型文件（见下方说明） |

> **注意**：Embedding 模型文件体积较大（jina-code ~1.5 GB，bge-m3 ~570 MB），**不包含在本仓库中**，需要单独下载（见[下载 Embedding 模型](#下载-embedding-模型)）。

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/rag-unity.git
cd rag-unity

# 2. 安装依赖
pip install -r requirements.txt

# 3. 复制配置文件
cp .env.example .env
```

或使用一键安装向导（会自动安装依赖、引导填写配置、注册工程）：

```bash
python main.py setup
```

---

## 下载 Embedding 模型

项目使用两个本地 Embedding 模型，需在首次使用前下载到 `models/` 目录：

### 方式一：通过 huggingface-hub 下载（推荐）

```bash
pip install huggingface-hub

# C# 代码专用 Embedding
python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v2-base-code', local_dir='models/jina-code-embeddings-1.5b')"

# 多语言通用 Embedding（用于 Scene/Prefab/Asset/Audio/Image）
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
```

### 方式二：手动下载

从以下地址下载并放到对应目录：

| 模型 | 地址 | 目标目录 |
|------|------|---------|
| jina-embeddings-v2-base-code | [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) | `models/jina-code-embeddings-1.5b/` |
| bge-m3 | [HuggingFace](https://huggingface.co/BAAI/bge-m3) | `models/bge-m3/` |

> 国内网络可通过 `HF_ENDPOINT=https://hf-mirror.com` 加速下载。

---

## 快速开始

> 请在开始前确认已有可用的embedding模型。

### 1. 配置 API Key

编辑 `.env`，填入你的 LLM API Key：

```env
LLM_PROVIDER=claude          # 使用 claude 或 openai
ANTHROPIC_API_KEY=sk-ant-xxx # Claude 用户填此项
OPENAI_API_KEY=sk-xxx        # OpenAI 用户填此项
```

### 2. 注册 Unity 工程并建立索引

```bash
python main.py index --add /path/to/YourUnityProject --name MyGame
```

建立索引约需 1–5 分钟（取决于工程体量）。

### 3. 启动

```bash
# Web Dashboard（默认，推荐新用户）
python main.py

# 或 CLI 交互模式
python main.py cli

# 或单条问题直接返回
python main.py cli -q "玩家跳跃逻辑在哪"
```

---

## 使用界面

### CLI 用法

```bash
python main.py cli                        # 进入交互式对话
python main.py cli --project MyGame       # 指定工程
python main.py cli --provider openai      # 指定 Provider
python main.py cli -q "问题"              # 单条问答后退出
python main.py cli --search "关键词"      # 仅做向量检索，不调用 LLM
```

#### 交互式指令速查

**��础**

| 指令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/status` | 当前工程和模型状态 |
| `/topk <N>` | 设置检索数量 |
| `/model` | 查看 / 切换模型 |
| `/provider [claude\|openai]` | 切换 LLM Provider |

**LangGraph 新增命令**

| 指令 | 说明 |
|------|------|
| `/rollback [steps]` | 回滚对话 N 步（默认 1） |
| `/history` | 查看检查点历史 |
| `/approval` | 查看审批配置 |
| `/yolo` | 关于 YOLO 模式的说明 |

**工程管理**

| 指令 | 说明 |
|------|------|
| `/projects` | 列出所有已注册工程 |
| `/switch <name-or-id>` | 切换活跃工程 |
| `/add <path> [name]` | 注册新工程 |
| `/remove <name-or-id>` | 移除工程 |
| `/index` | 全量重建��引 |
| `/update` | 增量更新索引 |

**对话管理**

| 指令 | 说明 |
|------|------|
| `/new` | 新建对话 |
| `/conversations` | 列出历史对话 |
| `/load <id>` | 加载历史对话 |
| `/delete <id>` | 删除对话 |
| `/rename <id> <title>` | 重命名对话 |
| `/reset` | 清空当前对话 |

**检索**

| 指令 | 说明 |
|------|------|
| `/search <query> [--domain code] [--path Assets/UI] [--topk 12]` | 向量检索（带过滤） |
| `/code <query>` | 仅搜索代码域 |
| `/scene <query>` | 仅搜索场景域 |
| `/prefab <query>` | 仅搜索 Prefab 域 |
| `/classes` | 列出所有已索引的类 |
| `/class <class-name>` | 查看指定类的索引片段 |
| `/scenes` | 列出所有场景 |

**开发工具**

| 指令 | 说明 |
|------|------|
| `/read <path> [max_lines]` | 读取工程内文件 |
| `/ls [path] [pattern]` | 列出目录 |
| `/grep <query> [path]` | 在文件中搜索文本 |
| `/run <command>` | 在工程目录执行命令 |

**Skill**

| 指令 | 说明 |
|------|------|
| `/skills` | 列出已安装的 Skill |
| `/skill <name>` | 查看 Skill 详情 |

---

### Dashboard 用法

```bash
python main.py dashboard
# 默认访问 http://127.0.0.1:5000
```

**主要功能模块**

#### 聊天
- 下拉选择工程，在输入框提问
- 可选筛选条件：`path`（文件路径前缀）、`domain`（代码域）、`top_k`（检索数量）
- Agent 会自动读写文件、搜索代码，并流式返回结果

#### 检索面板
- 不调用 LLM，直接跑向量检索
- 支持 `query / domain / path / top_k` 多维过滤
- 适合验证索引效果、定位文件

#### 文件编辑器
1. 在文件浏览器中打开文件
2. 手动修改内容
3. 点击「预览 Diff」查看差异
4. 确认无误后点「保存」

#### AI Draft
1. 打开目标文件
2. 用自然语言描述修改意图（如"给 OnDamage 加一个无敌帧判断"）
3. 点击 `AI Draft` → 草案进入编辑器和 Draft Queue
4. 逐条 `Apply / Discard` 或批量 `Apply All / Discard All`

---

## 索引管理

```bash
# 全量重建（修改了 chunk 逻辑 / embedding 模型时必须执行）
python main.py index --project MyGame

# 增量更新（仅重新索引变更的文件）
python main.py index --update --project MyGame

# 查看索引统计
python main.py index --stats --project MyGame

# 列出所有已注册工程
python main.py index --list

# 注册新工程
python main.py index --add /path/to/project --name MyGame
```

**建议全量重建的时机**：
- 修改了 chunk 策略（`core/chunker.py`）
- 更换或更新了 embedding 模型
- 从旧版本升级
- 大规模批量修改工程文件

**文件监听模式**（开发时后台运行，自动增量更新）：

```bash
python main.py watch --project MyGame
```

---

## 配置参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | `claude` 或 `openai` | `claude` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | — |
| `CLAUDE_MODEL` | Claude 模型 ID | `claude-sonnet-4-6` |
| `OPENAI_API_KEY` | OpenAI API Key | — |
| `OPENAI_MODEL` | OpenAI 模型 ID | `gpt-4o` |
| `OPENAI_BASE_URL` | 自定义 API 地址（Azure、DeepSeek、本地模型等） | 空（使用 OpenAI 官方） |
| `PROJECTS_DB_PATH` | 工程注册表路径 | `./projects.json` |
| `CHROMA_DB_ROOT` | 向量数据库根目录 | `./rag_databases` |
| `RAG_TOP_K` | 默认检索返回数量 | `6` |
| `CHUNK_MAX_TOKENS` | 单块最大 token 估算值 | `600` |
| `UNITY_VERSION` | 默认 Unity 版本（注入 system prompt） | `2022.3 LTS` |
| `RENDER_PIPELINE` | 默认渲染管线 | `URP` |
| `DASHBOARD_HOST` | Dashboard 监听地址 | `127.0.0.1` |
| `DASHBOARD_PORT` | Dashboard 端口 | `5000` |

---

## 目录结构

```text
rag-unity/
├── main.py                  # 统一入口（cli / dashboard / index / watch / setup）
├── requirements.txt
├── .env.example             # 配置模板，复制为 .env 后填写
├── .env                     # 实际配置（不提交到 Git）
├── projects.json            # 工程注册表（运行时生成，不提交到 Git）
│
├── core/
│   ├── config.py            # 全局配置，读取 .env
│   ├── projects.py          # 多工程管理
│   ├── chunker.py           # 文件分块策略（C# / YAML / 音频 / 图片）
│   ├── conversations.py     # 对话持久化管理
│   ├── tools.py             # Agent 工具定义与执行（沙盒化）
│   └── skills.py            # Claude Code Skill 加载器
│
├── rag/
│   ├── indexer.py           # 索引构建（全量 / 增量）
│   ├── retriever.py         # 多域检索，支持 domain / path 过滤
│   └── watcher.py           # 文件系统监听，触发自动更新
│
├── llm/
│   ├── agent.py             # UnityAgent：Tool Use 循环 + RAG 集成
│   └── providers.py         # LLM 抽象层（ClaudeProvider / OpenAIProvider）
│
├── ui/
│   ├── cli.py               # Rich 终端交互界面
│   └── dashboard.py         # Flask Web Dashboard
│
├── models/                  # Embedding 模型（需手动下载，不提交到 Git）
│   ├── jina-code-embeddings-1.5b/   # 用于 code 域
│   └── bge-m3/                      # 用于其他域
│
├── rag_databases/           # Chroma 向量数据库（运行时生成，不提交到 Git）
└── conversations/           # 对话历史（运行时生成，不提交到 Git）
```

---

## 技术架构

### Embedding 路由策略

| 文件类型 | 扩展名 | 域 | Embedding 模型 |
|---------|--------|-----|---------------|
| C# 脚本 | `.cs` | `code` | jina-code-embeddings-1.5b |
| 场景 | `.unity` | `scene` | bge-m3 |
| Prefab | `.prefab` | `prefab` | bge-m3 |
| Asset | `.asset`, `.spriteatlas` | `asset` | bge-m3 |
| 音频（Wwise） | `.wwu`, `.bnk`, `.wem`, `.wav`, `.ogg`, `.mp3` | `audio` | bge-m3（名称/路径索引） |
| 图片 | `.png`, `.jpg`, `.tga`, `.psd`, `.exr` | `image` | bge-m3（名称/路径索引） |

### Agent Tool Use 循环

```
用户输入
  ↓
RAG 检索（相关代码片段注入上下文）
  ↓
循环（最多 10 轮）：
  ├─ 调用 LLM（Claude / OpenAI）
  ├─ 有工具调用 → 沙盒执行 → 结果追加消息 → 继续
  └─ 无工具调用 → 返回最终答案
  ↓
自动保存对话
```

### 12 个内置工具

| 类别 | 工具 | 说明 |
|------|------|------|
| 文件 | `read_file` | 读取文件（前 500 行） |
| 文件 | `write_file` | 创建 / 覆盖文件 |
| 文件 | `replace_in_file` | 精确文本替换 |
| 文件 | `diff_file` | 生成 unified diff |
| 探索 | `list_files` | 列出目录 |
| 探索 | `search_code` | 正则 / 文本搜索 |
| 系统 | `run_command` | 执行 Shell 命令（cwd 锁定为工程目录） |
| RAG | `rag_search` | 语义向量检索 |
| RAG | `rag_update` | 增量更新索引 |
| RAG | `rag_rebuild` | 全量重建索引 |
| RAG | `rag_stats` | 查看索引统计 |
| Skill | `use_skill` | 调用已安装的 Skill |

所有文件操作均通过沙盒路径校验，限定在工程目录内，防止路径遍历。

---

## 已知限制

- **图片 / 音频**：目前以文件名、路径、引用关系为主要索引维度，不做真正的视觉 / 音频语义 embedding
- **AI Draft**：单文件草案模式，不支持跨多文件的事务式修改
- **Draft Queue / Tasks**：当前为内存状态，重启 Dashboard 后不保留

### RAG 效果调优建议

如果检索效果不理想，可以从以下方向入手：

1. 调整 `domain / path / top_k` 过滤条件
2. 缩小索引范围，在 `config.py` 的 `IGNORED_DIRS` 中排除无关目录（如 Plugins、ThirdParty）
3. 调整 `CHUNK_MAX_TOKENS` 控制分块粒度
4. 优化 Scene / Prefab 的 chunk 结构（`core/chunker.py`）
5. 为图片增加 OCR / SpriteAtlas / 被引用关系索引

---

## 常见问题

**Q: 第一次启动很慢？**
A: sentence-transformers 首次加载本地模型需要数十秒，属于正常现象，后续启动会快很多。

**Q: 如何接入 DeepSeek / 国内兼容接口？**
A: 将 `LLM_PROVIDER` 设为 `openai`，并设置 `OPENAI_BASE_URL` 和对应的 Key 即可。例如：
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-deepseek-key
OPENAI_MODEL=deepseek-chat
OPENAI_BASE_URL=https://api.deepseek.com
```

**Q: 修改了 embedding 模型后需要重建索引吗？**
A: 是的，必须全量重建。新旧向量维度 / 空间不同，混合使用会导致检索结果不正确。

**Q: 多人协作时可以共享索引吗？**
A: 索引数据库（`rag_databases/`）是本地二进制格式，理论上可以打包分发，但不建议提交到 Git（体积大、平台相关）。建议每人在本地自行建立索引。

**Q: 如何新增 Skill？**
A: 参考 Claude Code Skill 规范，将 Skill 定义文件放到对应目录，重启后即可在 CLI 用 `/skills` 查看。

**Q: 如何配置审批规则？**
A: 编辑 `projects.json` 中的 `approval_config` 字段：
```json
{
  "project_id": "abc123",
  "approval_config": {
    "mode": "batch",
    "tools": {
      "run_command": true,
      "rag_rebuild": true,
      "write_file": false
    }
  }
}
```

**Q: YOLO 模式是什么？**
A: YOLO (You Only Live Once) 模式是一种审批模式，在第一次审批时选择 "yolo"，之后该会话中的所有工具调用都会自动批准，无需再次确认。适合信任 Agent 的场景。

**Q: 如何回滚错误的操作？**
A: 使用 `/rollback [steps]` 命令回滚对话。例如 `/rollback 2` 回滚 2 步。使用 `/history` 查看可用的检查点。

**Q: 流式输出不工作？**
A: 流式输出需要 LangGraph agent。确保使用的是 `UnityLangGraphAgent` 而非旧的 `UnityAgent`。

---

## 升级到 v2.0 (LangGraph)

### 自动迁移

运行迁移脚本：
```bash
python migrate_conversations.py --all
```

或交互式选择：
```bash
python migrate_conversations.py
```

### 手动迁移

现有对话格式已兼容，无需手动迁移。LangGraph 会在首次使用时自动创建检查点。

### 配置更新

在 `projects.json` 中为每个项目添加审批配置（可选）：
```json
{
  "projects": [
    {
      "project_id": "abc123",
      "name": "MyGame",
      "path": "/path/to/project",
      "approval_config": {
        "mode": "batch",
        "tools": {
          "run_command": true,
          "rag_rebuild": true
        }
      }
    }
  ]
}
```

---

## 技术架构 (v2.0)

### LangGraph 状态图

```
[START]
   ↓
[RAG Retrieval] ← 语义检索，注入上下文
   ↓
[LLM Call] ← 调用 Claude/OpenAI（支持流式）
   ↓
[Has Tool Calls?]
   ↓           ↓
[Yes]       [No] → [Format Output] → [END]
   ↓
[Approval Gate] ← 人工审批（可配置）
   ↓
[Tool Execution] ← 执行工具，记录结果
   ↓
[Should Continue?]
   ↓           ↓
[Yes]       [No]
   ↓           ↓
[LLM Call]  [Format Output] → [END]
```

### 检查点系统

- **存储**: SQLite (per-project)
- **位置**: `{project.db_path}/langgraph_checkpoints.db`
- **内容**: 完整状态快照（messages, tool_results, metadata）
- **清理**: 自动保留最近 50 个检查点

### 审批系统

- **配置级别**: Per-project
- **模式**:
  - `auto`: 无审批
  - `batch`: 批量审批（默认）
  - `yolo`: 一次批准后自动执行
- **默认危险工具**: `run_command`, `rag_rebuild`

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。

本项目为本地开发辅助工具，建议在受控环境下使用，尤其是启用文件写入和命令执行能力时，请确保对 Agent 操作范围有所了解。
