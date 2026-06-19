# PorscheAgent — 项目架构文档

> 极简 LLM Agent 框架，支持 DeepSeek / OpenAI 兼容提供商，含三层记忆管理。

## 1. 项目目录结构

```
PorscheAgent/
├── porsche_agent/                # Python 主包（11 个源文件）
│   ├── __init__.py               # 公开 API 导出
│   ├── agent.py                  # Agent 核心：ReAct 循环 + 消息工厂函数
│   ├── llm.py                    # LLM 提供商抽象层（DeepSeek / OpenAI）
│   ├── tools.py                  # @tool 装饰器 + Python 类型 → JSON Schema
│   ├── builtin_tools.py          # 内置工具：shell_command / read_file / write_file
│   ├── config.py                 # 配置数据类，从环境变量读取
│   ├── cli.py                    # CLI 入口：porsche chat / porsche run
│   └── memory/                   # 记忆子系统（4 个文件）
│       ├── __init__.py           # 记忆模块公开导出
│       ├── context.py            # 上下文管理器：滑动窗口 + LLM 摘要
│       ├── short_term.py         # 短期记忆：会话级 KV 存储 + TTL
│       └── long_term.py          # 长期记忆：语义检索 Python 客户端
│
├── vectorstore/                  # Go 向量存储服务（3 个文件）
│   ├── go.mod                    # Go 模块定义
│   ├── main.go                   # HTTP 服务器（6 个端点）
│   └── store/
│       └── store.go              # 向量索引 + 余弦相似度搜索 + JSON 持久化
│
├── tests/                        # 单元测试（42 个用例）
│   ├── __init__.py               # 空文件，使 tests 成为包
│   ├── test_agent.py             # Agent + 消息工厂函数测试
│   ├── test_config.py            # Config.from_env() 测试
│   ├── test_tools.py             # Tool 数据类 + 装饰器 + Schema 测试
│   └── test_memory.py            # ContextManager + STM + LTM 测试
│
├── examples/
│   └── basic_agent.py            # 最小使用示例
├── docs/
│   └── ARCHITECTURE.md           # 本文档
├── pyproject.toml                # Python 构建配置（hatchling）
├── README.md                     # 项目说明
├── CLAUDE.md                     # Claude Code 工作指引
└── .env.example                  # 环境变量模板
```

## 2. 文件详细说明

### 2.1 核心模块

#### `porsche_agent/agent.py`（140 行）

Agent 核心类，实现 ReAct（Reasoning + Acting）循环。

| 函数/类 | 签名 | 说明 |
|----------|------|------|
| `system_msg` | `(content: str) -> dict` | 构造 `{"role": "system", "content": ...}` |
| `user_msg` | `(content: str) -> dict` | 构造 `{"role": "user", "content": ...}` |
| `tool_msg` | `(tool_call_id: str, content: str) -> dict` | 构造 `{"role": "tool", ...}` |
| `Agent.__init__` | `(llm, tools=None, system_prompt=None, max_iterations=10, context_manager=None, short_term_memory=None, long_term_memory=None)` | 初始化 Agent，自动注册记忆工具 |
| `Agent.run` | `(user_message: str) -> str` | 执行 ReAct 循环，返回最终响应 |

**ReAct 循环流程：**

```
1. LTM 语义检索（如启用）→ 注入相关长期记忆
2. STM 查询（如启用）→ 注入会话内记忆
3. ContextManager.build_context()（如启用）→ 构建上下文窗口 + 摘要
4. 发送 messages + tools → LLM
5. 如果有 tool_calls → 执行工具 → 追加 tool 结果 → 回到步骤 4
6. 如果无 tool_calls → 返回 content 文本
```

**向后兼容：** `context_manager` / `short_term_memory` / `long_term_memory` 均为可选参数。传 `None` 时，Agent 使用原始 `self.history` 列表行为。

---

#### `porsche_agent/llm.py`（99 行）

LLM 提供商抽象层。使用 OpenAI Python SDK 作为底层 HTTP 客户端。

| 类/函数 | 说明 |
|----------|------|
| `LLMProvider(ABC)` | 抽象基类，定义 `chat(messages, tools) -> dict` 和 `embedding_client` 属性 |
| `DeepSeekProvider` | 指向 `https://api.deepseek.com`，默认模型 `deepseek-v4-pro` |
| `OpenAIProvider` | 指向 OpenAI 官方 API，默认模型 `gpt-4o` |
| `create_provider(config)` | 工厂函数，根据 `config.llm_provider` 返回对应实例 |

**设计决策：** Provider 只做单次 HTTP 调用，不处理 tool call 链。循环逻辑由 Agent 全权管理。

**`embedding_client` 属性：** 返回底层 `OpenAI` 客户端实例，供 `LongTermMemory` 调用 `client.embeddings.create()` 生成向量。

---

#### `porsche_agent/tools.py`（103 行）

工具系统，通过装饰器将 Python 函数转为 LLM 可调用的工具。

| 类/函数 | 说明 |
|----------|------|
| `Tool` (dataclass) | `name`, `description`, `parameters`（JSON Schema）, `func` |
| `Tool.__call__(**kwargs) -> str` | 调用包装函数，结果转为字符串 |
| `Tool.to_openai_schema() -> dict` | 输出 OpenAI 兼容的 tool schema |
| `tool(name=None, description=None)` | 装饰器，将函数转为 Tool 实例 |
| `_build_json_schema(func) -> dict` | 从函数签名和类型标注推断 JSON Schema |
| `_python_type_to_json_type(tp) -> str` | Python 类型 → JSON Schema 类型映射 |

**类型映射支持：**

| Python 类型 | JSON Schema 类型 |
|-------------|-----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `Optional[X]` / `X \| None` | 原类型 + `nullable: true` |

支持默认值参数（排除在 `required` 之外）。

---

#### `porsche_agent/builtin_tools.py`（31 行）

三个内置工具，CLI 默认加载。

| 工具 | 签名 | 说明 |
|------|------|------|
| `shell_command` | `(command: str) -> str` | 执行 shell 命令，超时 30s，返回 stdout/stderr |
| `read_file` | `(path: str) -> str` | 读取文件内容，支持 `~` 展开 |
| `write_file` | `(path: str, content: str) -> str` | 写入文件，自动创建父目录 |

---

#### `porsche_agent/config.py`（62 行）

配置数据类，从环境变量读取。

| 字段 | 默认值 | 环境变量 |
|------|--------|----------|
| `llm_provider` | `"deepseek"` | `PORSCHE_LLM_PROVIDER` |
| `api_key` | `None` | `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` |
| `base_url` | `"https://api.deepseek.com"` | `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` |
| `model` | `"deepseek-v4-pro"` | `DEEPSEEK_MODEL` / `OPENAI_MODEL` |
| `max_iterations` | `10` | `PORSCHE_MAX_ITERATIONS` |
| `system_prompt` | `None` | `PORSCHE_SYSTEM_PROMPT` |
| `temperature` | `0.0` | `PORSCHE_TEMPERATURE` |
| `context_window` | `20` | `PORSCHE_CONTEXT_WINDOW` |
| `context_summary_trigger` | `10` | `PORSCHE_CONTEXT_SUMMARY_TRIGGER` |
| `short_term_memory` | `True` | `PORSCHE_SHORT_TERM_MEMORY` |
| `long_term_memory` | `False` | `PORSCHE_LONG_TERM_MEMORY` |
| `vector_store_url` | `"http://127.0.0.1:9876"` | `PORSCHE_VECTOR_STORE_URL` |
| `embedding_model` | `"deepseek-embedding"` | `PORSCHE_EMBEDDING_MODEL` |

---

### 2.2 记忆子系统

#### `porsche_agent/memory/context.py`（80 行）

上下文管理器，维护全量历史但只暴露滑动窗口 + 摘要给 LLM。

| 类 | 说明 |
|----|------|
| `ContextManager.__init__(llm, max_window=20, summary_trigger=10)` | 设置窗口大小和摘要触发阈值 |
| `ContextManager.add(message)` | 追加消息到全量历史 |
| `ContextManager.build_context(extra_context=None)` | 构建 LLM 上下文：窗口内的直接返回，超出的先摘要再返回窗口 + 摘要 |
| `ContextManager.clear()` | 重置所有状态 |
| `ContextManager._summarize()` | 内部方法：将旧消息发给 LLM 生成摘要，支持增量摘要 |

**摘要逻辑：**

```
full_history ──→ [旧消息摘要] ──→ [最后 N 条消息]
                   ↑ 每 10 条新消息触发一次
```

- 摘要作为 system 消息注入，不影响 user/assistant 交替模式
- 原始 system prompt 始终保留在首位
- 懒触发：只在 `build_context()` 时检查是否需要摘要，避免工具循环中频繁 LLM 调用
- 支持增量摘要：当前摘要作为"Previous summary"传入下一轮

---

#### `porsche_agent/memory/short_term.py`（72 行）

会话级短期记忆，基于内存 dict + TTL 懒过期。

| 类/函数 | 说明 |
|----------|------|
| `Fact` (dataclass) | `key`, `value`, `created_at`, `ttl`（可选，秒） |
| `ShortTermMemory.put(key, value, ttl=None)` | 存储事实 |
| `ShortTermMemory.get(key) -> str\|None` | 获取事实，过期返回 None |
| `ShortTermMemory.delete(key) -> bool` | 删除事实 |
| `ShortTermMemory.all() -> dict` | 获取所有未过期事实 |
| `ShortTermMemory.to_prompt_fragment() -> str` | 格式化为 `[Session Memory]` 注入 LLM 上下文 |
| `create_memory_tools(stm) -> list[Tool]` | 生成 `remember` / `recall` 两个工具 |

**TTL 策略：** 在 `get()` 和 `all()` 时懒检查，过期即删。无后台线程，无并发问题。

**工具注入：** 传入 `short_term_memory` 给 Agent 时，`remember` / `recall` 自动注册，Agent 可在对话中自行决定何时存储/召回信息。

---

#### `porsche_agent/memory/long_term.py`（102 行）

持久化长期记忆，基于语义向量检索。

| 类/函数 | 说明 |
|----------|------|
| `LongTermMemory.__init__(embedding_client, embedding_model, vector_store_url)` | 初始化，接收 OpenAI 客户端和向量服务地址 |
| `LongTermMemory.add(memory_id, text, metadata)` | 嵌入文本 → 发送向量到 Go 服务 |
| `LongTermMemory.search(query, k=5) -> list[dict]` | 嵌入查询 → 搜索 Top-K |
| `LongTermMemory.delete(memory_id) -> bool` | 删除记忆 |
| `LongTermMemory.save(path)` | 持久化到 JSON 文件 |
| `LongTermMemory.load(path)` | 从 JSON 文件恢复 |
| `LongTermMemory._embed(text) -> list[float]` | 调用 OpenAI embedding API |
| `create_long_term_memory_tools(ltm) -> list[Tool]` | 生成 `remember_forever` / `search_memory` 工具 |

**数据流：**

```
Python add("user prefers dark mode")
  → OpenAI embedding API → [0.12, 0.34, ...]
    → HTTP POST /add → Go 向量索引

Python search("what theme?")
  → OpenAI embedding API → [0.11, 0.33, ...]
    → HTTP POST /search → Go 余弦相似度 → Top-K 结果
```

**通信：** 使用 Python stdlib `urllib.request`，零额外依赖。

---

### 2.3 Go 向量存储服务

#### `vectorstore/main.go`（178 行）

HTTP 服务器，监听默认端口 `9876`。无外部依赖，仅使用 Go 标准库。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 返回 `{"status": "ok", "count": N}` |
| `/add` | POST | 添加/更新向量 `{"id": str, "vector": [float32], "metadata": {}}` |
| `/search` | POST | 搜索 Top-K `{"vector": [float32], "k": int}` |
| `/delete` | DELETE | 删除向量 `{"id": str}` |
| `/save` | POST | 持久化到文件 `{"path": str}` |
| `/load` | POST | 从文件恢复 `{"path": str}` |

**启动特性：** 自动尝试从 `porsche_memory.json`（可配置 `VECTORSTORE_PATH`）恢复数据。

**维度：** 默认 1536，匹配 `deepseek-embedding` 和 `text-embedding-3-small`。

---

#### `vectorstore/store/store.go`（140 行）

核心存储引擎，线程安全（`sync.RWMutex`）。

| 类型/函数 | 说明 |
|-----------|------|
| `Entry` | 包含 `ID`, `Vector`, `Metadata` |
| `SearchResult` | 包含 `Entry` + `Distance`（余弦相似度） |
| `VectorStore` | 索引容器，`map[string]Entry` 存储 |
| `Search(query, k)` | 暴力余弦相似度搜索 O(n)，Top-K 排序 |
| `Save(path)` / `Load(path)` | JSON 序列化/反序列化持久化 |

**设计决策：** 使用暴力搜索而非近似索引（HNSW），因为 Agent 场景下记忆量级（数百到数千条）下 O(n) 完全够用。无需任何 Go 外部依赖，编译后的二进制约 9MB，可直接分发。

---

#### `porsche_agent/cli.py`（101 行）

CLI 入口，注册为 `porsche` 命令。

**子命令：**

| 命令 | 说明 |
|------|------|
| `porsche chat` | 交互式对话，自动装配 ContextManager + STM + LTM（按配置）+ 内置工具 |
| `porsche run SCRIPT` | 执行 Python 脚本，脚本可 `import porsche_agent` 自由组合 |

**REPL 特性：**
- 提示符 `>` 输入，`/exit` 或 Ctrl+D 退出
- 显示启动信息：提供商、模型、记忆模块状态
- RuntimeError（超出最大迭代数）被捕获并打印，不崩溃

---

### 2.4 对外公开 API

```python
# __init__.py 导出的全部符号
from porsche_agent import (
    Agent,                # Agent 核心类
    LLMProvider,          # LLM 提供商抽象基类
    DeepSeekProvider,     # DeepSeek 提供商
    OpenAIProvider,       # OpenAI 提供商
    create_provider,      # 工厂函数：Config → LLMProvider
    Tool,                 # 工具数据类
    tool,                 # @tool 装饰器
    Config,               # 配置数据类
    ContextManager,       # 上下文管理器
    ShortTermMemory,      # 短期记忆
    LongTermMemory,       # 长期记忆
)
```

## 3. 环境变量速查

```bash
# ---- 核心配置（必填） ----
export DEEPSEEK_API_KEY=sk-...        # DeepSeek API Key

# ---- 核心配置（可选） ----
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-v4-pro
export PORSCHE_MAX_ITERATIONS=10
export PORSCHE_SYSTEM_PROMPT="你是一个有用的助手"
export PORSCHE_TEMPERATURE=0.0

# ---- 记忆配置 ----
export PORSCHE_CONTEXT_WINDOW=20           # 滑动窗口大小
export PORSCHE_CONTEXT_SUMMARY_TRIGGER=10  # 摘要触发阈值
export PORSCHE_SHORT_TERM_MEMORY=true      # 启用短期记忆
export PORSCHE_LONG_TERM_MEMORY=false      # 启用长期记忆（需 Go 服务）
export PORSCHE_VECTOR_STORE_URL=http://127.0.0.1:9876
export PORSCHE_EMBEDDING_MODEL=deepseek-embedding

# ---- OpenAI 模式 ----
# export PORSCHE_LLM_PROVIDER=openai
# export OPENAI_API_KEY=sk-...
# export OPENAI_MODEL=gpt-4o
```

## 4. 项目启动

### 4.1 基本安装与运行

```bash
# 1. 克隆 / 进入项目目录
cd PorscheAgent

# 2. 创建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. 设置 API Key
export DEEPSEEK_API_KEY=sk-your-api-key-here

# 4. 启动交互对话
porsche chat

# 5. 运行示例脚本
porsche run examples/basic_agent.py
```

### 4.2 启用长期记忆（Go 向量服务）

```bash
# 终端 1：启动 Go 向量服务
cd vectorstore
go build -o vectorstore .
./vectorstore
# 输出: Vector store listening on :9876 (dim=1536)

# 终端 2：启用 LTM 并启动 chat
cd PorscheAgent
source .venv/bin/activate
export DEEPSEEK_API_KEY=sk-...
export PORSCHE_LONG_TERM_MEMORY=true
porsche chat
# 输出: PorscheAgent chat [deepseek:deepseek-v4-pro]
#        context window=20, stm=True, ltm=True
```

## 5. 使用方式

### 5.1 库模式 — 基础用法

```python
from porsche_agent import Agent, DeepSeekProvider, tool

# 定义自定义工具
@tool(description="获取城市天气")
def get_weather(city: str) -> str:
    return f"{city}: 晴, 25°C"

# 创建 Agent
llm = DeepSeekProvider(api_key="sk-...")
agent = Agent(llm=llm, tools=[get_weather])
print(agent.run("北京今天天气怎么样？"))
```

### 5.2 库模式 — 带记忆

```python
from porsche_agent import (
    Agent, DeepSeekProvider,
    ContextManager, ShortTermMemory, LongTermMemory,
)

llm = DeepSeekProvider(api_key="sk-...")

# 三层记忆
ctx = ContextManager(llm=llm, max_window=20)
stm = ShortTermMemory()
ltm = LongTermMemory(
    embedding_client=llm.embedding_client,
    embedding_model="deepseek-embedding",
)

agent = Agent(
    llm=llm,
    context_manager=ctx,
    short_term_memory=stm,
    long_term_memory=ltm,
    system_prompt="你是一个有用的助手，请用中文回答。",
)

# Agent 会自动获得以下工具：
#   remember / recall（短期记忆）
#   remember_forever / search_memory（长期记忆）

print(agent.run("我叫小明，住在北京。"))
# Agent 可以使用 remember_forever 存储这些信息

print(agent.run("我叫什么名字？"))
# Agent 可以使用 recall 或 search_memory 检索
```

### 5.3 库模式 — OpenAI 提供商

```python
from porsche_agent import Agent, OpenAIProvider

llm = OpenAIProvider(api_key="sk-...", model="gpt-4o")
agent = Agent(llm=llm)
print(agent.run("Hello!"))
```

### 5.4 CLI 模式

```bash
# 交互对话
porsche chat --system-prompt "你是一个有用的助手" --max-iterations 20

# 运行用户脚本
porsche run my_script.py
```

### 5.5 使用 Config 驱动

```python
from porsche_agent import Config, create_provider, Agent

config = Config.from_env()               # 自动读环境变量
llm = create_provider(config)
agent = Agent(
    llm=llm,
    system_prompt=config.system_prompt,
    max_iterations=config.max_iterations,
)
```

## 6. 架构全景图

```
                          ┌──────────────────────┐
                          │    CLI (cli.py)       │
                          │  porsche chat / run   │
                          └──────────┬───────────┘
                                     │ 组装
                  ┌──────────────────▼──────────────────┐
                  │            Config (config.py)        │
                  │         从环境变量读取配置             │
                  └──────────────────┬──────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │                  Agent (agent.py)            │
              │  ┌──────────────────────────────────────┐   │
              │  │           ReAct 循环                  │   │
              │  │   think → act → observe → repeat     │   │
              │  └──────────────────────────────────────┘   │
              │                                             │
              │  ┌──────────┐ ┌───────────┐ ┌───────────┐  │
              │  │ Context  │ │ Short-term│ │ Long-term │  │
              │  │ Manager  │ │ Memory    │ │ Memory    │  │
              │  │          │ │           │ │           │  │
              │  │ 滑动窗口 │ │ 会话KV    │ │ Python API│  │
              │  │ +LLM摘要 │ │ +TTL      │ │           │  │
              │  └──────────┘ └───────────┘ └─────┬─────┘  │
              └───────────────────────────────────┼────────┘
                                                  │ HTTP
              ┌───────────────────────────────────▼────────┐
              │         Go Vector Store (vectorstore/)      │
              │     POST /add /search /delete /save /load   │
              │         余弦相似度  |  JSON 持久化           │
              └─────────────────────────────────────────────┘

              ┌────────────────────────────────────────────┐
              │           LLM Provider (llm.py)            │
              │  ┌─────────────────┐ ┌──────────────────┐  │
              │  │ DeepSeekProvider│ │ OpenAIProvider   │  │
              │  │ (OpenAI SDK)    │ │ (OpenAI SDK)     │  │
              │  └────────┬────────┘ └────────┬─────────┘  │
              │           │ chat + embeddings  │            │
              └───────────┼────────────────────┼────────────┘
                          ▼                    ▼
                   DeepSeek API          OpenAI API
```

## 7. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 消息格式 | OpenAI 原始 dict | 零转换开销，直接兼容 API |
| Provider 职责 | 仅单次 HTTP 调用 | Agent 拥有循环，Provider 无状态 |
| Tool 定义 | `@tool` 装饰器 | 最简 API，零样板代码 |
| 上下文管理 | 滑动窗口 + 懒摘要 | 避免工具循环中频繁 LLM 调用 |
| STM TTL | 懒过期（读写时检查） | 无需后台线程，无并发问题 |
| Go 向量搜索 | 暴力余弦相似度 | Agent 量级下 O(n) 足够，零外部依赖 |
| HTTP 通信 | stdlib `urllib` | 零额外 Python 依赖 |
| Python 版本 | >= 3.10 | 支持 `X \| None` 现代语法 |
