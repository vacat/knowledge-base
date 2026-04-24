# 本地 LLM 知识库

> X Bookmarks / RSS → 自动抓取 → LLM 分类打分 → Obsidian Vault

**架构灵感**：Karpathy 的 [LLM Knowledge Base](https://github.com/karpathy/llm-wiki)  
**核心理念**：vault = 代码仓库，LLM = 编译器，raw → wiki → outputs 单向流动

---

## 架构图

```
X bookmark                    RSS Feeds                  微信公众号
    │                              │                            │
    └──────────┬───────────────────┘                            │
               ▼                                                ▼
     GitHub Actions (云端 cron 每30分钟)
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   刷新token  抓内容   LLM分类
       │       │        │
       └───────┼────────┘
               ▼
      raw/<category>/*.md
               │
               ▼ (git push)
         GitHub Repo (source of truth)
               │
               ▼ (obsidian-git pull)
         本地 Obsidian Vault
               │
               ▼ (Claude Code 润色)
      wiki/concepts → wiki/summaries → outputs/
```

---

## 目录结构

```
.
├── .github/
│   └── workflows/
│       └── ingest.yml        # GitHub Actions workflow
├── scripts/
│   ├── ingest.py             # 主抓取脚本
│   └── validate_env.py       # 环境变量校验
├── data/
│   ├── feeds.json            # RSS 源配置
│   ├── processed_bookmarks.json  # X bookmarks 增量记录
│   └── feed_state.json       # RSS 抓取状态
├── raw/                      # 原始内容（抓取后未经处理）
│   ├── AI-ML/
│   ├── coding/
│   ├── design/
│   ├── productivity/
│   ├── research/
│   ├── business/
│   └── other/
├── wiki/                     # AI 润色后的内容
│   ├── concepts/             # 概念页面
│   └── summaries/             # 摘要页面
├── outputs/                  # 最终输出（报告、生成内容）
└── requirements.txt
```

---

## 快速开始

### 1. 创建 GitHub Repo

```bash
# 在 GitHub 上创建新仓库，然后：
git clone https://github.com/YOUR_USERNAME/knowledge-base.git
cd knowledge-base
git remote add upstream https://github.com/YOUR_USERNAME/knowledge-base.git
```

### 2. 配置 GitHub Secrets

在 GitHub 仓库的 **Settings → Secrets and variables → Actions** 中添加：

| Secret | 必需 | 说明 |
|--------|------|------|
| `GH_PAT` | ✅ | GitHub Personal Access Token（需要 repo 读写权限）|
| `OPENAI_API_KEY` | 推荐 | GPT-4o-mini 用于分类打分 |
| `ANTHROPIC_API_KEY` | 可选 | Claude Haiku 作为备选 |
| `ZHIPU_API_KEY` | 可选 | GLM-4-Flash 作为备选 |
| `JINA_API_KEY` | 可选 | Jina Reader 提升内容抓取质量 |
| `X_BEARER_TOKEN` | 可选 | X API v2 Bearer Token |
| `X_USER_ID` | 可选 | X User ID（数字形式，用于获取 bookmarks）|

> **没有 X token？** 只用 RSS 模式也能跑满功能，X bookmark 会被跳过。

### 3. 获取 X User ID

1. 访问 [platform.x.com](https://developer.x.com)
2. 创建 App，获取 Bearer Token
3. 用 curl 获取 User ID：
```bash
curl -s "https://api.twitter.com/2/users/by/username/YOUR_USERNAME" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" | jq '.data.id'
```

### 4. 启用 GitHub Actions

```bash
git add .
git commit -m "chore: initial knowledge base setup"
git push origin main
```

然后在 GitHub 仓库的 **Actions** 页面手动触发第一次运行。

### 5. 克隆到本地 Obsidian

```bash
# 克隆仓库到 Obsidian vault 目录
git clone https://github.com/YOUR_USERNAME/knowledge-base.git /path/to/your-vault

# 安装 obsidian-git 插件（自动同步）
# 或者手动：
git pull origin main
```

---

## 数据源配置

### RSS Feeds

编辑 `data/feeds.json`，添加你关注的订阅源：

```json
{
  "Hacker News": "https://hnrss.org/frontpage",
  "AI/ML ArXiv": "https://arxiv.org/rss/cs.AI",
  "即刻": "https://web.okjike.com/rss",
  "少数派": "https://sspai.com/feed",
  "科技资讯": "https://www.ifanr.com/feed"
}
```

### X Bookmarks

确保 Secrets 中配置了 `X_BEARER_TOKEN` 和 `X_USER_ID`，系统会自动抓取书签中包含 URL 的推文。

---

## LLM 模型选择

系统支持多 Provider，按优先级自动选择：

```bash
# 在 workflow 或本地运行时指定
export LLM_PROVIDER=openai   # GPT-4o-mini（推荐）
export LLM_PROVIDER=zhipu    # GLM-4-Flash（国产，便宜）
export LLM_PROVIDER=anthropic # Claude Haiku
```

**推荐配置**：

| 场景 | 推荐模型 | 价格参考 |
|------|---------|---------|
| 通用分类 | GPT-4o-mini | $0.15 / $0.60 per M tokens |
| 国产首选 | GLM-4-Flash | ¥0.5 / M tokens |
| 低成本备用 | Qwen-turbo | ¥0.3 / M tokens |

---

## 本地运行（可选）

不通过 GitHub Actions，也可以在本地跑：

```bash
# 克隆
git clone https://github.com/YOUR_USERNAME/knowledge-base.git
cd knowledge-base

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API keys

# 运行抓取
python scripts/ingest.py
```

---

## 添加微信公众号支持

微信被大多数工具 block，需要特殊处理。参考 [wechat-article-ingest skill](../skills/wechat-article-ingest/SKILL.md)：

1. 注册 **即刻** 或使用 **RSSHub** 生成微信公众号 RSS
2. 在 `data/feeds.json` 中添加 RSS 地址
3. 如果需要更强的微信文章抓取，可以配合 [browser tool](../skills/dogfood/) 使用 Playwright 渲染

---

## 常见问题

### X API Token 过期
X 的 Access Token 通常 30 天后过期。访问 [developer.x.com](https://developer.x.com) 重新生成，并更新 `X_BEARER_TOKEN` secret。

### Jina Reader 限流
免费 Jina 账号有频率限制。切换到备选方案：
```python
# 在 ingest.py 中注释掉 JINA_API_KEY
# 或使用 BRAVE_API_KEY 作为搜索备选
```

### GitHub Actions 超时
2000 分钟/月的免费额度。确保 `ingest.yml` 中 `timeout-minutes: 25`，每次运行不超过 25 分钟。

### 单文件过大
内容超过 8000 字会被截断。如需完整存档，可以分片存储或使用外部存储（如 S3）。

---

## 扩展方向

- **知识图谱**：用 `wiki/concepts/` 构建概念关系网
- **定时报告**：每月生成 `outputs/monthly-digest.md`
- **Airtable 同步**：把高分内容同步到 Airtable 管理
- **Notion 备份**：用 Notion API 备份关键笔记
- **全文搜索**：接入 Elasticsearch 或 Typesense

---

## License

MIT — 随意 fork 和定制。
