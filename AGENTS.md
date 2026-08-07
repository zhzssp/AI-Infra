# AI-Infra Agent 说明

本仓库自学材料偏「总览 + 路线」；细节常靠对话补齐。按下面约定工作。

## 会话启动

开始处理本仓库任务前，先读本文件；需要写笔记时再读并遵循对应 skill。

## 写笔记（必用 skill）

当用户要求**写笔记 / 做笔记 / 记录回答 / 记下来**（或同等意思）时：

1. **立即读取并遵循**项目 skill：[`.cursor/skills/ai-infra-notes/SKILL.md`](.cursor/skills/ai-infra-notes/SKILL.md)
2. 只记录用户**认可**的解释（通常是刚问完、用户满意后要求沉淀的那次回答）
3. 产物固定为两步：
   - 在 `docs/notes/` 写一篇简短笔记
   - 在对应知识文档（如 `docs/llvm-learning-guide.md`）的适当位置加一条「速记」引用

未要求写笔记时，不要主动创建 `docs/notes/` 文件。

## 知识文档 vs 笔记

| | 职责 |
|--|------|
| `docs/*-learning-guide.md`、`docs/ai-compiler-foundations.md`、`docs/paper-notes/` | 主知识库：路线、框架、过关标准 |
| `docs/notes/` | 对话沉淀的短注：某个概念的认可版解释 |

主文档保持可扫读；深挖细节进 `notes/`，再用引用挂回去。
