---
name: ai-infra-notes
description: >-
  Capture an approved Q&A explanation into docs/notes/ and add a backlink from
  the matching knowledge guide. Use when the user asks to 写笔记、做笔记、记录回答、
  记下来, or otherwise save a prior explanation into notes the same way as
  docs/notes/llvm-phi.md.
---

# AI-Infra 问答笔记

把用户**认可**的解释沉淀为短笔记，并挂回知识文档。只在用户明确要求「写笔记 / 做笔记 / 记录回答」等时执行；不要主动写笔记。

## 工作流

按顺序完成，缺信息就问一句再动手：

1. **确认内容**：笔记写的是用户刚认可的那次回答（可按用户补充微调），不是另起炉灶重写长文。
2. **确认挂靠点**：找出对应的知识文档与章节（如 `docs/llvm-learning-guide.md` §2.3）。用户若 `@` 了文件，以该文件为准。
3. **写笔记**：在 `docs/notes/` 新建一个简短 Markdown（见下方模板）。
4. **加引用**：在知识文档讲解该概念的**适当位置**插入速记引用（见下方格式）。
5. **收尾**：回复笔记路径 + 引用加在哪一节；不主动 commit。

## 笔记文件

- 路径：`docs/notes/<topic>.md`
- 命名：短横线、小写、主题清晰（例：`llvm-phi.md`、`gep-inbounds.md`）
- 已存在同主题文件：追加/修订该文件，不要无意义新建
- 篇幅：短；只保留定义、关键规则、必要示例、与相邻概念的对照

### 模板

```markdown
# <标题>

> 来源：[`<knowledge-doc>.md`](../<knowledge-doc>.md) §<节号或小节名>

## 是什么

<一两段：定义 + 要解决的问题>

## <关键规则 / 要点>

1. ...
2. ...

## <可选：对照 / 何时会出现>

...
```

示例：[docs/notes/llvm-phi.md](../../../docs/notes/llvm-phi.md)

## 知识文档引用

插在该概念讲解正文附近（规则句之后、延伸对比之前较合适），用相对路径指向 `notes/`：

```markdown
> **速记**：[notes/<topic>.md](./notes/<topic>.md) —— <一句话摘要>。
```

要求：

- 一句话摘要能让扫读者决定是否点开
- 路径按知识文档所在目录计算（`docs/*.md` → `./notes/...`；`docs/paper-notes/*.md` → `../notes/...`）
- 不重复粘贴整篇笔记；不改动无关章节

## 不要做的事

- 用户没说写笔记时，不要创建 `docs/notes/` 文件
- 不要把笔记写成第二份学习指南（避免长篇展开）
- 不要更新 `docs/README.md` / 根 `README.md`，除非用户要求
- 不要 commit，除非用户明确要求
