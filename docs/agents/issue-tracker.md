# Issue 跟踪器：GitHub

本仓库的 issue 和规格以 GitHub issue 形式存放。所有操作使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文使用 heredoc。
- **读取 issue**：`gh issue view <number> --comments`，用 `jq` 过滤评论，并同时获取 labels。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，配合适当的 `--label` 和 `--state` 过滤。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **添加 / 移除标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：`gh issue close <number> --comment "..."`

从 `git remote -v` 推断仓库；在克隆目录内运行 `gh` 时会自动完成。

## 将 PR 作为 triage 面

**将 PR 作为请求面：否。** _（如果本仓库把外部 PR 视为功能请求，改为 `yes`；`/triage` 会读取此标志。）_

设为 `yes` 后，PR 与 issue 走相同的标签和状态，使用对应的 `gh pr` 命令：

- **读取 PR**：`gh pr view <number> --comments`，diff 用 `gh pr diff <number>`。
- **列出待 triage 的外部 PR**：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，仅保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR` 或 `NONE` 的（排除 `OWNER`/`MEMBER`/`COLLABORATOR`）。
- **评论 / 标签 / 关闭**：`gh pr comment`、`gh pr edit --add-label`/`--remove-label`、`gh pr close`。

GitHub 的 issue 与 PR 共享同一编号空间，因此单独的 `#42` 可能是任一者：先用 `gh pr view 42` 解析，失败再回退到 `gh issue view 42`。

## 当技能要求“发布到 issue 跟踪器”

创建一个 GitHub issue。

## 当技能要求“获取相关工单”

运行 `gh issue view <number> --comments`。

## Wayfinding 操作

供 `/wayfinder` 使用。**map** 是一个 issue，**子工单**也是 issue。

- **Map**：一个带 `wayfinder:map` 标签的 issue，正文保存 Notes / Decisions-so-far / Fog 内容。`gh issue create --label wayfinder:map`。
- **子工单**：作为 map 的 GitHub sub-issue 关联（对 sub-issues 端点调用 `gh api`）。未启用 sub-issues 时，把子工单加入 map 正文的任务列表，并在子工单正文顶部写 `Part of #<map>`。标签：`wayfinder:<type>`（`research`/`prototype`/`grilling`/`task`）。被认领后，工单分配给执行的开发者。
- **阻塞**：GitHub **原生 issue 依赖**，这是标准且 UI 可见的表示。添加边：`gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`，其中 `<blocker-db-id>` 是阻塞者的数字**数据库 id**（`gh api repos/<owner>/<repo>/issues/<n> --jq .id`，_不是_ `#number` 或 `node_id`）。GitHub 通过 `issue_dependencies_summary.blocked_by` 报告（仅开放的阻塞者，即实时门禁）。依赖不可用时，回退到子工单正文顶部的 `Blocked by: #<n>, #<n>` 行。当所有阻塞者都关闭时，工单解除阻塞。
- **Frontier 查询**：列出 map 的开放子工单（`gh issue list --state open`，限定在 map 的 sub-issues / 任务列表），丢弃有开放阻塞者（`issue_dependencies_summary.blocked_by > 0`，或 `Blocked by` 行中有开放 issue）或已有 assignee 的；按 map 顺序取第一个。
- **认领**：`gh issue edit <n> --add-assignee @me`，会话的第一次写入。
- **解决**：`gh issue comment <n> --body "<answer>"`，然后 `gh issue close <n>`，再把上下文指针（要点 + 链接）追加到 map 的 Decisions-so-far。
