---
name: document-search
description: 知识库检索，通过调用子agent rag 让其在内部知识库中检索一些内容，并返回相关信息
triggers:
  - 用户需要在知识库查询一些知识
  - 你有一些不明白的名词或内容
---
# Document_search SKILL
最佳实践
1. 首先，针对你要解决的问题：
   - 基于问题来思考一个查询语句（查询改写）
   - 如果你认为一个查询语句无法覆盖，那需要拆解为多个查询语句依次查询并汇总。
2. 接下来调用subagent()工具，让subagent发起一次查询
