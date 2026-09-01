# 2.1 · 异步排序

核心顺序：

```text
st.shared
→ fence.proxy.async
→ wgmma.fence
→ wgmma.mma_async
→ wgmma.commit_group
→ wgmma.wait_group
```

`fence.proxy.async` 处理 generic proxy 与 async proxy 的可见性；`wgmma.fence`
约束 WGMMA 相关寄存器访问；`commit_group` 建立异步组；`wait_group` 等待允许
的未完成组数降到指定上限。

题面判断答案依次为：错、错、对。

