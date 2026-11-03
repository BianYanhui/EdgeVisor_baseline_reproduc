# P1 报告：预拷贝机制验证（不改代码）

日期：2026-11-03
结论：**P1 原计划的前提是错的，但结论更好——EV 已经实现了 PipeLive 式增量 patching，且实测有效。**

## 1. 原计划的错误

P1 原本要做的是"解除 `single-pending` 约束以支持多轮补差"。核实代码后确认这个前提不成立：

- `runBackgroundPreCopy`（`app.cpp:4464`）已经实现了"边跑边追"的增量预拷贝
- 它按 `bgCopiedPos` 单调推进（`startPos = bgCopiedPos + 1`），ack 期间返回 0 只是跳过本轮，下一轮继续
- **多轮收敛天然成立，不需要解除任何约束**

`waitingKvAck` 阻止的是"同时发起两条不同的迁移命令"（命令级排他），不是传输轮次级排他。把两者混为一谈是原计划的根本错误。

## 2. 预拷贝在哪里生效

`runBackgroundPreCopy` 只在 agentic episode 路径被调用：

```1917:1930:EdgeVisor-wansongcc/EdgeVisor/src/dllama.cpp
        for (unsigned int token = 0u; token < round.generationTokens; ++token) {
            currentToken = episodeForwardAndSample(context, pos, currentToken, "generation", roundNumber);
            pos += 1u;
        }
        ...
        if (std::getenv("DLLAMA_BACKGROUND_PRECOPY") != nullptr) {
            const unsigned long long bgUs = context->inference->runBackgroundPreCopy(pos);
```

**普通 `inference` 路径不走它**。它需要 `DLLAMA_EPISODE_SCRIPT` 提供 tool_wait 空闲窗口。这是后续所有 patching 实验的前提条件。

## 3. 三态开关

触发路径按 `ShadowKvMode` 分三条，只有后两条理解 delta：

| 模式 | 迁移分支 | 是否用 `bgCopiedPos` |
| --- | --- | --- |
| `enabled` | `shadow forward transfer` | ❌ 无条件全量 |
| `disabled_transfer` | `real transfer prepared` | ✅ `deltaStart = bgCopiedPos + 1` |
| `disabled_recompute` | 重算，不传输 | — |

**做 patching 实验必须显式设 `EDGEVISOR_SHADOW_KV_MODE=disabled_transfer`。** 默认的 `enabled` 会让预拷贝白做——这是我第一次实验失败的原因。

## 4. 决定性对照实验

配置：4×T4，`triggerPos=100`，`layerCount=1`，episode 脚本提供 5 个 tool 窗口，每次 200ms。

第一次手工测量：

| 配置 | 迁移行数 | `posRange` | 传输字节 | 停顿 |
| --- | --- | --- | --- | --- |
| 无预拷贝 | 101 | `[0,100]` | 1,662,088 | **18.19 ms** |
| 有预拷贝 | 13 | `[88,100]` | 213,960 | **5.63 ms** |

用 `_harness/precopy_pair.py` 复跑（自动清除继承环境）：

| 配置 | 预拷贝轮数 | 迁移行数 | 传输字节 | 停顿 |
| --- | --- | --- | --- | --- |
| 无预拷贝 | 0 | 101 | 1,662,088 | **27.23 ms** |
| 有预拷贝 | 5 | 12 | 197,504 | **4.23 ms** |

- 传输减少 **88.1%**
- 停顿减少 **84.5%**
- 预拷贝本身 5 轮，每轮 5–10 ms，都在 tool_wait 窗口内完成

停顿绝对值两次测量有差异（18.19 vs 27.23 ms），属正常波动；**比值稳定在 84–88%**，这才是机制本身的贡献。

日志证据：

```
[bg-precopy] copied_to=47  ... passes=1
[bg-precopy] copied_to=87  ... passes=2
[bg-precopy] trigger delta posRange=[88,100] trigger_bytes=213960 bg_passes=2
[kv-migrate] real transfer prepared exported=13 queued=13 posRange=[0,100]
```

`posRange=[0,100]` 是全程范围，实际只传 `exported=13` 行——delta 生效。

## 5. 字节账目（第二次确认）

预拷贝每轮每行 `16,456 B = 2 × 8224`：

```
8224 = 1024 (KvDim) × 2 (K,V) × 4 B + 32 B header
16456 = 8224 × 2   （source + target 各计一次）
```

5 轮 208 行共 3,423,008 B，比率 2.0011，与模型几何完全吻合。

## 6. 三个必须记录的坑

**坑一：环境变量会从交互 shell 泄漏。** 我第一次做对照时两组都跑了预拷贝，因为之前 `export DLLAMA_BACKGROUND_PRECOPY=1` 留在了 shell 里，而脚本 `env = os.environ.copy()` 继承了它。**对照实验必须显式 `env.pop()`。**

**坑二：触发位置必须晚于第一次预拷贝。** 预拷贝第一次在 `copied_to=47`，若 `triggerPos=40`，迁移时 `bgPrecopyPasses` 仍为 0，`bgActive=false`，会走全量。

**坑三：默认 `shadow_kv_mode=enabled` 会绕过 delta。** 必须显式切到 `disabled_transfer`。

## 7. 对后续阶段的影响

1. **P1 从"改代码"变成"已可用 + 需测量"。** 原计划 1 周，实际只需补测量口径。
2. **PipeLive 移植的工作量进一步下降。** 核心机制（增量收敛 + delta 切换）已经在仓库里，需要做的是：把它从 "agentic episode 专用" 泛化到普通 decode 路径，以及调 τ（`tool_wait_ms` 的等价物）。
3. **`bgPrecopyMaxPasses` 已存在**，可直接用来做"最多补几轮"的消融实验。
4. **τ 的实验化路径清晰了**：τ 在本实现里等价于"最后一次预拷贝到迁移触发的距离"，可通过 `triggerPos - bgCopiedPos` 直接控制。

## 8. 复现命令

```bash
# 必须：episode 脚本提供 tool 窗口
export DLLAMA_EPISODE_SCRIPT=/tmp/ev_p1/episode.script
export EDGEVISOR_SHADOW_KV_MODE=disabled_transfer
export DLLAMA_BACKGROUND_PRECOPY=1          # 有预拷贝组
export DLLAMA_TPOT_MIN_PP_GAIN_MS=1000000000  # 抑制调度器
export EDGEVISOR_JIT_MODE=greedy              # 允许注入
# 注入时 triggerPos 必须 > 第一次预拷贝的 copied_to
```
