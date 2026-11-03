# P0 报告：本地 4×T4 基座验证与迁移测量

日期：2026-11-03
分支：`main`（EdgeVisor-wansongcc）
结论：**P0 通过**。四卡 PP 与真实 KV 迁移链路均已在本地跑通，测量口径可用。

## 1. 环境事实

| 项 | 值 |
| --- | --- |
| GPU | 4 × Tesla T4 15GB，cc 7.5 / sm_75 |
| GPU 互联 | PCIe 经 host bridge（GPU0-1 同 NUMA，GPU2-3 同 NUMA，跨组走 SYS） |
| 驱动 / CUDA | 595.99.02 / 12.8（`/usr/local/cuda` → `cuda-12.8`） |
| CPU / RAM | 48 核 / 187 GB |
| 权限 | 普通用户，无 sudo；已在 `docker` 组（网络整形走容器，不碰 host 网络） |
| 模型 | llama3.2-3B-instruct Q40，`Dim=3072, HeadDim=128, KvDim=1024, nHeads=24, 28 层` |

## 2. 编译

旧二进制为 2026-09-25，源码已到 2026-11-02，必须先重编译。

```
CUDA_ARCHS=75 requested=auto   # Makefile 自动探测，值正确
dllama backends: cuda=1 vulkan=0
```

`/usr/local/cuda` 符号链接已存在，无需额外指定 `CUDA_HOME`。

## 3. 单卡基线

```
device: backend=cuda threads=8
buffer: q80
tokens/s: 27.69 (36.11 ms/tok)
输出: "Paris."
```

## 4. 四卡 PP 基线

每卡一个 worker 进程（`--gpu-index 1/2/3`），root 在 GPU 0。

| 配置 | tokens/s | ms/tok | topology |
| --- | --- | --- | --- |
| 静态均分 `1@7*1@7*1@7*1@7` | 21.51 | 46.5 | 手工指定 |
| `--auto` 首轮 | 17.53 | 57.0 | speed-pack 后回到均分 |

`--auto` 首轮反而更慢，因为 speed-pack 探测 + 权重全量加载占了时间；这一轮不是稳态数字，只用于确认路径。

## 5. 迁移实测（关键结果）

通过 plan UDS 注入 `set_pp_migration`，在固定 position 触发。

```
op: set_pp_migration
cmd: { mode: "exact", stageIndex: 0, fromNode: 0, toNode: 1,
       layerCount: 1, triggerPos: 40, triggerLayer: 24 }
```

得到的 `pp_migration_recover` 事件：

| 字段 | 值 |
| --- | --- |
| `trigger_pos` | 40 |
| 路由 | 0 → 1 |
| `stall_time_ms` | **43.363** |
| `state_transfer_bytes` | **4,048,208**（3.86 MiB） |
| `t_state_prepare_ms` | 10.033 |
| `t_recover_ms` | 33.326 |
| `recompute_tokens_or_layers` | **0** |
| `binding_update_count` | 6 |
| `apply_success` | true |

第二次迁移（2 → 1，pos=65）：`stall 0.326 ms`，`bytes 0` —— 走的是无需传输的路径。

**这是 P0 最重要的产出**：EV 已经能在四卡上完成一次带真实 KV 传输的 PP 迁移，且三个核心度量（停顿、字节、重算）都有结构化落盘。

## 6. 由实测推出的 ρ

| 量 | 值 | 来源 |
| --- | --- | --- |
| 每 token 每层 KV（k+v，F32） | 8 KiB | `KvDim=1024` × 2 × 4B |
| 单次迁移平均速率 | ≈ 922 position/s | 4.05 MB / 43.4 ms |
| decode 速率 r | ≈ 17–21 tok/s | 本轮实测 |
| **ρ = r / B** | **≈ 0.018** | 17 / 922 |

结论：**ρ 远小于 1，一轮即可把残差压到 τ=50 以下**。

按 2048 上下文外推：

| 量 | 值 |
| --- | --- |
| 全量迁移耗时 | ≈ 2.2 s |
| 期间新增残差 | ≈ 38 token（< τ=50） |
| 结论 | 单轮收敛，无需第二轮 |

对照停等：全量 2048 需 ≈ 2.2 s 阻塞；patching 收尾只需补 38 token ≈ 41 ms。**约 54 倍差距**，与 PipeLive 论文量级一致。

## 7. 待确认项（必须在 P1 前解决）

**字节账目对不上，需核实到底传了什么。**

- 若只传 1 层 × 41 position：应为 `41 × 8 KiB ≈ 328 KiB`
- 实测 `4.05 MB`，约为该值的 **12.3 倍**
- `binding_update_count = 6`，而 `layerCount = 1`

三种可能，需逐一排除：

1. `state_transfer_bytes` 统计的是双向或含协议开销
2. 实际迁移的层数不是 1（`binding_update_count` 反映真实层数）
3. 传输包含整个 stage 的 KV 而非单层

这一项不查清，所有"迁移字节"口径都不可信，是 P1 之前的第一优先。

## 8. 已确认可用的控制面

| 能力 | 入口 | 备注 |
| --- | --- | --- |
| 触发迁移 | UDS `set_pp_migration` | plan socket 仅在 `--auto` 下启动 |
| 绕过 JIT 门控 | `EDGEVISOR_JIT_MODE=greedy` | `enabled` 模式要求在线估计值，注入命令没有，必被拒 |
| 迁移事件落盘 | `EDGEVISOR_ABLATION_LOG_PATH` | JSONL，含 stall/bytes/recompute |
| 迁移详细日志 | `DLLAMA_MIGRATION_VERBOSE_LOG=1` | |

## 9. 复现命令

```bash
# 编译
cd /home/byh/B01/EdgeVisor-wansongcc && make -j8 dllama

# 基线 + auto
cd /home/byh/B01/baseline/_harness
python3 measure_migration.py --out results/

# 注入一次迁移
python3 inject_migration.py \
  --steps 200 --trigger-pos 40 --trigger-layer 24 --layer-count 1 \
  --prompt "Explain in detail the history and construction of the Eiffel Tower, including the engineers involved" \
  --out results/inject
```

## 10. 对后续阶段的影响

1. **P1 的优先级不变**：`single-pending` 仍需解除，因为 patching 要靠多轮补差，即便本机 ρ 很低，也必须能连续多轮才能测出收敛曲线。
2. **ρ 实测值可用于实验设计**：本机上 patching 会一轮收敛，所以要在**人为压带宽**下才有对比价值——这正好是 Docker + `tc netem` 的用途。
3. **τ 的选择可实验化**：既然残差增长可预测（≈ 17 tok/s），τ 可以按"停顿时长预算"反推，而不是照抄论文的 50。
