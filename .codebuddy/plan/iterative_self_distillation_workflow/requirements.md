# 需求文档：迭代式自蒸馏训练 Workflow 自动化

## 引言

本功能旨在将现有的 **"数据生成 → 数据合并 → 训练 → 评估"** 四步人工串联流程，自动化为一个可一键启动、可迭代 N 轮的 shell workflow。

目标仓库：`/apdcephfs_jn3/share_535475/common/dellwu/RE-START`

### ⚠️ 关键设计约束：不修改任何现有脚本

为保证原有手工实验流程**零破坏**，本 workflow 采用"**克隆 + auto_ 前缀**"策略：

| 原脚本（保持原样，绝不修改） | 新增的自动化副本 |
|---|---|
| `scripts/DeepSeek/run_pipline.sh` | `scripts/DeepSeek/auto_run_pipline.sh` |
| `scripts/DeepSeek/run_ultrafeedback.sh` | `scripts/DeepSeek/auto_run_ultrafeedback.sh` |
| `data/make_train_data.py` | `data/auto_make_train_data.py` |
| `scripts/DeepSeek/sft_peft_train.sh` | `scripts/DeepSeek/auto_sft_peft_train.sh` |
| `scripts/DeepSeek/run_evaluation_pool.sh` | `scripts/DeepSeek/auto_run_evaluation_pool.sh` |
| —— | `scripts/DeepSeek/auto_run_iterative_workflow.sh` **（新增的顶层编排脚本）** |

> **只有 `auto_*` 前缀的脚本会被 workflow 调用；原 5 个脚本永远不被 workflow 触碰。**
> `auto_*` 脚本内部的业务逻辑（模型生成、数据过滤、训练参数、评估调度）必须与原脚本 **功能行为一致**，仅在以下两方面扩展：① 接受环境变量/CLI 参数作为输入；② 集成到 workflow 编排。

（另外 `LlamaFactory/data/dataset_info.json` 已预先注册 restart-1/2/3 三个数据集条目，本 workflow 不修改该文件。）

### 迭代语义重申

- 第 k 轮数据生成的**输入模型** = 第 k-1 轮训练产出的 LoRA-merged 模型（第 1 轮用 base 模型）。
- 第 k 轮训练的**基础模型**：始终是 **base 模型** `DeepSeek-R1-Distill-Qwen-14B`，**不是上一轮产出的模型**。即每轮都是从 base 模型重新做 LoRA PEFT 训练，只是训练数据随轮次累积更新。
- 最终每轮会在 `LlamaFactory/models/` 下产出一份新的 merged 模型。

> 说明：本任务是 self-distillation 方法论本质需要的"多轮数据自举"——用模型 M_k 生成数据 → 基于 base 重新训练得到 M_{k+1}。**训练链并非"M_k → 继续 SFT → M_{k+1}"，而是"base + D_k → M_k"**。这与用户偏好中"每个实验基于同一 base 模型独立训练"完全一致。

---

## 名称与路径约定（auto_workflow 需严格遵循）

- **基础模型**：`BASE_MODEL_NAME=DeepSeek-R1-Distill-Qwen-14B`
  路径：`BASE_MODEL_PATH=/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B`
- **第 k 轮产出的 merged 模型 tag**（k ∈ {1,2,3}）：
  ```
  ROUND_MODEL_TAG[k] = ${BASE_MODEL_NAME}_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint_r64_64_5e-5_${LR_SCHED}_bs_4_ep_3
  ```
  `LR_SCHED` 默认 `cosine`，可通过环境变量覆盖。
- **第 k 轮 merged 模型路径**：
  `/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/${ROUND_MODEL_TAG[k]}`
- **第 k 轮训练数据集名**（dataset_info.json 已注册）：
  `wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint`
- **第 k 轮训练 JSON 绝对路径**：
  `/apdcephfs_jn3/share_535475/common/dellwu/RE-START/data/wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint.json`

**各轮生成模型 M_k（数据生成时输入的模型）**
- `M_1 = DeepSeek-R1-Distill-Qwen-14B`（base）
- `M_k (k≥2) = ROUND_MODEL_TAG[k-1]`

**auto_run_pipline.sh 的产物路径（与原 run_pipline.sh 完全一致，仅复用）**
- `…/evaluate/results/wildjailbreak_train_${M_k}/merged/step2_${M_k}_self_align_v2_prefix_random_wildguard_0_5000.json`
- `…/evaluate/results/wildjailbreak_train_${M_k}/merged/step4_${M_k}_self_align_v2_prefix_0_hint16_wildguard_0_5000.json`

**auto_run_ultrafeedback.sh 的产物路径**
- `…/evaluate/results/ultrafeedback_train_1k_${M_k}/${M_k}_self_align_v2_prefix_0_wildguard.json`

**第 k 轮 file_list 组装规则**（严格对齐 `make_train_data.py` 内历史注释中"第一/二/三轮"的实际写法）

- 恒定三项：
  1. 第 k 轮 wildjailbreak step2：`…/evaluate/results/wildjailbreak_train_${M_k}/merged/step2_${M_k}_self_align_v2_prefix_random_wildguard_0_5000.json`
  2. 第 k 轮 wildjailbreak step4：`…/evaluate/results/wildjailbreak_train_${M_k}/merged/step4_${M_k}_self_align_v2_prefix_0_hint16_wildguard_0_5000.json`
  3. **base 模型的 ultrafeedback 产物**：`ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_wildguard.json`
- 当 k ≥ 2 时**固定追加一项**：
  4. **第 1 轮训练后模型**（即 `ROUND_MODEL_TAG[1]`，下称 `RESTART1_TAG`）对 ultrafeedback 的产物：
     `ultrafeedback_train_1k_${RESTART1_TAG}/${RESTART1_TAG}_self_align_v2_prefix_0_wildguard.json`

> ⚠️ **重要澄清**：从 k=2 起每轮都**只**追加 restart-1 模型的 ultrafeedback，**不**追加"上一轮"（M_k = ROUND_MODEL_TAG[k-1]）模型的 ultrafeedback。即 k=3 时 file_list 里不会出现 restart-2 模型产出的 ultrafeedback。该规则与 `make_train_data.py` 中第三轮注释行为完全一致（经用户 2026-05-06 确认）。

> 📌 **由此带来的副作用**：对于 k ≥ 3，auto_run_ultrafeedback.sh 的产出文件其实不再被下一轮 file_list 引用，但为保持"每轮流程完整对称"与未来扩展的灵活性，workflow 仍会在每轮 M_k 上执行一次 ultrafeedback 生成（不因 k≥3 而跳过）。如果用户希望跳过，可通过 `SKIP_STAGES="generate_ultrafeedback"` 加针对性轮次控制（本期不做该级别细分，保持简单）。

---

## 需求

### 需求 1：顶层编排脚本 `auto_run_iterative_workflow.sh`

**用户故事：** 作为一名研究员，我希望有一个 `auto_run_iterative_workflow.sh` 顶层脚本，一条命令完成 N 轮"生成-合并-训练-评估"，以便彻底摆脱每轮 5 处手工改代码。

#### 验收标准
1. WHEN 用户执行 `bash scripts/DeepSeek/auto_run_iterative_workflow.sh` THEN 系统 SHALL 按 k=1..NUM_ROUNDS 顺序依次完整执行所有轮次，无需任何人工干预。
2. WHEN 用户设置 `NUM_ROUNDS=N`（默认 3）THEN 系统 SHALL 执行 N 轮迭代。
3. WHEN 用户设置 `START_ROUND=k`（默认 1）THEN 系统 SHALL 从第 k 轮开始（断点续跑）。
4. IF 任意子阶段返回非零退出码 THEN 系统 SHALL 立即中止整个 workflow，打印失败的轮次、阶段、日志绝对路径，并以非零码退出。
5. WHEN workflow 启动时 THEN 系统 SHALL 打印"执行计划表"（每轮：输入模型 M_k、输出模型 tag、训练数据集名、训练 JSON 路径），并在进入每轮前 echo 分隔横幅。
6. WHEN workflow 全部完成 THEN 系统 SHALL 打印一份汇总表，列出每轮产出的模型路径和评估日志目录。
7. WHEN 顶层脚本调用任意子阶段 THEN 系统 SHALL 只调用 `auto_*` 前缀脚本，绝不调用原 5 个手工脚本。

### 需求 2：`auto_run_pipline.sh`（wildjailbreak 数据生成副本）

**用户故事：** 作为一名研究员，我希望有一份可接受环境变量的 `auto_run_pipline.sh`，在功能上与原脚本等价，以便 workflow 按轮次传入模型。

#### 验收标准
1. WHEN `auto_run_pipline.sh` 被调用 THEN 系统 SHALL 在业务行为（8 卡分片、Step 0-4 流程、产物命名）上与原 `run_pipline.sh` **完全一致**。
2. WHEN 调用时已设置 `MODEL_PATH` / `MODEL_TAG` 环境变量 THEN 系统 SHALL 使用该值作为生成模型；若未设置，SHALL 以非零码报错退出（auto 版本强制要求显式传入，不允许依赖硬编码）。
3. WHEN 调用时设置 `BASE_MODEL_TAG` 环境变量 THEN 系统 SHALL 使用该值确定 Step 0 缓存路径；若未设置 THEN 默认使用 `DeepSeek-R1-Distill-Qwen-14B`。
4. WHEN 第 2/3 轮调用时 Step 0 缓存已存在 THEN 系统 SHALL 自动跳过 Step 0（保留原脚本 FORCE_STEP0 逻辑）。
5. WHEN 脚本执行完毕 THEN 系统 SHALL 确保 step2 / step4 两个 merged JSON 产物存在。

### 需求 3：`auto_run_ultrafeedback.sh`（ultrafeedback 数据生成副本）

**用户故事：** 作为一名研究员，我希望有一份可接受环境变量的 `auto_run_ultrafeedback.sh` 副本，以便 workflow 按轮次调用。

#### 验收标准
1. WHEN `auto_run_ultrafeedback.sh` 被调用 THEN 系统 SHALL 在业务行为上与原 `run_ultrafeedback.sh` 完全一致（单卡 vLLM 生成 + 隐含的 wildguard moderation 产物命名）。
2. WHEN 调用时已设置 `MODEL_PATH` / `MODEL_TAG` 环境变量 THEN 系统 SHALL 使用该值；若未设置 SHALL 以非零码报错退出。
3. WHEN 指定 `GPU_ID` 环境变量 THEN 系统 SHALL 使用该 GPU；默认 GPU 0。
4. WHEN 执行完毕 THEN 系统 SHALL 确保 `${MODEL_TAG}_self_align_v2_prefix_0_wildguard.json` 产物存在（即原脚本命名 `${MODEL_TAG}_self_align_v2_prefix_0.json` 经 wildguard 后的产物，需确认原脚本是否在内部触发 moderation；若不触发则本 auto 版本 SHALL 追加一次 wildguard 评估以产出最终 `_wildguard.json` 文件，以匹配需求 5 的 file_list）。

> **说明给实现者**：原 `run_ultrafeedback.sh` 目前只跑生成未跑 moderation，但 `make_train_data.py` 注释里引用的是 `*_wildguard.json`。这意味着原流程中 ultrafeedback 的 moderation 是**另外手动跑的**。`auto_run_ultrafeedback.sh` 必须将 moderation 也一并集成，使产物与 `make_train_data.py` 预期一致。

### 需求 4：`auto_make_train_data.py`（数据合并副本）

**用户故事：** 作为一名研究员，我希望有一份支持 CLI 参数的 `auto_make_train_data.py`，原 `make_train_data.py` 保留作为人工脚本。

#### 验收标准
1. WHEN `auto_make_train_data.py` 以 `--file_list A.json B.json C.json --output_file /path/to/out.json` 被调用 THEN 系统 SHALL 按原 `process_data()` 的过滤 + messages 构建逻辑写出 `out.json`。
2. WHEN `--file_list` 或 `--output_file` 任一缺失 THEN 系统 SHALL 以非零码报错退出并打印 usage（auto 版本强制 CLI，不支持回退硬编码）。
3. IF 某个输入文件不存在或为空 THEN 系统 SHALL 打印 warning 并继续处理剩余文件；IF 所有文件均缺失 THEN 系统 SHALL 以非零码退出。
4. WHEN 完成 THEN 系统 SHALL 在 stdout 打印最终条目数与输出文件绝对路径。
5. `auto_make_train_data.py` 的过滤规则、messages 构建、think 字段拼接逻辑与原 `make_train_data.py` **完全一致**。

### 需求 5：`auto_sft_peft_train.sh`（训练副本）

**用户故事：** 作为一名研究员，我希望有一份可接受环境变量的训练脚本副本，以便 workflow 按轮传入 `dataset_name`。

#### 验收标准
1. WHEN `auto_sft_peft_train.sh` 以环境变量 `DATASET_NAME` 调用 THEN 系统 SHALL 使用该值替换原脚本内硬编码的 `dataset_name`。
2. WHEN 设置 `BASE_MODEL_NAME` / `MODEL_NAME_OR_PATH` 环境变量 THEN 系统 SHALL 优先使用这些值；未设置则 SHALL 使用默认 base 模型（`DeepSeek-R1-Distill-Qwen-14B`）。
3. WHEN 设置 `LR_SCHEDULER_TYPE` 环境变量 THEN 系统 SHALL 使用该值；默认 `cosine`。
4. WHEN `DATASET_NAME` 未设置 THEN 系统 SHALL 以非零码报错退出（auto 版本强制 CLI）。
5. WHEN 训练 + export 均成功 THEN 系统 SHALL 确保 `/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/${target_model_name}` 目录内至少含 `config.json` 与权重文件。
6. WHEN workflow 启动第 k 轮训练前 THEN workflow SHALL 校验 `dataset_info.json` 中存在键 `wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint`；若缺失则中止。
7. IF 第 k 轮目标模型目录已存在且非空 THEN workflow SHALL 默认跳过该轮训练，除非 `FORCE_TRAIN=1`。
8. `auto_sft_peft_train.sh` 的所有训练超参（lora_rank/alpha=64、lr=5e-5、bs=4、epoch=3、cutoff_len=4096、fp16、template=deepseek3、DEVICES=0-7）必须与原 `sft_peft_train.sh` **完全一致**。

### 需求 6：`auto_run_evaluation_pool.sh`（评估副本）

**用户故事：** 作为一名研究员，我希望有一份可接受环境变量 `model_list` 的评估脚本副本，以便 workflow 按轮评估当轮新模型。

#### 验收标准
1. WHEN `auto_run_evaluation_pool.sh` 以 `MODEL_LIST_OVERRIDE="name1|path1;name2|path2"` 被调用 THEN 系统 SHALL 以该值解析出的数组作为 `model_list`。
2. WHEN `MODEL_LIST_OVERRIDE` 未设置 THEN 系统 SHALL 以非零码报错退出（auto 版本强制 CLI）。
3. WHEN workflow 第 k 轮训练成功 THEN workflow SHALL 以 `MODEL_LIST_OVERRIDE="${ROUND_MODEL_TAG[k]}|${PATH_k}"` 调用该脚本。
4. IF 第 k 轮期望的评估产物（通过检查约定的 results 文件是否存在）已完整 THEN workflow SHALL 默认跳过该轮评估，除非 `FORCE_EVAL=1`。
5. 评估的 dataset 列表、judge 模型、CodeAttack 作业、GPU 池调度等逻辑必须与原 `run_evaluation_pool.sh` **完全一致**。

### 需求 7：日志、可观测性与幂等性

**用户故事：** 作为一名研究员，我希望 workflow 的每轮每阶段日志集中归档，并具备幂等能力，以便长任务中途失败时能精准定位问题并断点续跑。

#### 验收标准
1. WHEN workflow 启动 THEN 系统 SHALL 创建目录 `…/RE-START/logs/auto_workflow_$(date +%Y%m%d_%H%M%S)/` 作为本次运行的总日志根。
2. WHEN 每个阶段（`generate_wildjailbreak` / `generate_ultrafeedback` / `make_train_data` / `train` / `eval`）运行 THEN 系统 SHALL 将 stdout+stderr 重定向到 `${LOG_ROOT}/round${k}/${stage}.log`，同时在终端打印关键进度摘要。
3. WHEN 阶段失败 THEN 系统 SHALL 在终端明确提示失败 log 的绝对路径。
4. WHEN 设置 `DRY_RUN=1` THEN 系统 SHALL 仅打印每条将执行的命令及解析后的参数，不实际调用任何子脚本。
5. WHEN 设置 `SKIP_STAGES="generate,merge_data"`（逗号分隔）THEN 系统 SHALL 在每轮内跳过相应阶段。
6. 所有幂等判定必须基于"关键产物文件存在且非空"，不检查文件内容正确性；用户可用 `FORCE_TRAIN=1` / `FORCE_EVAL=1` / `FORCE_GEN=1` 强制重跑。

### 需求 8：前置校验与友好错误提示

**用户故事：** 作为一名研究员，我希望 workflow 启动时做关键前置检查，快速失败，而非数小时训练后才报错。

#### 验收标准
1. WHEN workflow 启动 THEN 系统 SHALL 校验：
   - 所有 `auto_*` 子脚本存在且可读（6 个文件）
   - `BASE_MODEL_PATH` 指向的基础模型目录存在
   - `LlamaFactory/data/dataset_info.json` 存在且包含 restart-1..NUM_ROUNDS 的 N 个键
   - `qwen3_env` 与 `llamafactory_env` 的 python 可执行（路径 `-f` 判定）
   - GPU 压测脚本 `/apdcephfs_jn3/share_535475/common/dellwu/OpenRubric/gpu_stress_8gpu.py` 存在且可读（配合需求 9）
2. IF 任一检查失败 THEN 系统 SHALL 打印具体缺失项并以非零码退出。
3. WHEN 所有检查通过 THEN 系统 SHALL 打印 `[preflight] OK` 并继续。

### 需求 9：workflow 结束后自动占满 GPU（长期占卡收尾）

**用户故事：** 作为一名共享集群用户，我希望 workflow 无论成功完成还是中途失败，都自动在当前节点启动一段 GPU 压测程序，把 GPU 利用率刷满，以便避免空闲时被调度器或他人抢占节点。

#### 验收标准
1. WHEN workflow 正常执行完所有轮次 THEN 系统 SHALL 在退出前执行命令：
   ```
   python /apdcephfs_jn3/share_535475/common/dellwu/OpenRubric/gpu_stress_8gpu.py
   ```
   并将该进程的 stdout/stderr 追加到 `${LOG_ROOT}/gpu_stress.log`。
2. WHEN workflow 因任一阶段失败而中止（包括 preflight 失败之外的任意非零退出）THEN 系统 SHALL 在打印失败信息 **之后**、退出 **之前**，同样执行第 1 项中的 GPU 压测命令。
3. IF `preflight` 阶段自身失败（即 workflow 还未进入任何一轮） THEN 系统 **仍 SHALL** 启动 GPU 压测命令，以满足用户"任一失败情况都占卡"的诉求。
4. WHEN 设置环境变量 `SKIP_GPU_STRESS=1` THEN 系统 SHALL 跳过 GPU 压测（用于调试 / DRY_RUN 场景）。
5. WHEN `DRY_RUN=1` THEN 系统 SHALL 默认跳过 GPU 压测，只打印"would run: python …/gpu_stress_8gpu.py"。
6. GPU 压测必须以 **前台阻塞方式** 运行（即 workflow 脚本会长期挂在该命令上，直至用户手动 Ctrl+C 中止），从而持续占用 GPU。不使用 `nohup` / `&` 后台化。
7. WHEN workflow 收到 SIGINT (Ctrl+C) 处于非 GPU 压测阶段 THEN 系统 SHALL 先打印中止原因再启动 GPU 压测（通过 `trap` 机制实现）。

## 非功能性要求与约束

1. **零侵入原脚本**：原 5 个脚本 + `make_train_data.py` **绝不修改**。所有变更落在 `auto_*` 前缀的新文件中。
2. **auto_\* 文件之间的行为一致性**：除参数化接口与编排集成外，auto_* 脚本的业务逻辑（超参、产物命名、模型调用顺序等）必须与对应原脚本一致，确保 workflow 产出可被用户旧的后续分析流程复用。
3. **Shell 兼容**：顶层 workflow 与 auto_* 脚本均使用 bash。
4. **不修改 `dataset_info.json`**：restart-1/2/3 已预注册，只做存在性校验。
5. **不引入新第三方依赖**：仅复用现有 conda env（`qwen3_env`、`llamafactory_env`）与现有 Python 脚本。

---

## 边界情况

- **Step 0 缓存复用**：所有轮次 `auto_run_pipline.sh` 都固定 `BASE_MODEL_TAG=DeepSeek-R1-Distill-Qwen-14B`，第 2/3 轮直接命中 Step 0 缓存。
- **ultrafeedback base 模型产物缺失**：第 1 轮 `file_list` 依赖 `ultrafeedback_train_1k_DeepSeek-R1-Distill-Qwen-14B/DeepSeek-R1-Distill-Qwen-14B_self_align_v2_prefix_0_wildguard.json`。workflow 启动时如检测到该文件缺失，SHALL 在第 1 轮 `generate_ultrafeedback` 阶段前先为 base 模型触发一次 `auto_run_ultrafeedback.sh`，以保证 file_list 完整。
- **`*_wildguard.json` 命名一致性**：原 `run_ultrafeedback.sh` 仅生成 `${MODEL_TAG}_self_align_v2_prefix_0.json`（不带 `_wildguard` 后缀），但 `make_train_data.py` 期望 `_wildguard.json`。`auto_run_ultrafeedback.sh` 必须在生成后**追加一次 wildguard moderation 判定**，使最终产物命名为 `${MODEL_TAG}_self_align_v2_prefix_0_wildguard.json`，填补这一缺口。
- **幂等误判**：判定是否"已完成"仅通过关键产物文件 **存在 + 非空**；内容可疑时用户应用 `FORCE_*=1` 强制重跑。
- **磁盘空间**：3 轮合计 3 份 merged 模型（每份约 30+GB）+ 大量中间 JSON，workflow 启动时打印友好提示但不硬校验。
- **GPU 占用**：`auto_run_pipline.sh`（8 卡）与 `auto_run_ultrafeedback.sh`（单卡）由 workflow **串行调用**，避免 GPU 争用。用户如需并行，可自行修改顶层脚本（不在本期需求内）。
- **GPU 压测收尾的前台阻塞行为**：需求 9 的 GPU 压测是**前台阻塞**运行，意味着 workflow 完成或失败后终端不会立即交还给用户，这是用户主动的"占卡"诉求。用户如无此诉求，可设置 `SKIP_GPU_STRESS=1` 跳过。
