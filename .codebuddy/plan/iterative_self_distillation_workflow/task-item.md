# 实施计划

- [ ] 1. 克隆并参数化 `scripts/DeepSeek/auto_run_pipline.sh`
  - 基于原 `scripts/DeepSeek/run_pipline.sh` 完整复制一份新文件 `auto_run_pipline.sh`
  - 脚本头部增加 `MODEL_PATH` / `MODEL_TAG` 环境变量读取分支，任一缺失时打印错误并非零码退出
  - 增加 `BASE_MODEL_TAG` 环境变量读取，默认值 `DeepSeek-R1-Distill-Qwen-14B`，保留原 `FORCE_STEP0` 缓存复用逻辑
  - 保留 Step 0~4 的 8 卡分片流程与所有产物命名（step2/step4 merged JSON）完全一致
  - 末尾新增 step2 / step4 merged JSON 的"存在且非空"校验，缺失则以非零码退出
  - _需求：2.1、2.2、2.3、2.4、2.5_

- [ ] 2. 克隆并参数化 `scripts/DeepSeek/auto_run_ultrafeedback.sh`（集成 wildguard moderation）
  - 基于原 `scripts/DeepSeek/run_ultrafeedback.sh` 完整复制一份新文件 `auto_run_ultrafeedback.sh`
  - 脚本头部读取 `MODEL_PATH` / `MODEL_TAG` / `GPU_ID`（默认 0）；`MODEL_PATH` 或 `MODEL_TAG` 缺失时非零码退出
  - 保留原 vLLM 单卡生成逻辑，业务行为与原脚本一致，产出 `${MODEL_TAG}_self_align_v2_prefix_0.json`
  - **新增步骤**：在生成完成后追加一次 wildguard moderation（调用仓库内现有 wildguard 评估入口），使最终产物为 `${MODEL_TAG}_self_align_v2_prefix_0_wildguard.json`；若 `_wildguard.json` 已存在则跳过该步
  - 末尾校验 `_wildguard.json` 存在且非空，缺失则以非零码退出
  - _需求：3.1、3.2、3.3、3.4、边界情况"*_wildguard.json 命名一致性"_

- [ ] 3. 创建 CLI 版 `data/auto_make_train_data.py`
  - 从原 `data/make_train_data.py` 迁移 `process_data()` 过滤/messages/think 拼接逻辑到新文件，行为完全一致
  - 使用 `argparse` 增加 `--file_list`（nargs='+'，必填）与 `--output_file`（必填）参数；任一缺失则打印 usage 并非零码退出
  - 单个输入文件不存在或为空时打印 warning 并继续处理；全部缺失时非零码退出
  - 成功写出后在 stdout 打印最终条目数与输出文件绝对路径
  - _需求：4.1、4.2、4.3、4.4、4.5_

- [ ] 4. 克隆并参数化 `scripts/DeepSeek/auto_sft_peft_train.sh`
  - 基于原 `scripts/DeepSeek/sft_peft_train.sh` 完整复制一份新文件 `auto_sft_peft_train.sh`
  - 读取环境变量 `DATASET_NAME`（缺失则非零码退出）/ `BASE_MODEL_NAME` / `MODEL_NAME_OR_PATH` / `LR_SCHEDULER_TYPE`（默认 `cosine`）
  - 保留全部训练超参（lora r/alpha=64、lr=5e-5、bs=4、epoch=3、cutoff_len=4096、fp16、template=deepseek3、DEVICES=0-7）
  - 保留 train + export 两阶段流程，结束后校验 `LlamaFactory/models/${target_model_name}/config.json` 与权重文件存在
  - _需求：5.1、5.2、5.3、5.4、5.5、5.8_

- [ ] 5. 克隆并参数化 `scripts/DeepSeek/auto_run_evaluation_pool.sh`
  - 基于原 `scripts/DeepSeek/run_evaluation_pool.sh` 完整复制一份新文件 `auto_run_evaluation_pool.sh`
  - 脚本头部解析 `MODEL_LIST_OVERRIDE` 环境变量（格式 `name1|path1;name2|path2`），填充至原 `model_list` 数组；未设置时以非零码退出
  - 保留原 dataset 列表、judge 模型、CodeAttack 作业、GPU 池调度等全部逻辑
  - _需求：6.1、6.2、6.5_

- [ ] 6. 实现顶层编排脚本骨架 `scripts/DeepSeek/auto_run_iterative_workflow.sh`（常量/路径函数/计划表/汇总表）
  - 新建 `auto_run_iterative_workflow.sh`，定义常量 `BASE_MODEL_NAME` / `BASE_MODEL_PATH` / `LR_SCHED`（读 `LR_SCHEDULER_TYPE`，默认 `cosine`）
  - 读取环境变量：`NUM_ROUNDS`（默认 3）/ `START_ROUND`（默认 1）/ `DRY_RUN` / `SKIP_STAGES` / `FORCE_TRAIN` / `FORCE_EVAL` / `FORCE_GEN` / `SKIP_GPU_STRESS`
  - 实现辅助函数：`compute_round_model_tag(k)` / `compute_input_model_tag_for_round(k)` / `compute_file_list_for_round(k)`；后者严格按需求实现"恒定三项 + k≥2 追加 restart-1 模型 ultrafeedback"规则（**不追加上一轮 M_k 的 ultrafeedback**）
  - 启动时打印"执行计划表"：列出每轮 M_k、输出 tag、训练数据集名、训练 JSON 路径、file_list 预览
  - 完成后打印汇总表：每轮产出的模型路径与评估日志目录
  - _需求：1.1、1.2、1.3、1.5、1.6、1.7，名称与路径约定中 file_list 组装规则_

- [ ] 7. 在顶层脚本中实现 `preflight()` 前置校验
  - 校验 6 个 `auto_*` 文件（5 个阶段脚本 + 顶层脚本自身）存在且可读
  - 校验 `BASE_MODEL_PATH` 目录存在
  - 用 python3 解析 `LlamaFactory/data/dataset_info.json`，校验 `wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint`（k=1..NUM_ROUNDS）N 个键均存在
  - 校验 `qwen3_env` / `llamafactory_env` 对应 python 可执行文件存在（`-f` 判定）
  - 校验 GPU 压测脚本 `/apdcephfs_jn3/share_535475/common/dellwu/OpenRubric/gpu_stress_8gpu.py` 存在可读
  - 任一失败则打印缺失项并非零码退出（之后仍会触发需求 9 的 GPU 压测，由任务 9 的 trap 机制保证）；全部通过则打印 `[preflight] OK`
  - _需求：8.1、8.2、8.3_

- [ ] 8. 在顶层脚本中实现日志基础设施与阶段执行封装
  - 启动时创建 `logs/auto_workflow_$(date +%Y%m%d_%H%M%S)/` 作为 `LOG_ROOT`，并在 workflow 开头 `export` 出去供所有子脚本引用
  - 实现 `run_stage(round_k, stage_name, cmd...)`：将 stdout+stderr 重定向至 `${LOG_ROOT}/round${k}/${stage}.log`，终端打印阶段开始/结束摘要
  - 阶段失败时打印失败 round/stage 与 log 绝对路径，并以非零码退出整个 workflow
  - `DRY_RUN=1` 时仅打印命令与解析后参数、不实际执行；`SKIP_STAGES` 命中阶段名时跳过
  - 进入每轮前 echo 分隔横幅 `===== Round k =====`
  - _需求：1.4、7.1、7.2、7.3、7.4、7.5、7.6_

- [ ] 9. 在顶层脚本中实现 GPU 压测收尾（trap 机制）
  - 定义 `gpu_stress_finalize()`：当 `SKIP_GPU_STRESS=1` 或 `DRY_RUN=1` 时打印 "would run: python …/gpu_stress_8gpu.py" 并 return；否则以**前台阻塞**方式执行 `python /apdcephfs_jn3/share_535475/common/dellwu/OpenRubric/gpu_stress_8gpu.py`，stdout/stderr 追加到 `${LOG_ROOT}/gpu_stress.log`
  - 在脚本头部（`set -e` 之后）注册 `trap gpu_stress_finalize EXIT`，确保无论 workflow 正常结束、preflight 失败、任一阶段失败、Ctrl+C（SIGINT）都会触发 GPU 压测
  - 对 SIGINT 追加一个 `trap` 句柄：先打印"workflow interrupted"信息再走 EXIT trap
  - 确保失败分支在打印失败日志路径之后才触发 EXIT trap（通过 `exit <nonzero>` 顺序保证）
  - _需求：9.1、9.2、9.3、9.4、9.5、9.6、9.7_

- [ ] 10. 在顶层脚本中实现每轮四阶段主循环
  - 主循环 `for k in $(seq $START_ROUND $NUM_ROUNDS)` 中依次调用 5 个阶段：
    - **pre-round-1 保障**：若 k=1 且 base 模型的 `ultrafeedback_train_1k_${BASE_MODEL_NAME}/${BASE_MODEL_NAME}_self_align_v2_prefix_0_wildguard.json` 缺失，则先以 `MODEL_PATH=${BASE_MODEL_PATH} MODEL_TAG=${BASE_MODEL_NAME}` 调用一次 `auto_run_ultrafeedback.sh` 补齐
    - **generate_wildjailbreak**：通过环境变量传 `MODEL_PATH`/`MODEL_TAG=M_k` + 固定 `BASE_MODEL_TAG=DeepSeek-R1-Distill-Qwen-14B`，调用 `auto_run_pipline.sh`；step2/step4 产物已存在且非空时跳过（除非 `FORCE_GEN=1`）
    - **generate_ultrafeedback**：传 `MODEL_PATH`/`MODEL_TAG=M_k` 调用 `auto_run_ultrafeedback.sh`；`_wildguard.json` 已存在且非空时跳过（除非 `FORCE_GEN=1`）
    - **make_train_data**：按 `compute_file_list_for_round(k)` 生成 file_list（**k≥2 时仅追加 restart-1 的 ultrafeedback，不追加 M_k 的 ultrafeedback**），`--output_file=…/data/wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint.json`，调用 `auto_make_train_data.py`
    - **train**：校验 `dataset_info.json` 含 restart-${k} 键；目标模型目录非空且 `FORCE_TRAIN≠1` 则跳过；否则通过环境变量 `DATASET_NAME`/`BASE_MODEL_NAME`/`MODEL_NAME_OR_PATH=${BASE_MODEL_PATH}`/`LR_SCHEDULER_TYPE=${LR_SCHED}` 调用 `auto_sft_peft_train.sh`
    - **eval**：构造 `MODEL_LIST_OVERRIDE="${ROUND_MODEL_TAG[k]}|${PATH_k}"` 调用 `auto_run_evaluation_pool.sh`；已完成且 `FORCE_EVAL≠1` 则跳过
  - 最后执行 `DRY_RUN=1` 烟雾测试，校验 3 轮计划表中 M_k / 输出 tag / file_list（尤其 k=3 时不含 restart-2 ultrafeedback）/ 训练 JSON 路径均符合需求约定，并验证 EXIT trap 能触发 GPU 压测（用 `SKIP_GPU_STRESS=1` 避免真跑）
  - _需求：1.1、1.7、2.*、3.*、4.*、5.6、5.7、6.3、6.4，名称与路径约定，边界情况"Step 0 缓存复用"/"ultrafeedback base 模型产物缺失"_
