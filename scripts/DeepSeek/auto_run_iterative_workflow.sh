#!/bin/bash
# =============================================================================
# [AUTO] Iterative self-distillation workflow orchestrator
# -----------------------------------------------------------------------------
# 顶层编排脚本：一条命令完成 N 轮 "generate -> merge_data -> train -> eval" 迭代。
#
# 迭代语义（第 k 轮, k = 1..NUM_ROUNDS）：
#   - 输入模型 M_k:
#       k == 1 -> base 模型 DeepSeek-R1-Distill-Qwen-14B
#       k >= 2 -> ROUND_MODEL_TAG[k-1]（上一轮训练产出的 merged 模型）
#   - 第 k 轮基于 base 模型 + 第 k 轮训练数据 独立训练出 ROUND_MODEL_TAG[k]
#     （链式的是"数据生成所用的模型"，训练的 base 始终不变）
#
# 文件清单：
#   - scripts/DeepSeek/auto_run_pipline.sh
#   - scripts/DeepSeek/auto_run_ultrafeedback.sh
#   - data/auto_make_train_data.py
#   - scripts/DeepSeek/auto_sft_peft_train.sh
#   - scripts/DeepSeek/auto_run_evaluation_pool.sh
#   - scripts/DeepSeek/auto_run_iterative_workflow.sh  (this file)
#
# 常用参数（全部通过环境变量传入）：
#   NUM_ROUNDS       迭代总轮数（默认 3）
#   START_ROUND      起始轮次（默认 1，用于断点续跑）
#   LR_SCHEDULER_TYPE 默认 cosine
#   FORCE_GEN        =1 时强制重跑生成阶段（即使产物已存在）
#   FORCE_TRAIN      =1 时强制重跑训练阶段
#   FORCE_EVAL       =1 时强制重跑评估阶段
#   SKIP_STAGES      逗号分隔，允许 generate_wildjailbreak / generate_ultrafeedback
#                    / make_train_data / train / eval
#   DRY_RUN          =1 时仅打印每条命令，不实际执行
#   SKIP_GPU_STRESS  =1 时跳过结尾的 GPU 占卡程序
# =============================================================================

set -uo pipefail

# =============================================================================
# Constants (path & naming conventions; must match requirements.md)
# =============================================================================
REPO_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START"
LLAMA_FACTORY_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory"
LLAMA_FACTORY_MODELS_DIR="${LLAMA_FACTORY_ROOT}/models"
DATASET_INFO_JSON="${LLAMA_FACTORY_ROOT}/data/dataset_info.json"

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL_PATH="/apdcephfs_nj4/share_300616873/hunyuan/external/${BASE_MODEL_NAME}"

SCRIPTS_DIR="${REPO_ROOT}/scripts/DeepSeek"
DATA_DIR="${REPO_ROOT}/data"

AUTO_PIPLINE_SH="${SCRIPTS_DIR}/auto_run_pipline.sh"
AUTO_ULTRAFEEDBACK_SH="${SCRIPTS_DIR}/auto_run_ultrafeedback.sh"
AUTO_MAKE_TRAIN_DATA_PY="${DATA_DIR}/auto_make_train_data.py"
AUTO_SFT_PEFT_TRAIN_SH="${SCRIPTS_DIR}/auto_sft_peft_train.sh"
AUTO_EVAL_POOL_SH="${SCRIPTS_DIR}/auto_run_evaluation_pool.sh"
SELF_SCRIPT="${SCRIPTS_DIR}/auto_run_iterative_workflow.sh"

GPU_STRESS_PY="/apdcephfs_jn3/share_535475/common/dellwu/OpenRubric/gpu_stress_8gpu.py"

QWEN3_ENV_PY="/apdcephfs_jn3/share_535475/common/dellwu/envs/qwen3_env/bin/python"
LLAMAFACTORY_ENV_PY="/apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env/bin/python"

# =============================================================================
# Environment-variable-driven knobs
# =============================================================================
NUM_ROUNDS="${NUM_ROUNDS:-3}"
START_ROUND="${START_ROUND:-1}"
LR_SCHED="${LR_SCHEDULER_TYPE:-constant}"
FORCE_GEN="${FORCE_GEN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
SKIP_STAGES="${SKIP_STAGES:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_GPU_STRESS="${SKIP_GPU_STRESS:-0}"

# =============================================================================
# Logging infrastructure (requirement 7)
# =============================================================================
LOG_ROOT="${REPO_ROOT}/logs/auto_workflow_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"
export LOG_ROOT

WORKFLOW_STATUS="unknown"
WORKFLOW_FAIL_ROUND=""
WORKFLOW_FAIL_STAGE=""
WORKFLOW_FAIL_LOG=""

# -----------------------------------------------------------------------------
# gpu_stress_finalize: executed via EXIT trap. Runs gpu_stress_8gpu.py in
# foreground-blocking mode, so the shell stays resident after workflow exits.
# -----------------------------------------------------------------------------
gpu_stress_finalize() {
    local rc="$?"   # preserve original exit code

    echo ""
    echo "========================================"
    echo "[workflow] exiting (rc=${rc}), status=${WORKFLOW_STATUS}"
    if [[ -n "${WORKFLOW_FAIL_ROUND}" ]]; then
        echo "  failed round  : ${WORKFLOW_FAIL_ROUND}"
        echo "  failed stage  : ${WORKFLOW_FAIL_STAGE}"
        echo "  failure log   : ${WORKFLOW_FAIL_LOG}"
    fi
    echo "  log root      : ${LOG_ROOT}"
    echo "========================================"

    if [[ "${SKIP_GPU_STRESS}" == "1" || "${DRY_RUN}" == "1" ]]; then
        echo "[gpu_stress] SKIPPED (SKIP_GPU_STRESS=${SKIP_GPU_STRESS}, DRY_RUN=${DRY_RUN})"
        echo "[gpu_stress] would run: python ${GPU_STRESS_PY}"
        return 0
    fi

    echo "[gpu_stress] starting foreground-blocking GPU saturation..."
    echo "[gpu_stress] command: python ${GPU_STRESS_PY}"
    echo "[gpu_stress] log    : ${LOG_ROOT}/gpu_stress.log"
    echo "[gpu_stress] press Ctrl+C to release the node."

    # Run in foreground so the script blocks here indefinitely, occupying GPUs.
    # Stdout/stderr appended to gpu_stress.log while still echoed to terminal
    # via tee for visibility.
    python "${GPU_STRESS_PY}" 2>&1 | tee -a "${LOG_ROOT}/gpu_stress.log"
}

on_interrupt() {
    echo ""
    echo "[workflow] *** interrupted by SIGINT (Ctrl+C) ***"
    WORKFLOW_STATUS="interrupted"
    # Exit triggers EXIT trap -> gpu_stress_finalize
    exit 130
}

trap gpu_stress_finalize EXIT
trap on_interrupt INT

# =============================================================================
# Helper: path / tag / file_list computation (requirement 6, naming convention)
# =============================================================================

# ROUND_MODEL_TAG[k]
compute_round_model_tag() {
    local k="$1"
    echo "${BASE_MODEL_NAME}_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint_r64_64_5e-5_${LR_SCHED}_bs_4_ep_3"
}

compute_round_model_path() {
    local k="$1"
    echo "${LLAMA_FACTORY_MODELS_DIR}/$(compute_round_model_tag "${k}")"
}

# M_k: data-generation input model
compute_input_model_tag_for_round() {
    local k="$1"
    if [[ "${k}" -eq 1 ]]; then
        echo "${BASE_MODEL_NAME}"
    else
        compute_round_model_tag "$((k - 1))"
    fi
}

compute_input_model_path_for_round() {
    local k="$1"
    if [[ "${k}" -eq 1 ]]; then
        echo "${BASE_MODEL_PATH}"
    else
        compute_round_model_path "$((k - 1))"
    fi
}

# Dataset name key registered in dataset_info.json
compute_dataset_name() {
    local k="$1"
    echo "wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint"
}

compute_train_json_path() {
    local k="$1"
    echo "${DATA_DIR}/wildjailbreak_ultrafeedback_DS-R1-14B_restart-${k}_with_hint.json"
}

# Wildjailbreak step2/step4 produced by auto_run_pipline.sh for model M_k
compute_wj_step2_path() {
    local mk_tag="$1"
    echo "${REPO_ROOT}/evaluate/results/wildjailbreak_train_${mk_tag}/merged/step2_${mk_tag}_self_align_v2_prefix_random_wildguard_0_5000.json"
}
compute_wj_step4_path() {
    local mk_tag="$1"
    echo "${REPO_ROOT}/evaluate/results/wildjailbreak_train_${mk_tag}/merged/step4_${mk_tag}_self_align_v2_prefix_0_hint16_wildguard_0_5000.json"
}

# Ultrafeedback wildguard path for any model tag
compute_uf_wildguard_path() {
    local mk_tag="$1"
    echo "${REPO_ROOT}/evaluate/results/ultrafeedback_train_1k_${mk_tag}/${mk_tag}_self_align_v2_prefix_0_wildguard.json"
}

# -----------------------------------------------------------------------------
# compute_file_list_for_round(k): emit one path per line.
#
# Rule (2026-05-06 user-confirmed):
#   Always: step2(M_k), step4(M_k), uf_wildguard(base_model)
#   When k >= 2: append uf_wildguard(ROUND_MODEL_TAG[1])  -- restart-1 only
#   Do NOT append uf_wildguard of "previous round" (ROUND_MODEL_TAG[k-1])
# -----------------------------------------------------------------------------
compute_file_list_for_round() {
    local k="$1"
    local mk_tag; mk_tag="$(compute_input_model_tag_for_round "${k}")"
    compute_wj_step2_path "${mk_tag}"
    compute_wj_step4_path "${mk_tag}"
    compute_uf_wildguard_path "${BASE_MODEL_NAME}"
    if [[ "${k}" -ge 2 ]]; then
        local restart1_tag; restart1_tag="$(compute_round_model_tag 1)"
        compute_uf_wildguard_path "${restart1_tag}"
    fi
}

# =============================================================================
# SKIP_STAGES helper
# =============================================================================
should_skip_stage() {
    local stage_name="$1"
    [[ -z "${SKIP_STAGES}" ]] && return 1
    local IFS=','
    local s
    for s in ${SKIP_STAGES}; do
        [[ "${s}" == "${stage_name}" ]] && return 0
    done
    return 1
}

# =============================================================================
# run_stage: wrap one stage's execution with logging + DRY_RUN support.
# -----------------------------------------------------------------------------
# Usage:  run_stage <round_k> <stage_name> -- cmd args...
# Environment-variable assignments in cmd are supported via `env`.
# -----------------------------------------------------------------------------
run_stage() {
    local k="$1"; shift
    local stage_name="$1"; shift
    # remaining args are the command
    local round_log_dir="${LOG_ROOT}/round${k}"
    mkdir -p "${round_log_dir}"
    local log_file="${round_log_dir}/${stage_name}.log"

    echo ""
    echo "---- [round ${k}] stage '${stage_name}' start (log: ${log_file}) ----"

    if should_skip_stage "${stage_name}"; then
        echo "[round ${k}] stage '${stage_name}' SKIPPED (via SKIP_STAGES)"
        return 0
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] would execute:"
        printf '    %q ' "$@"
        printf '\n'
        return 0
    fi

    # Execute; tee for mild terminal visibility, full log in file.
    set +e
    ( "$@" ) >"${log_file}" 2>&1
    local rc=$?
    set -e

    if [[ "${rc}" -ne 0 ]]; then
        WORKFLOW_STATUS="failed"
        WORKFLOW_FAIL_ROUND="${k}"
        WORKFLOW_FAIL_STAGE="${stage_name}"
        WORKFLOW_FAIL_LOG="${log_file}"
        echo "[FATAL] round ${k} / stage '${stage_name}' failed (rc=${rc})."
        echo "        see log: ${log_file}"
        echo "        tail of log:"
        tail -n 50 "${log_file}" | sed 's/^/        | /'
        exit "${rc}"
    fi

    echo "[round ${k}] stage '${stage_name}' OK"
}

# =============================================================================
# Preflight checks (requirement 8)
# =============================================================================
preflight() {
    echo "========================================"
    echo "[preflight] running pre-flight checks..."

    local fail=0

    # (a) auto_* scripts readable
    local script
    for script in \
        "${AUTO_PIPLINE_SH}" \
        "${AUTO_ULTRAFEEDBACK_SH}" \
        "${AUTO_MAKE_TRAIN_DATA_PY}" \
        "${AUTO_SFT_PEFT_TRAIN_SH}" \
        "${AUTO_EVAL_POOL_SH}" \
        "${SELF_SCRIPT}"; do
        if [[ ! -r "${script}" ]]; then
            echo "  [MISSING] script not readable: ${script}"
            fail=1
        fi
    done

    # (b) base model dir
    if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
        echo "  [MISSING] BASE_MODEL_PATH dir not found: ${BASE_MODEL_PATH}"
        fail=1
    fi

    # (c) dataset_info.json keys
    if [[ ! -f "${DATASET_INFO_JSON}" ]]; then
        echo "  [MISSING] dataset_info.json not found: ${DATASET_INFO_JSON}"
        fail=1
    else
        local missing_keys
        missing_keys=$(python3 - "${DATASET_INFO_JSON}" "${NUM_ROUNDS}" <<'PYEOF'
import json, sys
p = sys.argv[1]
n = int(sys.argv[2])
try:
    with open(p, "r", encoding="utf-8") as f:
        info = json.load(f)
except Exception as e:
    print(f"__load_error__:{e}")
    sys.exit(0)
missing = []
for k in range(1, n + 1):
    key = f"wildjailbreak_ultrafeedback_DS-R1-14B_restart-{k}_with_hint"
    if key not in info:
        missing.append(key)
print(",".join(missing))
PYEOF
        )
        if [[ "${missing_keys}" == __load_error__:* ]]; then
            echo "  [ERROR] cannot parse dataset_info.json: ${missing_keys}"
            fail=1
        elif [[ -n "${missing_keys}" ]]; then
            echo "  [MISSING] dataset_info.json keys missing: ${missing_keys}"
            fail=1
        fi
    fi

    # (d) conda envs' python
    if [[ ! -f "${QWEN3_ENV_PY}" ]]; then
        echo "  [MISSING] qwen3_env python: ${QWEN3_ENV_PY}"
        fail=1
    fi
    if [[ ! -f "${LLAMAFACTORY_ENV_PY}" ]]; then
        echo "  [MISSING] llamafactory_env python: ${LLAMAFACTORY_ENV_PY}"
        fail=1
    fi

    # (e) gpu stress script
    if [[ ! -r "${GPU_STRESS_PY}" ]]; then
        echo "  [MISSING] gpu_stress script: ${GPU_STRESS_PY}"
        fail=1
    fi

    if (( fail )); then
        WORKFLOW_STATUS="preflight_failed"
        echo "[preflight] FAILED. aborting (GPU stress finalizer will still run)."
        exit 10
    fi
    echo "[preflight] OK"
    echo "========================================"
}

# =============================================================================
# Execution plan printout (requirement 1.5)
# =============================================================================
print_plan_table() {
    echo ""
    echo "========================================"
    echo "[plan] iterative self-distillation workflow"
    echo "  NUM_ROUNDS       = ${NUM_ROUNDS}"
    echo "  START_ROUND      = ${START_ROUND}"
    echo "  LR_SCHED         = ${LR_SCHED}"
    echo "  FORCE_GEN        = ${FORCE_GEN}"
    echo "  FORCE_TRAIN      = ${FORCE_TRAIN}"
    echo "  FORCE_EVAL       = ${FORCE_EVAL}"
    echo "  SKIP_STAGES      = '${SKIP_STAGES}'"
    echo "  DRY_RUN          = ${DRY_RUN}"
    echo "  SKIP_GPU_STRESS  = ${SKIP_GPU_STRESS}"
    echo "  LOG_ROOT         = ${LOG_ROOT}"
    echo "----------------------------------------"
    local k
    for k in $(seq "${START_ROUND}" "${NUM_ROUNDS}"); do
        local mk_tag mk_path out_tag out_path ds_name ds_json
        mk_tag="$(compute_input_model_tag_for_round "${k}")"
        mk_path="$(compute_input_model_path_for_round "${k}")"
        out_tag="$(compute_round_model_tag "${k}")"
        out_path="$(compute_round_model_path "${k}")"
        ds_name="$(compute_dataset_name "${k}")"
        ds_json="$(compute_train_json_path "${k}")"
        echo "  [round ${k}]"
        echo "    M_k (input)         = ${mk_tag}"
        echo "    M_k path            = ${mk_path}"
        echo "    round output tag    = ${out_tag}"
        echo "    round output path   = ${out_path}"
        echo "    dataset name        = ${ds_name}"
        echo "    train json path     = ${ds_json}"
        echo "    file_list:"
        local f
        while IFS= read -r f; do
            echo "      * ${f}"
        done < <(compute_file_list_for_round "${k}")
    done
    echo "========================================"
}

print_summary_table() {
    echo ""
    echo "========================================"
    echo "[summary] workflow completed (status=${WORKFLOW_STATUS})"
    local k
    for k in $(seq "${START_ROUND}" "${NUM_ROUNDS}"); do
        local out_tag out_path
        out_tag="$(compute_round_model_tag "${k}")"
        out_path="$(compute_round_model_path "${k}")"
        echo "  [round ${k}]"
        echo "    merged model tag  = ${out_tag}"
        echo "    merged model path = ${out_path}"
        echo "    logs              = ${LOG_ROOT}/round${k}/"
    done
    echo "========================================"
}

# =============================================================================
# Per-round stage helpers
# =============================================================================

# ---- Stage: generate_wildjailbreak ----
stage_generate_wildjailbreak() {
    local k="$1"
    local mk_tag mk_path step2 step4
    mk_tag="$(compute_input_model_tag_for_round "${k}")"
    mk_path="$(compute_input_model_path_for_round "${k}")"
    step2="$(compute_wj_step2_path "${mk_tag}")"
    step4="$(compute_wj_step4_path "${mk_tag}")"

    if [[ "${FORCE_GEN}" != "1" && -s "${step2}" && -s "${step4}" ]]; then
        echo "[round ${k}] generate_wildjailbreak: artifacts already exist, skipping."
        echo "            ${step2}"
        echo "            ${step4}"
        return 0
    fi
    run_stage "${k}" "generate_wildjailbreak" \
        env MODEL_PATH="${mk_path}" MODEL_TAG="${mk_tag}" \
            BASE_MODEL_TAG="${BASE_MODEL_NAME}" \
            bash "${AUTO_PIPLINE_SH}"
}

# ---- Stage: generate_ultrafeedback ----
# Parameters:
#   $1 = round k  (used for log-dir grouping)
#   $2 = MODEL_TAG
#   $3 = MODEL_PATH
#   $4 = optional override stage_name (default: generate_ultrafeedback)
stage_generate_ultrafeedback() {
    local k="$1" mk_tag="$2" mk_path="$3"
    local stage_name="${4:-generate_ultrafeedback}"
    local uf_path; uf_path="$(compute_uf_wildguard_path "${mk_tag}")"

    if [[ "${FORCE_GEN}" != "1" && -s "${uf_path}" ]]; then
        echo "[round ${k}] ${stage_name}: artifact already exists, skipping."
        echo "            ${uf_path}"
        return 0
    fi
    run_stage "${k}" "${stage_name}" \
        env MODEL_PATH="${mk_path}" MODEL_TAG="${mk_tag}" \
            bash "${AUTO_ULTRAFEEDBACK_SH}"
}

# ---- Stage: make_train_data ----
stage_make_train_data() {
    local k="$1"
    local out_json; out_json="$(compute_train_json_path "${k}")"
    # Gather file_list
    local -a file_list=()
    local line
    while IFS= read -r line; do
        file_list+=("${line}")
    done < <(compute_file_list_for_round "${k}")

    run_stage "${k}" "make_train_data" \
        "${QWEN3_ENV_PY}" "${AUTO_MAKE_TRAIN_DATA_PY}" \
            --file_list "${file_list[@]}" \
            --output_file "${out_json}"
}

# ---- Stage: train ----
stage_train() {
    local k="$1"
    local out_tag out_path ds_name
    out_tag="$(compute_round_model_tag "${k}")"
    out_path="$(compute_round_model_path "${k}")"
    ds_name="$(compute_dataset_name "${k}")"

    # Idempotent skip: if merged dir already has weights and not forced
    if [[ "${FORCE_TRAIN}" != "1" && -d "${out_path}" && -f "${out_path}/config.json" ]]; then
        if ls "${out_path}"/*.safetensors >/dev/null 2>&1 \
           || ls "${out_path}"/pytorch_model*.bin >/dev/null 2>&1; then
            echo "[round ${k}] train: merged model already present, skipping: ${out_path}"
            return 0
        fi
    fi

    # Re-verify dataset_info.json key exists (safety net)
    if [[ "${DRY_RUN}" != "1" ]]; then
        local check
        check=$(python3 - "${DATASET_INFO_JSON}" "${ds_name}" <<'PYEOF'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    info = json.load(f)
print("OK" if sys.argv[2] in info else "MISSING")
PYEOF
        )
        if [[ "${check}" != "OK" ]]; then
            WORKFLOW_STATUS="failed"
            WORKFLOW_FAIL_ROUND="${k}"
            WORKFLOW_FAIL_STAGE="train"
            WORKFLOW_FAIL_LOG="${LOG_ROOT}/round${k}/train.log"
            echo "[FATAL] dataset_info.json missing key '${ds_name}', cannot train round ${k}."
            exit 11
        fi
    fi

    run_stage "${k}" "train" \
        env DATASET_NAME="${ds_name}" \
            BASE_MODEL_NAME="${BASE_MODEL_NAME}" \
            MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}" \
            LR_SCHEDULER_TYPE="${LR_SCHED}" \
            bash "${AUTO_SFT_PEFT_TRAIN_SH}"
}

# ---- Stage: eval ----
stage_eval() {
    local k="$1"
    local out_tag out_path
    out_tag="$(compute_round_model_tag "${k}")"
    out_path="$(compute_round_model_path "${k}")"

    # Idempotent skip heuristic: check if the advbench moderation result json
    # for this model tag already exists (one representative of a complete eval
    # batch). User can set FORCE_EVAL=1 to override.
    local sentinel="${REPO_ROOT}/evaluate/results/advbench/${out_tag}_tag_none_MD-Judge.json"
    if [[ "${FORCE_EVAL}" != "1" && -s "${sentinel}" ]]; then
        echo "[round ${k}] eval: sentinel artifact exists, skipping: ${sentinel}"
        return 0
    fi

    run_stage "${k}" "eval" \
        env MODEL_LIST_OVERRIDE="${out_tag}|${out_path}" \
            bash "${AUTO_EVAL_POOL_SH}"
}

# =============================================================================
# Main loop
# =============================================================================
main() {
    preflight
    print_plan_table

    # Edge-case guard: if round 1 will be run and base model uf_wildguard is
    # missing, run a pre-round ultrafeedback on the base model first (so that
    # round 1's file_list is complete).
    if [[ "${START_ROUND}" -le 1 ]]; then
        local base_uf; base_uf="$(compute_uf_wildguard_path "${BASE_MODEL_NAME}")"
        if [[ ! -s "${base_uf}" && "${FORCE_GEN}" != "1" ]]; then
            echo ""
            echo "[pre-round-1] base model ultrafeedback artifact missing, running..."
            echo "              target: ${base_uf}"
            stage_generate_ultrafeedback 1 "${BASE_MODEL_NAME}" "${BASE_MODEL_PATH}" \
                                         "pre_round_generate_ultrafeedback_base"
        fi
    fi

    local k
    for k in $(seq "${START_ROUND}" "${NUM_ROUNDS}"); do
        echo ""
        echo "========================================"
        echo "===== Round ${k} / ${NUM_ROUNDS} ====="
        echo "========================================"

        local mk_tag mk_path
        mk_tag="$(compute_input_model_tag_for_round "${k}")"
        mk_path="$(compute_input_model_path_for_round "${k}")"

        # Verify M_k exists on disk (especially for k>=2 depending on prior round's output)
        if [[ ! -e "${mk_path}" && "${DRY_RUN}" != "1" ]]; then
            WORKFLOW_STATUS="failed"
            WORKFLOW_FAIL_ROUND="${k}"
            WORKFLOW_FAIL_STAGE="generate_wildjailbreak"
            WORKFLOW_FAIL_LOG="${LOG_ROOT}/round${k}/generate_wildjailbreak.log"
            echo "[FATAL] round ${k}: input model path not found: ${mk_path}"
            exit 12
        fi

        # Stage 1: wildjailbreak data generation (auto_run_pipline.sh)
        stage_generate_wildjailbreak "${k}"

        # Stage 2: ultrafeedback data generation on M_k (auto_run_ultrafeedback.sh)
        stage_generate_ultrafeedback "${k}" "${mk_tag}" "${mk_path}"

        # Stage 3: merge training data (auto_make_train_data.py)
        stage_make_train_data "${k}"

        # Stage 4: train (auto_sft_peft_train.sh)
        stage_train "${k}"

        # Stage 5: eval (auto_run_evaluation_pool.sh)
        stage_eval "${k}"

        echo "[round ${k}] all stages done."
    done

    WORKFLOW_STATUS="succeeded"
    print_summary_table
    # Explicit exit 0 so EXIT trap sees rc=0 and runs gpu_stress.
    exit 0
}

main "$@"
