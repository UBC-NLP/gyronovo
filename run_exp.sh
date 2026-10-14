#!/usr/bin/env bash
set -u

: "${DATA:?Set DATA=/path/to/dataset before running run_exp.sh}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU scheduling knobs. You can edit the defaults here or override them from the shell.
# GPUS accepts physical CUDA IDs as either "0 1 2" or "0,1,2"; leave it empty to auto-detect.
# NUM_GPUS controls how many GPUs each experiment uses. Set it >1 for Lightning DDP.
# EXPERIMENTS_PER_GPU_GROUP allows intentionally sharing one GPU group between concurrent experiments.
# TRAIN_BATCH_SIZE overrides the YAML global train batch size.
# TRAIN_BATCH_SIZE_PER_GPU sets per-GPU train batch size and is converted to a global size per run.
# PREDICT_BATCH_SIZE overrides the YAML prediction batch size used by single-GPU denovo/export.
# N_WORKERS overrides YAML DataLoader workers; multi-GPU runs default to 0 to avoid
# multiplying large in-memory datasets across DDP ranks and worker processes.
GPUS="${GPUS:-}"
NUM_GPUS="${NUM_GPUS:-1}"
EXPERIMENTS_PER_GPU_GROUP="${EXPERIMENTS_PER_GPU_GROUP:-${EXPERIMENTS_PER_GPU:-1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
TRAIN_BATCH_SIZE_PER_GPU="${TRAIN_BATCH_SIZE_PER_GPU:-}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-}"
N_WORKERS="${N_WORKERS:-}"

DATASET_NAME="${DATASET_NAME:-$(basename "${DATA%/}")}"
if [[ -z "$DATASET_NAME" || "$DATASET_NAME" == "." || "$DATASET_NAME" == "/" ]]; then
  echo "Could not infer dataset name from DATA=${DATA}. Set DATASET_NAME=... explicitly." >&2
  exit 1
fi
DATASET_NAME="$(printf '%s' "$DATASET_NAME" | tr -c '[:alnum:]_.-' '_')"
echo "Dataset: ${DATASET_NAME} (${DATA})"

export NOVOBENCH_TRANSFORM_WORKERS="${NOVOBENCH_TRANSFORM_WORKERS:-1}"
echo "Preprocessing transform workers per experiment: ${NOVOBENCH_TRANSFORM_WORKERS}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_gpu_ids() {
  local raw="${1//,/ }"
  local gpu_id
  local parsed_gpu_ids=()

  raw="${raw//$'\n'/ }"
  raw="${raw//$'\t'/ }"
  read -r -a parsed_gpu_ids <<< "$raw"

  for gpu_id in "${parsed_gpu_ids[@]}"; do
    gpu_id=$(trim "$gpu_id")
    [[ -n "$gpu_id" ]] && printf '%s\n' "$gpu_id"
  done
}

detect_compute_gpus() {
  local min_mem="${MIN_GPU_MEMORY_MB:-10000}"
  local idx
  local name
  local mem

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  while IFS=',' read -r idx name mem; do
    idx=$(trim "$idx")
    name=$(trim "$name")
    mem=$(trim "$mem")

    [[ -z "$idx" || -z "$name" || -z "$mem" ]] && continue
    case "$name" in
      *Display*|*display*|*DISPLAY*)
        continue
        ;;
    esac
    [[ "$mem" =~ ^[0-9]+$ ]] || continue
    ((mem >= min_mem)) || continue

    printf '%s\n' "$idx"
  done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
}

GPU_IDS=()
if [[ -n "${GPUS:-}" ]]; then
  while IFS= read -r gpu_id; do
    [[ -n "$gpu_id" ]] && GPU_IDS+=("$gpu_id")
  done < <(parse_gpu_ids "$GPUS")
  echo "Configured GPU pool: ${GPU_IDS[*]:-(none)}"
else
  while IFS= read -r gpu_id; do
    [[ -n "$gpu_id" ]] && GPU_IDS+=("$gpu_id")
  done < <(detect_compute_gpus)
  echo "Auto-detected GPU pool: ${GPU_IDS[*]:-(none)}"
fi

if ((${#GPU_IDS[@]} == 0)); then
  echo "No compute GPUs found. Set GPUS=\"0 2\" to override, or lower MIN_GPU_MEMORY_MB=${MIN_GPU_MEMORY_MB:-10000}." >&2
  exit 1
fi

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer." >&2
  exit 1
fi

if ((${#GPU_IDS[@]} < NUM_GPUS)); then
  echo "NUM_GPUS=${NUM_GPUS} requested, but only ${#GPU_IDS[@]} GPU(s) are available in the selected pool: ${GPU_IDS[*]}" >&2
  exit 1
fi

if [[ ! "$EXPERIMENTS_PER_GPU_GROUP" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPERIMENTS_PER_GPU_GROUP must be a positive integer." >&2
  exit 1
fi

if [[ -n "$TRAIN_BATCH_SIZE" && -n "$TRAIN_BATCH_SIZE_PER_GPU" ]]; then
  echo "Set only one of TRAIN_BATCH_SIZE or TRAIN_BATCH_SIZE_PER_GPU." >&2
  exit 1
fi

for batch_override in TRAIN_BATCH_SIZE TRAIN_BATCH_SIZE_PER_GPU PREDICT_BATCH_SIZE; do
  batch_value="${!batch_override}"
  if [[ -n "$batch_value" && ! "$batch_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${batch_override} must be a positive integer when set." >&2
    exit 1
  fi
done

if [[ -n "$N_WORKERS" && ! "$N_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "N_WORKERS must be a non-negative integer when set." >&2
  exit 1
fi

GPU_GROUPS=()
GPU_GROUP_COUNTS=()
for ((group_start = 0; group_start < ${#GPU_IDS[@]}; group_start += NUM_GPUS)); do
  group_ids=("${GPU_IDS[@]:group_start:NUM_GPUS}")
  if ((${#group_ids[@]} < NUM_GPUS)); then
    echo "Leaving unused GPU(s) without a full NUM_GPUS=${NUM_GPUS} group: ${group_ids[*]}" >&2
    break
  fi
  group_csv="$(IFS=,; printf '%s' "${group_ids[*]}")"
  GPU_GROUPS+=("$group_csv")
  GPU_GROUP_COUNTS+=("${#group_ids[@]}")
done

WORKER_GPU_GROUPS=()
WORKER_GPU_COUNTS=()
for group_idx in "${!GPU_GROUPS[@]}"; do
  for ((worker_idx = 0; worker_idx < EXPERIMENTS_PER_GPU_GROUP; worker_idx++)); do
    WORKER_GPU_GROUPS+=("${GPU_GROUPS[$group_idx]}")
    WORKER_GPU_COUNTS+=("${GPU_GROUP_COUNTS[$group_idx]}")
  done
done

echo "Scheduling experiments across GPU(s): ${GPU_IDS[*]}"
echo "Each experiment sees ${NUM_GPUS} GPU(s); group(s): ${GPU_GROUPS[*]}"
echo "Allowing up to ${EXPERIMENTS_PER_GPU_GROUP} concurrent experiment(s) per GPU group (${#WORKER_GPU_GROUPS[@]} total slot(s))."

mkdir -p "outputs/eval/${DATASET_NAME}" "outputs/run_exp_logs/${DATASET_NAME}"

EXPERIMENTS=(
  "GyroNovo|lipnovo|configs/GyroNovo.yaml|outputs/GyroNovo/checkpoints"
  "lipnovo_base|lipnovo|configs/lipnovo_base.yaml|outputs/lipnovo/base/checkpoints"
)

select_checkpoint() {
  local ckpt_dir="$1"
  local best_path_file="${ckpt_dir}/best_model_path.txt"
  local ckpt=""

  if [[ -s "$best_path_file" ]]; then
    IFS= read -r ckpt < "$best_path_file" || true
    if [[ -n "$ckpt" && -f "$ckpt" ]]; then
      printf '%s\n' "$ckpt"
      return 0
    fi
    echo "Ignoring stale best checkpoint path in ${best_path_file}: ${ckpt}" >&2
  fi

  ls -t "$ckpt_dir"/*.ckpt 2>/dev/null | head -n 1 || true
}

scope_output_path() {
  local path="${1#./}"
  if [[ "$path" == outputs/* ]]; then
    printf 'outputs/%s/%s' "$DATASET_NAME" "${path#outputs/}"
  else
    printf '%s' "$path"
  fi
}

make_dataset_config() {
  local config="$1"
  local name="$2"
  local model="$3"
  local ckpt_dir="$4"
  local devices="$5"
  local suffix="$6"
  local train_batch_size="$7"
  local predict_batch_size="$8"
  local n_workers="$9"
  local config_dir="${RUN_STATE}/configs/${DATASET_NAME}"
  local generated_config="${config_dir}/${name}_${suffix}.yaml"

  mkdir -p "$config_dir"
  python - "$config" "$generated_config" "$DATASET_NAME" "$name" "$model" "$ckpt_dir" "$devices" "$train_batch_size" "$predict_batch_size" "$n_workers" <<'PY'
import os
import sys
import yaml

(
    config_path,
    output_path,
    dataset_name,
    experiment_name,
    model_name,
    checkpoint_dir,
    devices,
    train_batch_size,
    predict_batch_size,
    n_workers,
) = sys.argv[1:]

with open(config_path) as handle:
    config = yaml.safe_load(handle)


def scoped_output_path(value):
    if value in {None, ""}:
        return value
    text = str(value)
    if text.startswith("./outputs/"):
        return f"./outputs/{dataset_name}/{text[len('./outputs/'):]}"
    if text.startswith("outputs/"):
        return f"outputs/{dataset_name}/{text[len('outputs/'):]}"
    return text


def ensure_parent(path):
    if path in {None, ""}:
        return
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


config["dataset_name"] = dataset_name
config["model_save_folder_path"] = checkpoint_dir
config["devices"] = int(devices)
if train_batch_size:
    config["train_batch_size"] = int(train_batch_size)
if predict_batch_size:
    config["predict_batch_size"] = int(predict_batch_size)
if n_workers:
    config["n_workers"] = int(n_workers)
config["logger_save_path"] = scoped_output_path(config.get("logger_save_path"))

base_wandb_name = config.get("wandb_name") or experiment_name
config["wandb_name"] = f"{base_wandb_name}-{dataset_name}"

for section_name in ("lipnovo",):
    section = config.get(section_name)
    if not isinstance(section, dict):
        continue
    analysis = section.get("analysis")
    if isinstance(analysis, dict):
        for key in ("best_validation_predictions_output",):
            if key in analysis:
                analysis[key] = scoped_output_path(analysis[key])

ensure_parent(config.get("model_save_folder_path"))
ensure_parent(config.get("logger_save_path"))
for section_name in ("lipnovo",):
    section = config.get(section_name)
    if not isinstance(section, dict):
        continue
    analysis = section.get("analysis")
    if not isinstance(analysis, dict):
        continue
    for key in ("best_validation_predictions_output",):
        ensure_parent(analysis.get(key))

with open(output_path, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(output_path)
PY
}

run_experiment() {
  local name="$1"
  local model="$2"
  local config="$3"
  local ckpt_dir="$4"
  local gpu_group="$5"
  local gpu_count="$6"
  local script
  local out="outputs/eval/${DATASET_NAME}/${name}.csv"
  local ckpt
  local dataset_ckpt_dir
  local train_config
  local eval_config
  local eval_gpu
  local effective_train_batch_size="$TRAIN_BATCH_SIZE"
  local effective_predict_batch_size="$PREDICT_BATCH_SIZE"
  local effective_n_workers="$N_WORKERS"

  if [[ "$model" != "lipnovo" ]]; then
    echo "Unknown model type for ${name}: ${model}" >&2
    return 1
  fi
  script="run.py"

  if [[ -n "$TRAIN_BATCH_SIZE_PER_GPU" ]]; then
    effective_train_batch_size=$((TRAIN_BATCH_SIZE_PER_GPU * gpu_count))
  elif [[ -n "$TRAIN_BATCH_SIZE" && $((TRAIN_BATCH_SIZE % gpu_count)) -ne 0 ]]; then
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} is not divisible by NUM_GPUS=${gpu_count} for ${name}. Use TRAIN_BATCH_SIZE_PER_GPU or choose a divisible global batch size." >&2
    return 1
  fi

  if [[ -z "$effective_n_workers" && "$gpu_count" -gt 1 ]]; then
    effective_n_workers=0
  fi

  dataset_ckpt_dir="$(scope_output_path "$ckpt_dir")"
  train_config="$(make_dataset_config "$config" "$name" "$model" "$dataset_ckpt_dir" "$gpu_count" "train" "$effective_train_batch_size" "$effective_predict_batch_size" "$effective_n_workers")"
  eval_config="$(make_dataset_config "$config" "$name" "$model" "$dataset_ckpt_dir" "1" "denovo" "$effective_train_batch_size" "$effective_predict_batch_size" "$N_WORKERS")"
  eval_gpu="${gpu_group%%,*}"

  mkdir -p "$dataset_ckpt_dir" "$(dirname "$out")"

  export CUDA_VISIBLE_DEVICES="$gpu_group"
  export WANDB_MODE="${WANDB_MODE:-online}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: training on physical GPU(s) ${gpu_group} (${gpu_count} Lightning device(s))"
  if [[ -n "$effective_train_batch_size" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: train_batch_size=${effective_train_batch_size} global ($((effective_train_batch_size / gpu_count)) per GPU)"
  fi
  if [[ -n "$effective_n_workers" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: n_workers=${effective_n_workers} per rank"
  fi
  if ((gpu_count > 1)); then
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "torchrun is required for NUM_GPUS=${gpu_count} training but was not found in PATH." >&2
      return 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: launching torchrun with ${gpu_count} local process(es)"
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node "$gpu_count" \
      "$script" \
      --mode train \
      --data_path "$DATA" \
      --config_path "$train_config" || return 1
  else
    python "$script" \
      --mode train \
      --data_path "$DATA" \
      --config_path "$train_config" || return 1
  fi

  ckpt=$(select_checkpoint "$dataset_ckpt_dir")
  if [[ -z "$ckpt" ]]; then
    echo "${name}: no checkpoint found in ${dataset_ckpt_dir}" >&2
    return 1
  fi

  rm -f "$out"
  export CUDA_VISIBLE_DEVICES="$eval_gpu"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: denovo from ${ckpt} on physical GPU ${eval_gpu}"
  python "$script" \
    --mode denovo \
    --data_path "$DATA" \
    --ckpt_path "$ckpt" \
    --denovo_output_path "$out" \
    --config_path "$eval_config" || return 1

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name}: evaluating ${out}"
  python novobench/metrics/evaluate.py "$out" || return 1
}

PIDS=()
NAMES=()
EXIT_STATUS=0
FREE_SLOT=0
RUN_STATE=$(mktemp -d "${TMPDIR:-/tmp}/run_exp.XXXXXX")

cleanup() {
  local pid
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$RUN_STATE"
}

trap cleanup EXIT
trap 'EXIT_STATUS=130; exit 130' INT TERM

finish_slot() {
  local slot="$1"
  local pid="${PIDS[$slot]:-}"
  local name="${NAMES[$slot]:-unknown}"

  if [[ -n "$pid" ]]; then
    if ! wait "$pid"; then
      echo "${name}: failed, see outputs/run_exp_logs/${DATASET_NAME}/${name}.log" >&2
      EXIT_STATUS=1
    fi
  fi

  PIDS[$slot]=""
  NAMES[$slot]=""
  rm -f "$RUN_STATE/slot_${slot}.done"
}

wait_for_free_slot() {
  local slot

  while true; do
    for slot in "${!WORKER_GPU_GROUPS[@]}"; do
      if [[ -z "${PIDS[$slot]:-}" ]]; then
        FREE_SLOT="$slot"
        return
      fi

      if [[ -f "$RUN_STATE/slot_${slot}.done" ]]; then
        finish_slot "$slot"
        FREE_SLOT="$slot"
        return
      fi
    done
    sleep 15
  done
}

launch_slot() {
  local slot="$1"
  local name="$2"
  local model="$3"
  local config="$4"
  local ckpt_dir="$5"
  local gpu_group="${WORKER_GPU_GROUPS[$slot]}"
  local gpu_count="${WORKER_GPU_COUNTS[$slot]}"
  local done_file="$RUN_STATE/slot_${slot}.done"
  local log_file="outputs/run_exp_logs/${DATASET_NAME}/${name}.log"

  rm -f "$done_file"
  mkdir -p "$(dirname "$log_file")"
  (
    trap 'status=$?; printf "%s\n" "$status" > "$done_file"; exit "$status"' EXIT
    run_experiment "$name" "$model" "$config" "$ckpt_dir" "$gpu_group" "$gpu_count"
  ) > "$log_file" 2>&1 &

  PIDS[$slot]="$!"
  NAMES[$slot]="$name"
  echo "Started ${name} on GPU group ${gpu_group}; log: ${log_file}"
}

for idx in "${!EXPERIMENTS[@]}"; do
  IFS='|' read -r name model config ckpt_dir <<< "${EXPERIMENTS[$idx]}"
  wait_for_free_slot
  launch_slot "$FREE_SLOT" "$name" "$model" "$config" "$ckpt_dir"
done

while true; do
  active=0
  for slot in "${!WORKER_GPU_GROUPS[@]}"; do
    if [[ -n "${PIDS[$slot]:-}" ]]; then
      if [[ -f "$RUN_STATE/slot_${slot}.done" ]]; then
        finish_slot "$slot"
      else
        active=1
      fi
    fi
  done

  if ((active == 0)); then
    break
  fi

  sleep 15
done

exit "$EXIT_STATUS"
