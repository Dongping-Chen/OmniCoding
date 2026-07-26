#!/usr/bin/env bash

# Shared cold-start preparation for Code-X-SFT-9B jobs.
#
# This file must be sourced so the staged paths and cache environment are
# inherited by Ray, actor workers, and SGLang workers.

_code_x_checkpoint_is_complete() {
  local checkpoint_root="$1"
  local tracker="${checkpoint_root}/latest_checkpointed_iteration.txt"
  local iteration

  [[ -f "${tracker}" ]] || return 1
  iteration="$(<"${tracker}")"
  if [[ "${iteration}" == "release" ]]; then
    [[ -d "${checkpoint_root}/release" ]]
    return
  fi
  [[ "${iteration}" =~ ^[0-9]+$ ]] || return 1
  printf -v iteration '%07d' "$((10#${iteration}))"
  [[ -d "${checkpoint_root}/iter_${iteration}" ]]
}

_code_x_record_cold_start_stage() {
  local stage="$1"
  local elapsed_s="$2"
  local detail="$3"
  local timing_file="${COLD_START_TIMING_FILE:-}"

  printf '[cold-start] stage=%s elapsed_s=%s detail=%s\n' \
    "${stage}" "${elapsed_s}" "${detail}" >&2
  if [[ -n "${timing_file}" ]]; then
    mkdir -p "$(dirname -- "${timing_file}")"
    if [[ ! -s "${timing_file}" ]]; then
      printf 'stage\telapsed_s\tdetail\n' >"${timing_file}"
    fi
    printf '%s\t%s\t%s\n' "${stage}" "${elapsed_s}" "${detail}" >>"${timing_file}"
  fi
}

prepare_code_x_sft_9b_runtime() {
  local repo_root="$1"
  local source_venv="${OMNICODING_ORIGINAL_VENV_ROOT:-${VENV_ROOT:-${repo_root}/.venv}}"
  local source_relax="${OMNICODING_ORIGINAL_RELAX_ROOT:-${RELAX_ROOT:-${source_venv}/src/Relax}}"
  local source_megatron="${OMNICODING_ORIGINAL_MEGATRON_ROOT:-${MEGATRON_ROOT:-${source_venv}/src/Megatron-LM}}"
  local source_model="${OMNICODING_ORIGINAL_MODEL_PATH:-${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}}"
  local stage_to_nvme="${STAGE_TO_LOCAL_NVME:-1}"
  local preconvert_dcp="${PRECONVERT_ACTOR_DCP:-1}"
  local require_dcp="${REQUIRE_ACTOR_DCP:-0}"
  local local_root
  local cache_root
  local started_at
  local elapsed_s

  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Code-X-SFT-9B runtime preparation requires a Slurm allocation" >&2
    return 2
  fi
  if [[ "${stage_to_nvme}" != 0 && "${stage_to_nvme}" != 1 ]]; then
    echo "STAGE_TO_LOCAL_NVME must be 0 or 1" >&2
    return 2
  fi
  if [[ "${preconvert_dcp}" != 0 && "${preconvert_dcp}" != 1 ]]; then
    echo "PRECONVERT_ACTOR_DCP must be 0 or 1" >&2
    return 2
  fi
  if [[ "${require_dcp}" != 0 && "${require_dcp}" != 1 ]]; then
    echo "REQUIRE_ACTOR_DCP must be 0 or 1" >&2
    return 2
  fi

  export OMNICODING_ORIGINAL_VENV_ROOT="${source_venv}"
  export OMNICODING_ORIGINAL_RELAX_ROOT="${source_relax}"
  export OMNICODING_ORIGINAL_MEGATRON_ROOT="${source_megatron}"
  export OMNICODING_ORIGINAL_MODEL_PATH="${source_model}"
  export VENV_ROOT="${source_venv}"

  local_root="${LOCAL_RUNTIME_ROOT:-/tmp/omnicoding-${SLURM_JOB_ID}}"
  cache_root="${OMNICODING_CACHE_ROOT:-${repo_root}/.cache/rl-runtime}"
  export LOCAL_RUNTIME_ROOT="${local_root}"
  export OMNICODING_CACHE_ROOT="${cache_root}"
  export PYTHONPYCACHEPREFIX="${local_root}/pycache"
  export TRITON_CACHE_DIR="${cache_root}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
  export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
  export CUDA_CACHE_PATH="${cache_root}/cuda"
  export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
  mkdir -p \
    "${local_root}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${CUDA_CACHE_PATH}"

  if [[ "${stage_to_nvme}" == 1 ]]; then
    local staged_source_root="${local_root}/python"
    local staged_site="${staged_source_root}/site-packages"
    local staged_relax="${staged_source_root}/Relax"
    local staged_megatron="${staged_source_root}/Megatron-LM"
    local staged_omnicoding="${staged_source_root}/omnicoding-src"
    local source_site="${source_venv}/lib/python3.12/site-packages"
    local source_marker="${staged_source_root}/.complete"

    if [[ ! -f "${source_marker}" ]]; then
      started_at="$(date +%s)"
      mkdir -p "${staged_site}/megatron"
      rsync -a "${source_relax}/" "${staged_relax}/"
      rsync -a "${source_megatron}/" "${staged_megatron}/"
      rsync -a "${repo_root}/src/" "${staged_omnicoding}/"
      for package in transformers sglang ray; do
        if [[ -d "${source_site}/${package}" ]]; then
          rsync -a "${source_site}/${package}/" "${staged_site}/${package}/"
        fi
      done
      rsync -a \
        "${source_site}/megatron/bridge/" \
        "${staged_site}/megatron/bridge/"
      "${source_venv}/bin/python" -m compileall -q \
        "${staged_relax}/relax" \
        "${staged_megatron}/megatron" \
        "${staged_omnicoding}/omnicoding" \
        "${staged_site}/transformers" \
        "${staged_site}/sglang" \
        "${staged_site}/ray" \
        "${staged_site}/megatron/bridge"
      touch "${source_marker}"
      elapsed_s="$(( $(date +%s) - started_at ))"
      _code_x_record_cold_start_stage \
        "stage_python" "${elapsed_s}" "${staged_source_root}"
    fi

    export RELAX_ROOT="${staged_relax}"
    export MEGATRON_ROOT="${staged_megatron}"
    export PYTHONPATH="${staged_relax}:${staged_megatron}:${staged_omnicoding}:${staged_site}:${PYTHONPATH:-}"

    local staged_model="${local_root}/models/Code-X-SFT-9B"
    local model_marker="${staged_model}/.local-stage-complete"
    if [[ ! -f "${model_marker}" ]]; then
      started_at="$(date +%s)"
      mkdir -p "${staged_model}"
      rsync -a "${source_model}/" "${staged_model}/"
      touch "${model_marker}"
      elapsed_s="$(( $(date +%s) - started_at ))"
      _code_x_record_cold_start_stage \
        "stage_hf_model" "${elapsed_s}" "${staged_model}"
    fi
    export MODEL_PATH="${staged_model}"
  else
    export RELAX_ROOT="${source_relax}"
    export MEGATRON_ROOT="${source_megatron}"
    export MODEL_PATH="${source_model}"
    export PYTHONPATH="${source_relax}:${source_megatron}:${repo_root}/src:${PYTHONPATH:-}"
  fi

  if [[ -z "${ACTOR_LOAD:-}" && "${preconvert_dcp}" == 1 ]]; then
    local shared_dcp="${ACTOR_DCP_CACHE:-${repo_root}/models/Code-X-SFT-9B-mcore-tp${ACTOR_TP:-8}}"
    if ! _code_x_checkpoint_is_complete "${shared_dcp}"; then
      started_at="$(date +%s)"
      # shellcheck disable=SC1090
      source "${RELAX_ROOT}/scripts/models/qwen35-9B.sh"
      if "${source_venv}/bin/python" -m torch.distributed.run \
          --nproc-per-node "${NUM_GPUS:-8}" \
          "${RELAX_ROOT}/scripts/tools/convert_hf_to_torch_dist.py" \
          "${MODEL_ARGS[@]}" \
          --hf-checkpoint "${MODEL_PATH}" \
          --save "${shared_dcp}" \
          --megatron-to-hf-mode bridge \
          --tensor-model-parallel-size "${ACTOR_TP:-8}" \
          --pipeline-model-parallel-size 1 \
          --context-parallel-size "${ACTOR_CP:-1}" \
          --expert-model-parallel-size 1 \
          --expert-tensor-parallel-size 1 \
          --sequence-parallel \
          --no-rope-fusion; then
        elapsed_s="$(( $(date +%s) - started_at ))"
        _code_x_record_cold_start_stage \
          "convert_hf_to_dcp" "${elapsed_s}" "${shared_dcp}"
      else
        elapsed_s="$(( $(date +%s) - started_at ))"
        _code_x_record_cold_start_stage \
          "convert_hf_to_dcp_failed" "${elapsed_s}" "${shared_dcp}"
        if [[ "${require_dcp}" == 1 ]]; then
          return 1
        fi
        echo "DCP conversion failed; continuing with the original HF checkpoint" >&2
      fi
    fi
    if _code_x_checkpoint_is_complete "${shared_dcp}"; then
      export ACTOR_LOAD="${shared_dcp}"
    fi
  fi

  if [[ -n "${ACTOR_LOAD:-}" ]]; then
    if ! _code_x_checkpoint_is_complete "${ACTOR_LOAD}"; then
      echo "ACTOR_LOAD is not a complete Megatron checkpoint: ${ACTOR_LOAD}" >&2
      return 2
    fi
    if [[ "${stage_to_nvme}" == 1 && "${ACTOR_LOAD}" != "${local_root}/"* ]]; then
      local staged_dcp="${local_root}/checkpoints/$(basename -- "${ACTOR_LOAD}")"
      local dcp_marker="${staged_dcp}/.local-stage-complete"
      if [[ ! -f "${dcp_marker}" ]]; then
        started_at="$(date +%s)"
        mkdir -p "${staged_dcp}"
        rsync -a "${ACTOR_LOAD}/" "${staged_dcp}/"
        touch "${dcp_marker}"
        elapsed_s="$(( $(date +%s) - started_at ))"
        _code_x_record_cold_start_stage \
          "stage_actor_dcp" "${elapsed_s}" "${staged_dcp}"
      fi
      export ACTOR_LOAD="${staged_dcp}"
    fi
  fi
}
