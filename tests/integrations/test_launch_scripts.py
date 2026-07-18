from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _flag_value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def test_sft_defaults_match_released_27b_recipe(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "swift-arguments.txt"
    fake_swift = bin_dir / "swift"
    fake_swift.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_swift.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "MODEL": "Qwen/Qwen3.6-27B",
        "DATASET": str(tmp_path / "sft.jsonl"),
        "OUTPUT_DIR": str(tmp_path / "output"),
    }

    subprocess.run(
        ["bash", "integrations/ms_swift/train_sft.sh"],
        check=True,
        env=env,
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:1] == ["sft"]
    assert _flag_value(arguments, "--model") == "Qwen/Qwen3.6-27B"
    assert _flag_value(arguments, "--max_length") == "32000"
    assert _flag_value(arguments, "--num_train_epochs") == "2"
    assert _flag_value(arguments, "--lora_rank") == "64"
    assert _flag_value(arguments, "--lora_alpha") == "128"
    assert _flag_value(arguments, "--target_modules") == "all-linear"


def test_sglang_served_name_is_explicit(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "python-arguments.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "MODEL_PATH": "shuaishuaicdp/Code-X-SFT-27B",
        "SERVED_MODEL_NAME": "shuaishuaicdp/Code-X-SFT-27B",
    }

    subprocess.run(
        ["bash", "integrations/sglang/serve.sh"],
        check=True,
        env=env,
    )

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "sglang.launch_server", "--model-path"]
    assert _flag_value(arguments, "--model-path") == (
        "shuaishuaicdp/Code-X-SFT-27B"
    )
    assert _flag_value(arguments, "--served-model-name") == (
        "shuaishuaicdp/Code-X-SFT-27B"
    )


def test_slurm_output_directives_do_not_require_preexisting_directory() -> None:
    scripts = [
        Path("infra/slurm/harness_item.sbatch"),
        Path("infra/slurm/serve_sglang.sbatch"),
        Path("infra/slurm/train_sft.sbatch"),
        Path("infra/slurm/train_rl.sbatch"),
    ]
    for script in scripts:
        output_lines = [
            line for line in script.read_text(encoding="utf-8").splitlines()
            if line.startswith("#SBATCH --output=")
        ]
        assert len(output_lines) == 1
        assert "/" not in output_lines[0].split("=", 1)[1]


def test_relax_release_has_27b_recipe_and_safe_config_template() -> None:
    recipe = Path("recipes/rl_27b_gspo.sh").read_text(encoding="utf-8")
    assert "shuaishuaicdp/Code-X-SFT-27B" in recipe
    assert "--advantage-estimator gspo" in recipe
    assert "--custom-generate-function-path omnicoding.rl.rollout.generate" in recipe
    assert "--custom-rm-path omnicoding.rl.reward.reward_func_group" in recipe
    assert '--save "$SAVE_DIR"' in recipe
    assert '--save-interval "$SAVE_INTERVAL"' in recipe
    assert "Qwen3.5-9B" not in recipe
    assert '"ROLLOUT_COORDINATOR_TOKEN_FILE",' in recipe
    assert '"ROLLOUT_COORDINATOR_TOKEN",' not in recipe

    template = Path(
        "integrations/relax/coordinator.env.example"
    ).read_text(encoding="utf-8")
    for name in (
        "RL_TRAIN_JSONL",
        "DATASET_ROOT",
        "ROLLOUT_SBATCH_SCRIPT",
        "ROLLOUT_ALLOWED_SGLANG_ORIGINS",
        "ROLLOUT_ALLOWED_MODELS",
        "ROLLOUT_COORDINATOR_PUBLIC_URL",
        "ROLLOUT_COORDINATOR_TOKEN_FILE",
        "ROLLOUT_SGLANG_MODEL",
        "SAVE_DIR",
    ):
        assert name in template
    assert not any(
        line.startswith("export ROLLOUT_COORDINATOR_TOKEN=")
        for line in template.splitlines()
    )


def test_relax_recipe_keeps_token_value_out_of_ray_arguments(tmp_path: Path) -> None:
    relax_root = tmp_path / "Relax"
    model_config = relax_root / "scripts" / "models" / "qwen36-27B.sh"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("MODEL_ARGS=()\n", encoding="utf-8")
    megatron_root = tmp_path / "Megatron-LM"
    megatron_root.mkdir()
    prompt_parquet = tmp_path / "prompts.parquet"
    prompt_parquet.touch()
    token_file = tmp_path / "coordinator-token"
    token_value = "secret-that-must-not-enter-argv"
    token_file.write_text(f"{token_value}\n", encoding="utf-8")
    token_file.chmod(0o600)
    save_dir = tmp_path / "checkpoints"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "ray-arguments.txt"
    calls = tmp_path / "ray-calls.txt"
    fake_ray = bin_dir / "ray"
    fake_ray.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" >> \"$RAY_CALLS\"\n"
        "if [[ \"${1:-}\" == status ]]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_ray.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "RAY_CALLS": str(calls),
        "RELAX_ROOT": str(relax_root),
        "MEGATRON_ROOT": str(megatron_root),
        "PROMPT_PARQUET": str(prompt_parquet),
        "SAVE_DIR": str(save_dir),
        "ROLLOUT_COORDINATOR_PUBLIC_URL": "http://127.0.0.1:8910",
        "ROLLOUT_COORDINATOR_TOKEN_FILE": str(token_file),
    }
    env.pop("ROLLOUT_COORDINATOR_TOKEN", None)

    subprocess.run(["bash", "recipes/rl_27b_gspo.sh"], check=True, env=env)

    arguments_text = capture.read_text(encoding="utf-8")
    arguments = arguments_text.splitlines()
    assert token_value not in arguments_text
    assert str(token_file) in arguments_text
    assert _flag_value(arguments, "--save") == str(save_dir)
    assert _flag_value(arguments, "--save-interval") == "50"
    calls_text = calls.read_text(encoding="utf-8")
    assert "status\n--address=auto\n" in calls_text
    assert "job\nsubmit\n--address\nhttp://127.0.0.1:8265\n" in calls_text
