from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def test_merge_uses_multimodal_model_and_saves_processor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return "base-model"

    class FakeMerged:
        def save_pretrained(self, output, **kwargs):
            calls["save_model"] = (output, kwargs)

    class FakeAdapter:
        def merge_and_unload(self):
            calls["merge"] = True
            return FakeMerged()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base, adapter):
            calls["adapter"] = (base, adapter)
            return FakeAdapter()

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["processor"] = (model_id, kwargs)
            return cls()

        def save_pretrained(self, output):
            calls["save_processor"] = output

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_peft = ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.Qwen3_5ForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    script = Path("integrations/ms_swift/merge_lora.py")
    spec = importlib.util.spec_from_file_location("omnicoding_test_merge_lora", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--base",
            "Qwen/Qwen3.6-27B",
            "--adapter",
            str(adapter),
            "--output",
            str(output),
        ],
    )

    assert module.main() == 0
    assert calls["model"] == (
        "Qwen/Qwen3.6-27B",
        {
            "torch_dtype": "bf16",
            "device_map": "auto",
            "trust_remote_code": True,
        },
    )
    assert calls["adapter"] == ("base-model", adapter)
    assert calls["merge"] is True
    assert calls["save_model"] == (
        output,
        {"safe_serialization": True, "max_shard_size": "5GB"},
    )
    assert calls["processor"] == (
        "Qwen/Qwen3.6-27B",
        {"trust_remote_code": True},
    )
    assert calls["save_processor"] == output
