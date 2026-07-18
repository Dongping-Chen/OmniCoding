"""Unit tests for ``omnicoding.rl.tokenize.tokenize_trajectory_multimodal``.

Set ``OMNICODING_TEST_PROCESSOR`` to the local processor snapshot used by the
RL model to enable these integration tests.

These tests pin the loss-mask boundary algorithm against real Qwen3-VL chat
template behavior. The mistakes they catch all involve silent train/serve drift
— wrong loss positions or missing pixel_values would corrupt RL training without
crashing.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

import pytest


QWEN35_DIR = Path(os.environ.get("OMNICODING_TEST_PROCESSOR", ""))


@pytest.fixture(scope="module")
def processor():
    if not os.environ.get("OMNICODING_TEST_PROCESSOR") or not QWEN35_DIR.is_dir():
        pytest.skip("set OMNICODING_TEST_PROCESSOR to run multimodal integration tests")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(QWEN35_DIR), trust_remote_code=True)


@pytest.fixture(scope="module")
def tokenizer(processor):
    return processor.tokenizer


def _png_b64(size: tuple[int, int] = (56, 56), color: tuple[int, int, int] = (128, 128, 128)) -> str:
    """Build a tiny RGB PNG and return its base64 string (no data: prefix)."""
    from PIL import Image
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _data_url(size: tuple[int, int] = (56, 56)) -> str:
    return f"data:image/png;base64,{_png_b64(size)}"


def _kira_image_message(size: tuple[int, int] = (56, 56), tag: str = "x.jpg") -> dict:
    """Mimic kira's ``read_image_native`` output — the user message it injects
    after each ``image_read`` tool reply."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"image_read: '{tag}'"},
            {"type": "image_url", "image_url": {"url": _data_url(size)}},
        ],
    }


def _import_fn():
    from omnicoding.rl.tokenize import tokenize_trajectory_multimodal  # noqa: PLC0415
    return tokenize_trajectory_multimodal


# ─── basic shape + return values ─────────────────────────────────────────────


def test_empty(processor, tokenizer):
    fn = _import_fn()
    tokens, mask, resp_len, mm_in, mm_train = fn([], tokenizer, processor)
    assert tokens == [] and mask == [] and resp_len == 0
    assert mm_in is None and mm_train is None


def test_text_only_no_images_no_multimodal_inputs(processor, tokenizer):
    """Text-only trajectory should NOT populate multimodal_inputs/train_inputs."""
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokens, mask, resp_len, mm_in, mm_train = fn(msgs, tokenizer, processor)
    assert resp_len == len(mask) > 0
    assert mm_in is None
    assert mm_train is None


# ─── kira native image_read injection ────────────────────────────────────────


def test_one_image_read_populates_multimodal(processor, tokenizer):
    """Realistic kira-native trajectory: assistant calls image_read → tool ack →
    user message with the actual image data URL → final assistant answer.

    Verify:
      - input_ids contains <|image_pad|> tokens (processor expanded them).
      - multimodal_train_inputs has pixel_values + image_grid_thw.
      - All <|image_pad|> positions in response are loss=0 (they're observations).
      - Final assistant content + <|im_end|> get loss=1.
    """
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Look at the image."},
        {
            "role": "assistant",
            "content": "<think>read it</think>",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "image_read", "arguments": '{"file_path":"x.jpg"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "Loaded x.jpg (image/png, 124 bytes)."},
        _kira_image_message(),
        {"role": "assistant", "content": "<answer>blue</answer>"},
    ]
    tokens, mask, resp_len, mm_in, mm_train = fn(msgs, tokenizer, processor)

    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert image_pad_id in tokens, "expected <|image_pad|> tokens in input_ids"
    n_pad = sum(1 for t in tokens if t == image_pad_id)
    assert n_pad >= 4, f"expected ≥4 image_pad tokens for a 56×56 image, got {n_pad}"

    # multimodal_train_inputs populated
    assert mm_train is not None
    assert "pixel_values" in mm_train
    assert "image_grid_thw" in mm_train
    assert "mm_token_type_ids" not in mm_train
    assert mm_in is not None and len(mm_in["images"]) == 1

    # Loss mask alignment
    assert len(mask) == resp_len > 0
    response_tokens = tokens[-resp_len:]
    # Every image_pad position (which lives in response side, in the user msg)
    # must be loss=0
    for i, t in enumerate(response_tokens):
        if t == image_pad_id:
            assert mask[i] == 0, f"image_pad at response pos {i} got loss=1 (expected 0)"

    # The Qwen chat template renders each message as
    # ``<|im_start|>{role}\n{content}<|im_end|>\n``. The trailing newline
    # AFTER the final <|im_end|> is template scaffolding (not model output)
    # so it stays loss=0. The <|im_end|> itself sits one token before that.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    if response_tokens[-1] == im_end_id:
        # No trailing scaffolding newline — im_end IS the last token.
        assert mask[-1] == 1, "trailing <|im_end|> after assistant must be loss=1"
    else:
        # Last token is template scaffolding (likely \n) → loss=0; im_end is
        # one position earlier → loss=1.
        assert response_tokens[-2] == im_end_id, (
            f"expected response to end with <|im_end|> + scaffold-tok, "
            f"got tail ids {response_tokens[-3:]}, decoded={tokenizer.decode(response_tokens[-3:])!r}"
        )
        assert mask[-2] == 1, "<|im_end|> closing assistant content must be loss=1"
        assert mask[-1] == 0, "trailing scaffolding token must be loss=0"
        # Sanity: the trailing token should be a newline (or BOS-like scaffold).
        assert response_tokens[-1] in nl_id or response_tokens[-1] == nl_id[0]


def test_two_image_reads_in_one_trajectory(processor, tokenizer):
    """Two sequential image_read calls — both images appear in pixel_values
    along the leading batch dim."""
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "read both images"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "type": "function",
                            "function": {"name": "image_read", "arguments": '{"file_path":"a.jpg"}'}}],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "Loaded a"},
        _kira_image_message(tag="a.jpg"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "image_read", "arguments": '{"file_path":"b.jpg"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Loaded b"},
        _kira_image_message(tag="b.jpg"),
        {"role": "assistant", "content": "<answer>done</answer>"},
    ]
    tokens, mask, resp_len, mm_in, mm_train = fn(msgs, tokenizer, processor)
    assert mm_train is not None
    # image_grid_thw is shape [num_images, 3]
    grid = mm_train["image_grid_thw"]
    assert grid.shape[0] == 2, f"expected 2 images in image_grid_thw, got shape {tuple(grid.shape)}"
    assert mm_in is not None and len(mm_in["images"]) == 2


# ─── loss mask correctness ───────────────────────────────────────────────────


def test_role_marker_is_zero(processor, tokenizer):
    """The 3-token <|im_start|>assistant\\n role marker must be loss=0 even
    though the assistant content + <|im_end|> are loss=1."""
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "hello world"},
    ]
    tokens, mask, resp_len, _, _ = fn(msgs, tokenizer, processor)
    # First three response tokens are the role marker.
    assert mask[0] == 0 and mask[1] == 0 and mask[2] == 0, (
        f"role-marker tokens should be loss=0, got {mask[:3]}"
    )
    # Content tokens through <|im_end|> are loss=1; the trailing template
    # scaffolding newline (last token) stays loss=0.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    response_tokens = tokens[-resp_len:]
    if response_tokens[-1] == im_end_id:
        # No trailing newline — all content (positions 3..end) is loss=1.
        assert all(m == 1 for m in mask[3:]), f"content tokens should be 1, got {mask[3:]}"
    else:
        # Last token is scaffolding (\n) → 0; everything between role marker
        # and that final scaffold is loss=1 (covers content + <|im_end|>).
        assert all(m == 1 for m in mask[3:-1]), f"content tokens should be 1, got {mask[3:-1]}"
        assert mask[-1] == 0, "trailing scaffolding newline should be loss=0"


def test_user_and_tool_segments_are_all_zero(processor, tokenizer):
    """Non-assistant segments between assistant turns must be entirely loss=0,
    regardless of content (including image_pad)."""
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "execute_commands", "arguments": '{"keystrokes":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ls output"},
        {"role": "assistant", "content": "<answer>done</answer>"},
    ]
    tokens, mask, resp_len, _, _ = fn(msgs, tokenizer, processor)
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    asst_role = tokenizer.encode("assistant", add_special_tokens=False)[0]
    response_tokens = tokens[-resp_len:]
    # Walk segments; for every non-assistant segment, every position must be 0.
    boundaries = [i for i, t in enumerate(response_tokens) if t == im_start]
    boundaries.append(len(response_tokens))
    for k in range(len(boundaries) - 1):
        seg_start = boundaries[k]
        seg_end = boundaries[k + 1]
        if seg_start + 1 < len(response_tokens) and response_tokens[seg_start + 1] != asst_role:
            for j in range(seg_start, seg_end):
                assert mask[j] == 0, f"non-assistant segment pos {j} got loss=1"


def test_response_length_aligns_with_first_assistant(processor, tokenizer):
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    tokens, mask, resp_len, _, _ = fn(msgs, tokenizer, processor)
    assert len(mask) == resp_len
    assert len(tokens) >= resp_len
    # The position resp_len from the END should be a <|im_start|> with
    # role=assistant immediately after.
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    asst_role = tokenizer.encode("assistant", add_special_tokens=False)[0]
    split_pos = len(tokens) - resp_len
    assert tokens[split_pos] == im_start
    assert tokens[split_pos + 1] == asst_role


def test_loss_mask_ignores_raw_im_start_inside_content(tokenizer):
    """A literal ``<|im_start|>`` emitted in assistant text is content unless it
    starts a full ``<|im_start|>{role}\n`` marker."""
    from omnicoding.rl.tokenize import _build_loss_mask_from_ids, _specials

    specials = _specials(tokenizer)
    assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    tail_newline = tokenizer.encode("\n", add_special_tokens=False)
    input_ids = (
        assistant_marker
        + tokenizer.encode("before ", add_special_tokens=False)
        + [specials["im_start"]]
        + tokenizer.encode(" still assistant content", add_special_tokens=False)
        + [specials["im_end"]]
        + tail_newline
    )

    mask = _build_loss_mask_from_ids(input_ids, specials)
    im_end_pos = input_ids.index(specials["im_end"])
    assert mask[:3] == [0, 0, 0]
    assert all(v == 1 for v in mask[3 : im_end_pos + 1])
    assert all(v == 0 for v in mask[im_end_pos + 1 :])


# ─── kira preprocessor (data URL → PIL.Image) ────────────────────────────────


def test_kira_preprocessor_normalizes_arguments():
    from omnicoding.rl.tokenize import _kira_to_relax_messages
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "image_read",
                                      "arguments": '{"file_path":"x.jpg"}'}}]},
    ]
    out, imgs = _kira_to_relax_messages(msgs)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict), f"expected dict, got {type(args).__name__}: {args!r}"
    assert args == {"file_path": "x.jpg"}
    assert imgs == []


def test_kira_preprocessor_decodes_data_url_to_pil():
    from PIL import Image as PILImage
    from omnicoding.rl.tokenize import _kira_to_relax_messages
    url = _data_url((32, 32))
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}
    ]
    out, imgs = _kira_to_relax_messages(msgs)
    assert len(imgs) == 1 and isinstance(imgs[0], PILImage.Image)
    assert imgs[0].size == (32, 32)
    # Content was rewritten so the image part has type=image with PIL inline
    parts = out[0]["content"]
    assert any(p.get("type") == "image" and p.get("image") is imgs[0] for p in parts)


def test_kira_preprocessor_drops_corrupt_data_url():
    from omnicoding.rl.tokenize import _kira_to_relax_messages
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
        ]}
    ]
    out, imgs = _kira_to_relax_messages(msgs)
    # No image — corrupt URL gets dropped silently
    assert imgs == []
    parts = out[0]["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "image" not in types and "image_url" not in types


# ─── Qwen fold (image into tool reply) ───────────────────────────────────────


def test_fold_moves_image_into_tool_reply():
    """Realistic kira-stored trajectory:
        assistant(image_read tool_call) → tool(ack) → user([text, image_url])
    Folded to Qwen-shape:
        assistant(image_read tool_call) → tool([ack+text, image_url])
    The image_url moves into the tool reply's content list; the standalone
    user turn disappears.
    """
    from omnicoding.rl.tokenize import fold_native_image_read_messages_for_qwen
    msgs = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "image_read", "arguments": '{"file_path":"x"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "Loaded x."},
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: 'x'"},
            {"type": "image_url", "image_url": {"url": _data_url((32, 32))}},
        ]},
    ]
    out = fold_native_image_read_messages_for_qwen(msgs)
    # Standalone user-with-image turn folded away.
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool"], f"unexpected roles {roles}"
    # Tool message now carries a multimodal content list with the image part.
    tool_msg = out[-1]
    assert isinstance(tool_msg["content"], list)
    types = [p.get("type") for p in tool_msg["content"]]
    assert "image_url" in types or "image" in types, (
        f"expected image part in folded tool content, got types={types}"
    )


def test_fold_preserves_unrelated_user_turns():
    """A user turn with NO image (regular text) is left in place; only the
    follow-up user-with-image after a tool reply gets folded."""
    from omnicoding.rl.tokenize import fold_native_image_read_messages_for_qwen
    msgs = [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "follow-up text only"},
    ]
    out = fold_native_image_read_messages_for_qwen(msgs)
    assert out == msgs


def test_fold_skips_when_assistant_didnt_call_image_read():
    """If the preceding tool reply matched an ``execute_commands`` call, the
    image-bearing user turn must NOT be folded into it (it'd be miscategorized
    as a shell-output multimodal reply)."""
    from omnicoding.rl.tokenize import fold_native_image_read_messages_for_qwen
    msgs = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "execute_commands", "arguments": '{"keystrokes":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ls output"},
        {"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": _data_url((32, 32))}},
        ]},
    ]
    out = fold_native_image_read_messages_for_qwen(msgs)
    assert len(out) == len(msgs), "no fold expected; user turn should stay"
    assert out[-1]["role"] == "user"


def _load_kira_fold_reference():
    """Load the Kira reference from the same installed monorepo package."""
    from omnicoding.agents.kira.loop import _fold_native_image_read_messages_for_qwen

    return _fold_native_image_read_messages_for_qwen


def test_fold_parity_with_kira_reference():
    """Behavioral parity: feed a battery of representative trajectories through
    BOTH our inline fold and kira's source-of-truth fold; outputs must match."""
    kira_fold = _load_kira_fold_reference()
    if kira_fold is None:
        pytest.skip("kira/loop.py source not available")

    from omnicoding.rl.tokenize import fold_native_image_read_messages_for_qwen as ours
    url = _data_url((32, 32))
    img_part = {"type": "image_url", "image_url": {"url": url}}
    txt_part = lambda s: {"type": "text", "text": s}

    cases = [
        # 1. canonical native image_read flow
        [
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c0", "type": "function",
                             "function": {"name": "image_read", "arguments": '{"file_path":"a"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "Loaded a"},
            {"role": "user", "content": [txt_part("image_read: 'a'"), img_part]},
            {"role": "assistant", "content": "<answer>done</answer>"},
        ],
        # 2. plain text — no fold expected
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        # 3. assistant called execute_commands (NOT image_read) — image-bearing
        #    user turn must NOT fold into the wrong tool reply.
        [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c0", "type": "function",
                             "function": {"name": "execute_commands", "arguments": '{"keystrokes":"ls"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "ls out"},
            {"role": "user", "content": [txt_part("look"), img_part]},
        ],
        # 4. two consecutive image_reads (two folds in series)
        [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c0", "type": "function",
                             "function": {"name": "image_read", "arguments": '{"file_path":"a"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "Loaded a"},
            {"role": "user", "content": [txt_part("image_read: 'a'"), img_part]},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "image_read", "arguments": '{"file_path":"b"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Loaded b"},
            {"role": "user", "content": [txt_part("image_read: 'b'"), img_part]},
            {"role": "assistant", "content": "<answer>done</answer>"},
        ],
        # 5. tool reply already has list content (rare but legal)
        [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c0", "type": "function",
                             "function": {"name": "image_read", "arguments": '{"file_path":"a"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": [txt_part("Loaded a")]},
            {"role": "user", "content": [txt_part("image_read: 'a'"), img_part]},
        ],
    ]
    for i, case in enumerate(cases):
        a = ours(case)
        b = kira_fold(case)
        assert a == b, f"case {i} diverged:\n ours={a!r}\n kira={b!r}"


# ─── multimodal tokenize uses folded layout ──────────────────────────────────


def test_tokenize_emits_image_pad_inside_tool_response(processor, tokenizer):
    """After our preprocessor folds + the Qwen template renders, the
    ``<|image_pad|>`` should fall INSIDE a ``role=tool`` segment (rendered as
    ``<|im_start|>user\\n<tool_response>...``), NOT in a separate
    ``<|im_start|>user\\n`` segment without ``<tool_response>``.
    """
    fn = _import_fn()
    msgs = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "image_read", "arguments": '{"file_path":"x.jpg"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "Loaded x.jpg."},
        _kira_image_message(),
        {"role": "assistant", "content": "<answer>done</answer>"},
    ]
    tokens, mask, resp_len, mm_in, mm_train = fn(msgs, tokenizer, processor)

    # Find image_pad position
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_idx = next((i for i, t in enumerate(tokens) if t == image_pad_id), -1)
    assert pad_idx >= 0, "expected at least one <|image_pad|> token"

    # The text immediately preceding the first image_pad should contain
    # "<tool_response>" (proves the image lives inside a tool reply, not
    # in a standalone user turn). Decode a window before pad_idx.
    window_start = max(0, pad_idx - 50)
    text_before = tokenizer.decode(tokens[window_start:pad_idx], skip_special_tokens=False)
    assert "<tool_response>" in text_before, (
        f"image_pad not preceded by <tool_response> in window: {text_before!r}\n"
        "Means fold didn't apply — image is in standalone user turn (drift from SFT)."
    )

    # And the segment containing image_pad must be loss=0 (it's an observation).
    if pad_idx >= len(tokens) - resp_len:
        # pad lives in response side
        rel = pad_idx - (len(tokens) - resp_len)
        assert mask[rel] == 0, "image_pad inside tool segment must be loss=0"
