"""Build the user-message instruction passed to ``KiraAgent.run(...)`` from
one dataset record.

Kira already injects its own SYSTEM_PROMPT (terminal + image_read + task_complete
contract) so this module only adds task-specific content: media manifest,
question, options, and answer-format reminder.
"""

from __future__ import annotations

from .dataset import Record

# Same wrapper the dataset's grader expects.
ANSWER_TAG_REMINDER = (
    "Wrap your final answer in <answer>...</answer> tags. The grader pulls the "
    "LAST <answer>...</answer> in the trajectory; everything else is ignored."
)

# Media-handling guidance. Kira can only image_read PNG/JPG/etc., so videos and
# audios must be processed via shell tools first.
MEDIA_GUIDANCE = (
    "MEDIA HANDLING\n"
    "  - .mp4 / .mov / .webm / .mkv: extract keyframes with ffmpeg "
    "(`ffmpeg -i FILE -vf fps=1/2 frames/%03d.jpg`), then call image_read on the frames.\n"
    "  - .wav / .mp3 / .flac: render a spectrogram with sox or python "
    "(matplotlib + librosa) into a .png, OR transcribe with whisper "
    "(`whisper FILE --model small --output_format txt`), then "
    "image_read or `cat` the result.\n"
    "  - .jpg / .png / .gif / .webp: image_read directly.\n"
    "  - DO NOT pass .mp4 or .wav files to image_read — it only accepts images."
)


CONTINUE_PROMPT = (
    "Your last turn had no tool call. Every assistant turn must call exactly "
    "one tool. If you have the answer, emit `echo '<answer>X</answer>'` via "
    "execute_commands (or put the wrapper in your assistant content) and then "
    "call task_complete. If you are still working, call execute_commands or "
    "image_read to make progress."
)


def build_instruction(record: Record, staged_media: list[str]) -> str:
    parts: list[str] = []
    parts.append(MEDIA_GUIDANCE)
    parts.append("")

    if staged_media:
        parts.append("MEDIA FILES IN YOUR WORKSPACE")
        for rel in staged_media:
            parts.append(f"  - {rel}")
        parts.append("")
    else:
        parts.append("(No media files for this task — answer from the question text alone.)")
        parts.append("")

    parts.append("QUESTION")
    parts.append(record.question.strip())
    parts.append("")

    if record.options:
        parts.append("OPTIONS")
        for opt in record.options:
            parts.append(f"  {opt}")
        parts.append("")
        parts.append(
            "Answer with ONLY the option letter (e.g., A) inside <answer>...</answer>."
        )
        parts.append("")

    parts.append(ANSWER_TAG_REMINDER)
    return "\n".join(parts)
