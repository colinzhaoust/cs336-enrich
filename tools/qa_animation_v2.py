#!/usr/bin/env python3
"""Non-mutating QA for CS336 v2 animation candidates.

The validator reads a candidate MP4 and its sidecars. It never rewrites the
video. A contact sheet is created only when --contact-sheet is supplied.

OCR checks are advisory because equations and stylized text are not reliably
recognized. Browser-context review remains mandatory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)

    def check(self, condition: bool, success: str, failure: str) -> None:
        (self.passes if condition else self.failures).append(
            success if condition else failure
        )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def fps_value(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def top_level_atoms(path: Path) -> list[tuple[str, int]]:
    atoms: list[tuple[str, int]] = []
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + 8 <= file_size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) != 8:
                break
            size, raw_kind = struct.unpack(">I4s", header)
            kind = raw_kind.decode("ascii", errors="replace")
            header_size = 8
            if size == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    break
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                break
            atoms.append((kind, offset))
            offset += size
    return atoms


def sample_times(duration: float, count: int) -> list[float]:
    if count <= 1:
        return [duration / 2]
    edge = min(0.5, duration * 0.02)
    usable = max(0.0, duration - 2 * edge)
    return [edge + usable * index / (count - 1) for index in range(count)]


def extract_frames(video: Path, times: list[float], directory: Path) -> list[Path]:
    frames: list[Path] = []
    for index, timestamp in enumerate(times):
        output = directory / f"frame-{index:02d}.png"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-y",
                str(output),
            ]
        )
        frames.append(output)
    return frames


def make_contact_sheet(frames: list[Path], times: list[float], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Pillow is required for --contact-sheet") from exc

    columns = 3
    thumb_width = 640
    first = Image.open(frames[0]).convert("RGB")
    ratio = thumb_width / first.width
    thumb_height = round(first.height * ratio)
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    font = ImageFont.load_default()

    for index, (frame_path, timestamp) in enumerate(zip(frames, times)):
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        label = f"{timestamp:05.1f}s"
        draw.rectangle((x + 8, y + 8, x + 62, y + 25), fill=(20, 20, 20))
        draw.text((x + 12, y + 10), label, fill="white", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def ocr_warnings(
    frames: list[Path],
    times: list[float],
    actual_width: int,
    actual_height: int,
    expected_width: int,
    expected_height: int,
    safe_x: int,
    safe_y: int,
    minimum_source_height: int,
) -> list[str]:
    if not shutil.which("tesseract"):
        return ["tesseract not found; skipped advisory OCR checks"]

    scaled_safe_x = safe_x * actual_width / expected_width
    scaled_safe_y = safe_y * actual_height / expected_height
    scaled_min_height = minimum_source_height * actual_height / expected_height
    small: list[str] = []
    unsafe: list[str] = []

    for frame, timestamp in zip(frames, times):
        result = run(["tesseract", str(frame), "stdout", "--psm", "11", "tsv"])
        rows = result.stdout.splitlines()[1:]
        for row in rows:
            fields = row.split("\t")
            if len(fields) != 12 or fields[0] != "5":
                continue
            try:
                left, top, width, height = map(int, fields[6:10])
                confidence = float(fields[10])
            except ValueError:
                continue
            word = fields[11].strip()
            if confidence < 50 or len(word) < 2:
                continue
            if height < scaled_min_height:
                small.append(f"{timestamp:.1f}s {word!r} bbox height={height}px")
            if (
                left < scaled_safe_x
                or top < scaled_safe_y
                or left + width > actual_width - scaled_safe_x
                or top + height > actual_height - scaled_safe_y
            ):
                unsafe.append(f"{timestamp:.1f}s {word!r} bbox=({left},{top},{width},{height})")

    messages: list[str] = []
    if small:
        messages.append(
            "OCR found potentially small text (first 8): " + "; ".join(small[:8])
        )
    if unsafe:
        messages.append(
            "OCR found text in the safe margin (first 8): " + "; ".join(unsafe[:8])
        )
    if not messages:
        messages.append(
            "OCR found no recognized small/unsafe text; formulas still require manual review"
        )
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-duration", type=float, default=25.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--poster", type=Path)
    parser.add_argument(
        "--allow-missing-sidecars",
        action="store_true",
        help="Metadata audit only; production validation requires all sidecars.",
    )
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--safe-x", type=int, default=72)
    parser.add_argument("--safe-y", type=int, default=54)
    parser.add_argument(
        "--minimum-source-text-height",
        type=int,
        default=86,
        help="Advisory OCR bbox height on a 1080p master.",
    )
    parser.add_argument("--contact-sheet", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()

    if not args.video.is_file():
        print(f"FAIL: video does not exist: {args.video}", file=sys.stderr)
        return 2
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        print("FAIL: ffprobe and ffmpeg are required", file=sys.stderr)
        return 2

    metadata = probe(args.video)
    video_streams = [item for item in metadata["streams"] if item["codec_type"] == "video"]
    audio_streams = [item for item in metadata["streams"] if item["codec_type"] == "audio"]
    if len(video_streams) != 1:
        report.failures.append(f"expected one video stream, found {len(video_streams)}")
        video = video_streams[0] if video_streams else {}
    else:
        video = video_streams[0]

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    fps = fps_value(video.get("avg_frame_rate") or video.get("r_frame_rate", "0/0"))
    duration = float(metadata["format"].get("duration", 0.0))

    report.check(
        (width, height) == (args.width, args.height),
        f"resolution {width}x{height}",
        f"resolution {width}x{height}; expected {args.width}x{args.height}",
    )
    report.check(
        math.isclose(fps, args.fps, rel_tol=0.0, abs_tol=0.01),
        f"frame rate {fps:.3f} fps",
        f"frame rate {fps:.3f}; expected {args.fps:.3f} fps",
    )
    report.check(
        args.min_duration <= duration <= args.max_duration,
        f"duration {duration:.3f}s",
        f"duration {duration:.3f}s; expected {args.min_duration:g}-{args.max_duration:g}s",
    )
    report.check(
        video.get("codec_name") == "h264",
        "video codec h264",
        f"video codec {video.get('codec_name')}; expected h264",
    )
    report.check(
        video.get("pix_fmt") == "yuv420p",
        "pixel format yuv420p",
        f"pixel format {video.get('pix_fmt')}; expected yuv420p",
    )

    atoms = top_level_atoms(args.video)
    atom_names = [name for name, _ in atoms]
    faststart = "moov" in atom_names and "mdat" in atom_names and atom_names.index("moov") < atom_names.index("mdat")
    report.check(
        faststart,
        "MP4 fast-start atom order (moov before mdat)",
        f"MP4 is not fast-start; top-level atoms: {atom_names}",
    )

    if audio_streams:
        audio = audio_streams[0]
        report.check(
            audio.get("codec_name") == "aac",
            "audio codec AAC",
            f"audio codec {audio.get('codec_name')}; expected AAC",
        )
        report.check(
            int(audio.get("sample_rate", 0)) == 48000,
            "audio sample rate 48 kHz",
            f"audio sample rate {audio.get('sample_rate')}; expected 48000 Hz",
        )

    sidecars = {
        "transcript": args.transcript,
        "captions/descriptions": args.captions,
        "poster": args.poster,
    }
    for label, sidecar in sidecars.items():
        exists = sidecar is not None and sidecar.is_file()
        if args.allow_missing_sidecars and not exists:
            report.warnings.append(f"missing {label} sidecar")
        else:
            report.check(exists, f"{label} sidecar present", f"missing {label} sidecar")

    if args.poster and args.poster.is_file():
        try:
            from PIL import Image

            with Image.open(args.poster) as poster:
                poster_size = poster.size
            report.check(
                poster_size == (args.width, args.height),
                f"poster resolution {poster_size[0]}x{poster_size[1]}",
                f"poster resolution {poster_size[0]}x{poster_size[1]}; expected {args.width}x{args.height}",
            )
        except ImportError:
            report.warnings.append("Pillow not found; skipped poster dimension check")

    if args.ocr or args.contact_sheet:
        times = sample_times(duration, args.samples)
        with tempfile.TemporaryDirectory(prefix="cs336-animation-qa-") as temporary:
            frames = extract_frames(args.video, times, Path(temporary))
            if args.ocr:
                report.warnings.extend(
                    ocr_warnings(
                        frames,
                        times,
                        width,
                        height,
                        args.width,
                        args.height,
                        args.safe_x,
                        args.safe_y,
                        args.minimum_source_text_height,
                    )
                )
            if args.contact_sheet:
                make_contact_sheet(frames, times, args.contact_sheet)
                report.passes.append(f"wrote contact sheet {args.contact_sheet}")

    print(f"CS336 animation QA: {args.video}")
    for item in report.passes:
        print(f"PASS: {item}")
    for item in report.warnings:
        print(f"WARN: {item}")
    for item in report.failures:
        print(f"FAIL: {item}")
    print(
        f"SUMMARY: {len(report.passes)} pass, {len(report.warnings)} warning, "
        f"{len(report.failures)} fail"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
