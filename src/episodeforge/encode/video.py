"""Video and image encoding engine with multi-codec and hardware acceleration support."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import av
import av.video.stream
import numpy as np
from PIL import Image

CODEC_MAP = {
    "av1": "libsvtav1",
    "libsvtav1": "libsvtav1",
    "h264": "libx264" if "libx264" in av.codec.codecs_available else "h264_videotoolbox",
    "hevc": "hevc_videotoolbox" if "hevc_videotoolbox" in av.codec.codecs_available else "libx265",
    "h264_videotoolbox": "h264_videotoolbox",
    "hevc_videotoolbox": "hevc_videotoolbox",
    "libx264": "libx264",
    "libx265": "libx265",
}


def resize_image_array(
    arr: np.ndarray,
    target_res: tuple[int, int],
    mode: str = "center_crop",
) -> np.ndarray:
    """Resize an image array (H, W, 3) to target_res (width, height) with specified mode."""
    target_w, target_h = target_res
    orig_h, orig_w = arr.shape[:2]

    if (orig_w, orig_h) == (target_w, target_h):
        return arr

    pil_img = Image.fromarray(arr)

    if mode == "stretch":
        resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    elif mode == "center_crop":
        # Scale to cover target dimensions
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        scaled_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop to target_w, target_h
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        cropped = scaled_img.crop((left, top, right, bottom))
        return np.asarray(cropped, dtype=np.uint8)

    elif mode == "letterbox":
        # Scale to fit inside target dimensions
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        scaled_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Paste onto black background
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(scaled_img, (offset_x, offset_y))
        return np.asarray(canvas, dtype=np.uint8)

    else:
        resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)


def encode_video_from_frames(
    frame_paths: Sequence[str | Path],
    output_path: Path | str,
    fps: float,
    codec: str = "av1",
    crf: int = 28,
    bitrate_kbps: int | None = None,
    target_res: tuple[int, int] | None = None,
    resize_mode: str = "center_crop",
    pix_fmt: str = "yuv420p",
) -> Path:
    """Encode an ordered list of image frame files into a compressed MP4 video."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not frame_paths:
        raise ValueError("Cannot encode empty list of frames into video")

    # Read first frame to determine dimensions
    with Image.open(frame_paths[0]) as first_img:
        first_arr = np.asarray(first_img.convert("RGB"), dtype=np.uint8)

    if target_res is not None:
        first_arr = resize_image_array(first_arr, target_res, mode=resize_mode)

    height, width = first_arr.shape[:2]
    # Dimensions must be even for yuv420p
    width = width - (width % 2)
    height = height - (height % 2)

    av_codec = CODEC_MAP.get(codec.lower(), codec)

    # Container setup
    container = av.open(str(out_p), mode="w", format="mp4")
    raw_stream = container.add_stream(av_codec, rate=int(round(fps)))
    stream = cast(av.video.stream.VideoStream, raw_stream)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt

    # Bitrate / CRF options
    options: dict[str, Any] = {}
    if bitrate_kbps is not None:
        stream.bit_rate = int(bitrate_kbps * 1000)
    elif "videotoolbox" not in av_codec:
        options["crf"] = str(crf)

    if av_codec == "libsvtav1":
        options["preset"] = "8"  # Balanced fast encode for AV1

    stream.options = options

    for fp in frame_paths:
        with Image.open(fp) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

        if target_res is not None or (arr.shape[1], arr.shape[0]) != (width, height):
            arr = resize_image_array(arr, (width, height), mode=resize_mode)

        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return out_p


def encode_jpeg_bytes(img_arr: np.ndarray, quality: int = 90) -> bytes:
    """Encode an RGB uint8 array to JPEG bytes buffer."""
    buf = io.BytesIO()
    img = Image.fromarray(img_arr)
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
