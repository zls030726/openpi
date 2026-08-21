"""Convert LIBERO-Plus RLDS directly into perturbation-specific LeRobot datasets.

This follows OpenPI's ``convert_libero_rlds_to_lerobot.py`` frame-writing
contract, but classifies every RLDS episode from
``episode_metadata/file_path`` *before* calling ``add_frame``.  No mixed
LeRobot dataset is used as an intermediate.

The five released perturbation groups are converted independently.  An empty
``object`` dataset skeleton is emitted deliberately because the public mixed
RLDS archive has no object/objects-spanning source group.  Finally an ``all``
view is built from the split datasets: Parquet episode tables are reindexed and
videos are hard-linked (copied only if hard links are unavailable).  Thus the
RLDS is parsed once and the large video payload is encoded once.
"""

from __future__ import annotations

import av
import io
import os
import csv
import copy
import json
import math
import time
import types
import shutil
import argparse
import multiprocessing

from PIL import Image
from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor


from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.compute_stats import auto_downsample_height_width
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.libero_plus_rlds_reader_utils import classify_source_path, iter_episodes, load_shard_lengths


PERTURBATIONS = ("background", "camera", "light", "noise", "language", "object")

SOURCE_TO_PERTURBATION = {
    "env": "background",
    "camera_view": "camera",
    "light": "light",
    "noise": "noise",
    "language": "language",
}

PERTURBATION_TO_INDEX = {name: index for index, name in enumerate(PERTURBATIONS)}

SUITE_TO_INDEX = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_goal": 2,
    "libero_10": 3,
}

DATASET_NAMES = {
    **{name: f"libero_plus_{name}_lerobot" for name in PERTURBATIONS},
    "all": "libero_plus_all_lerobot",
}

REQUIRED_INDEX_COLUMNS = {
    "episode_index",
    "shard_index",
    "record_index",
    "source_group",
    "suite",
    "episode_length",
    "language_instruction",
    "source_path",
}

VIDEO_ENCODER_CONFIG = {"codec": "h264", "crf": 23, "gop": 10, "preset": "ultrafast"}


@dataclass(frozen=True)
class ManifestRow:
    output_episode_index: int
    source_episode_id: int
    source_shard_index: int
    source_record_index: int
    suite: str
    suite_index: int
    task: str
    perturbation: str
    perturbation_index: int
    source_group: str
    variant_index: int
    episode_length: int
    source_path: str


@dataclass(frozen=True)
class DatasetView:
    root: Path
    meta: LeRobotDatasetMetadata


def feature_spec() -> dict[str, dict[str, Any]]:
    """OpenPI LIBERO features plus integer provenance features.

    ``video`` is used instead of the minimal example's embedded ``image``
    storage so the result matches the existing LIBERO-Plus LeRobot releases
    and remains compact.  OpenPI consumes either representation identically.
    """
    return {
        "image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"state_{index}" for index in range(8)],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"action_{index}" for index in range(7)],
        },
        "perturbation_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "suite_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "source_episode_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "variant_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }


def mapping_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perturbation_to_index": PERTURBATION_TO_INDEX,
        "source_group_to_perturbation": SOURCE_TO_PERTURBATION,
        "suite_to_index": SUITE_TO_INDEX,
        "variant_index_definition": (
            "Zero-based encounter order within each "
            "(perturbation, suite, canonical task) group. The release does not "
            "store the publisher's sampled environment parameter ID."
        ),
        "source_episode_id_definition": (
            "Zero-based TFDS logical order: ascending shard_index, then record_index."
        ),
        "object_release_status": (
            "No object/objects-spanning source_group exists in the public mixed RLDS release; "
            "the object dataset is intentionally empty rather than inferred from duplicates."
        ),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_index_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"index manifest has no header: {path}")
        missing = REQUIRED_INDEX_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"index manifest is missing columns {sorted(missing)}")
        rows = list(reader)
    for expected_index, row in enumerate(rows):
        if int(row["episode_index"]) != expected_index:
            raise ValueError(
                f"index manifest row {expected_index} has episode_index={row['episode_index']}"
            )
        expected_perturbation = SOURCE_TO_PERTURBATION.get(row["source_group"])
        if expected_perturbation is None:
            raise ValueError(
                f"index manifest episode {expected_index} has unknown source_group={row['source_group']!r}"
            )
    return rows


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise ValueError(
                f"output root already exists: {output_root}. Use --overwrite only for a disposable rerun."
            )
        resolved = output_root.resolve()
        if resolved == Path(resolved.anchor) or len(resolved.parts) < 5:
            raise ValueError(f"refusing to recursively remove broad path: {resolved}")
        shutil.rmtree(resolved)
    output_root.mkdir(parents=True)


def create_dataset(
    output_root: Path,
    key: str,
    *,
    fps: int,
    image_writer_threads: int,
    image_writer_processes: int,
    direct_jpeg: bool,
) -> LeRobotDataset:
    name = DATASET_NAMES[key]
    dataset = LeRobotDataset.create(
        repo_id=name,
        root=output_root / name,
        robot_type="panda",
        fps=fps,
        features=feature_spec(),
        use_videos=True,
        image_writer_threads=0 if direct_jpeg else image_writer_threads,
        image_writer_processes=0 if direct_jpeg else image_writer_processes,
        video_backend="pyav",
    )
    if direct_jpeg:
        # add_frame still records its conventional temporary paths, but no
        # temporary image payload is written. encode_episode_videos consumes
        # the pending source JPEG byte strings directly, avoiding 4.4 million
        # small-file writes and reads on network storage.
        dataset._save_image = lambda image, path: None

        original_save_episode = dataset.save_episode

        def save_episode_with_memory_stats(self, episode_data=None) -> None:
            original_compute_stats = lerobot_dataset_module.compute_episode_stats

            def compute_stats_without_temp_files(data, features):
                non_image_data = {
                    key: value
                    for key, value in data.items()
                    if features[key]["dtype"] not in ("image", "video")
                }
                stats = original_compute_stats(non_image_data, features)
                sequences = getattr(self, "_pending_jpeg_sequences", None)
                if sequences is None:
                    raise RuntimeError("direct JPEG statistics have no pending episode payload")
                for video_key in self.meta.video_keys:
                    jpeg_sequence = sequences[video_key]
                    indices = sample_indices(len(jpeg_sequence))
                    samples = []
                    for index in indices:
                        with Image.open(io.BytesIO(jpeg_sequence[index])) as image:
                            array = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
                        samples.append(auto_downsample_height_width(array))
                    sample_array = np.stack(samples)
                    image_stats = get_feature_stats(
                        sample_array,
                        axis=(0, 2, 3),
                        keepdims=True,
                    )
                    stats[video_key] = {
                        stat_name: (
                            stat_value
                            if stat_name == "count"
                            else np.squeeze(stat_value / 255.0, axis=0)
                        )
                        for stat_name, stat_value in image_stats.items()
                    }
                return stats

            lerobot_dataset_module.compute_episode_stats = compute_stats_without_temp_files
            try:
                original_save_episode(episode_data)
            finally:
                lerobot_dataset_module.compute_episode_stats = original_compute_stats

        dataset.save_episode = types.MethodType(save_episode_with_memory_stats, dataset)

        def encode_pending_jpegs(self, episode_index: int) -> dict[str, str]:
            sequences = getattr(self, "_pending_jpeg_sequences", None)
            if sequences is None:
                raise RuntimeError("direct JPEG encoder has no pending episode payload")
            paths = {}
            for video_key in self.meta.video_keys:
                video_path = self.root / self.meta.get_video_file_path(episode_index, video_key)
                encode_jpeg_bytes_to_video(
                    sequences[video_key],
                    video_path,
                    self.fps,
                    **VIDEO_ENCODER_CONFIG,
                )
                paths[video_key] = str(video_path)
            return paths

        dataset.encode_episode_videos = types.MethodType(encode_pending_jpegs, dataset)
    return dataset


def create_empty_object_dataset(output_root: Path, *, fps: int) -> None:
    object_root = output_root / DATASET_NAMES["object"]
    object_meta = LeRobotDatasetMetadata.create(
        repo_id=DATASET_NAMES["object"],
        root=object_root,
        robot_type="panda",
        fps=fps,
        features=feature_spec(),
        use_videos=True,
    )
    del object_meta
    ensure_empty_jsonl_metadata(object_root)


def ensure_empty_jsonl_metadata(dataset_root: Path) -> None:
    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "episode_manifest.jsonl"):
        path = dataset_root / "meta" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def write_dataset_sidecars(dataset_root: Path, *, key: str, count: int) -> None:
    write_json(dataset_root / "meta" / "perturbation_mapping.json", mapping_document())
    if key == "object":
        status = (
            "This is an intentionally empty LeRobot v2.1 dataset skeleton. The public "
            "libero_plus_rlds archive has zero object/objects-spanning-labelled episodes."
        )
    elif key == "all":
        status = "Union of all five released perturbation groups, ordered by source_episode_id."
    else:
        status = f"Perturbation-specific dataset: {key}."
    (dataset_root / "README.md").write_text(
        "# LIBERO-Plus LeRobot conversion\n\n"
        f"{status}\n\nEpisodes: **{count:,}**.\n\n"
        "Additional per-frame integer fields are `perturbation_index`, `suite_index`, "
        "`source_episode_id`, and `variant_index`. String provenance is in "
        "`meta/episode_manifest.jsonl`; integer mappings are in "
        "`meta/perturbation_mapping.json`.\n",
        encoding="utf-8",
    )


def check_episode_against_index(episode, row: dict[str, str]) -> tuple[str, str, str]:
    source_group, suite = classify_source_path(episode.file_path)
    checks = {
        "episode_index": (episode.episode_index, int(row["episode_index"])),
        "shard_index": (episode.shard_index, int(row["shard_index"])),
        "record_index": (episode.record_index, int(row["record_index"])),
        "source_group": (source_group, row["source_group"]),
        "suite": (suite, row["suite"]),
        "episode_length": (episode.episode_length, int(row["episode_length"])),
        "instruction": (episode.instruction, row["language_instruction"]),
        "source_path": (episode.file_path, row["source_path"]),
    }
    mismatches = {key: value for key, value in checks.items() if value[0] != value[1]}
    if mismatches:
        raise ValueError(f"RLDS/index mismatch at episode {episode.episode_index}: {mismatches}")
    perturbation = SOURCE_TO_PERTURBATION[source_group]
    return source_group, suite, perturbation


def convert_splits(
    rlds_dir: Path,
    index_rows: list[dict[str, str]],
    output_root: Path,
    *,
    fps: int,
    max_episodes: int | None,
    verify_crc: bool,
    image_writer_threads: int,
    image_writer_processes: int,
    direct_jpeg: bool,
    progress_every: int,
    only_perturbation: str | None = None,
    write_master_manifest: bool = True,
) -> tuple[dict[str, LeRobotDataset], list[ManifestRow], dict[str, Any]]:
    selected_perturbations = (
        tuple(name for name in PERTURBATIONS if name != "object")
        if only_perturbation is None
        else (only_perturbation,)
    )
    if any(name not in SOURCE_TO_PERTURBATION.values() for name in selected_perturbations):
        raise ValueError(f"cannot convert selected perturbations {selected_perturbations}")
    writers = {
        perturbation: create_dataset(
            output_root,
            perturbation,
            fps=fps,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
            direct_jpeg=direct_jpeg,
        )
        for perturbation in selected_perturbations
    }

    master_manifest_path = output_root / "episode_manifest.jsonl"
    master_rows: list[ManifestRow] = []
    variant_counters: Counter[tuple[str, str, str]] = Counter()
    episode_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    suite_by_perturbation: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.monotonic()
    frame_total = 0

    try:
        for episode in iter_episodes(
            rlds_dir,
            max_episodes=max_episodes,
            verify_crc=verify_crc,
            include_source_groups=(
                None
                if only_perturbation is None
                else tuple(
                    source
                    for source, perturbation in SOURCE_TO_PERTURBATION.items()
                    if perturbation == only_perturbation
                )
            ),
        ):
            row = index_rows[episode.episode_index]
            source_group, suite, perturbation = check_episode_against_index(episode, row)
            dataset = writers[perturbation]
            output_episode_index = dataset.meta.total_episodes
            variant_key = (perturbation, suite, episode.instruction)
            variant_index = variant_counters[variant_key]
            variant_counters[variant_key] += 1

            scalar_fields = {
                "perturbation_index": np.asarray(
                    [PERTURBATION_TO_INDEX[perturbation]], dtype=np.int64
                ),
                "suite_index": np.asarray([SUITE_TO_INDEX[suite]], dtype=np.int64),
                "source_episode_id": np.asarray([episode.episode_index], dtype=np.int64),
                "variant_index": np.asarray([variant_index], dtype=np.int64),
            }
            if direct_jpeg:
                dataset._pending_jpeg_sequences = {
                    "image": episode.front_jpegs,
                    "wrist_image": episode.wrist_jpegs,
                }
            for frame_index in range(episode.episode_length):
                if direct_jpeg:
                    image_values = {}
                    opened_images = []
                    for key, jpeg in (
                        ("image", episode.front_jpegs[frame_index]),
                        ("wrist_image", episode.wrist_jpegs[frame_index]),
                    ):
                        image = Image.open(io.BytesIO(jpeg))
                        opened_images.append(image)
                        image_values[key] = image
                else:
                    image_values = {
                        "image": episode.decode_front(frame_index),
                        "wrist_image": episode.decode_wrist(frame_index),
                    }
                    opened_images = []
                try:
                    dataset.add_frame(
                        {
                            **image_values,
                            "state": episode.state[frame_index],
                            "actions": episode.action[frame_index],
                            **scalar_fields,
                            "task": episode.instruction,
                        }
                    )
                finally:
                    for image in opened_images:
                        image.close()
            dataset.save_episode()
            if direct_jpeg:
                dataset._pending_jpeg_sequences = None

            manifest_row = ManifestRow(
                output_episode_index=output_episode_index,
                source_episode_id=episode.episode_index,
                source_shard_index=episode.shard_index,
                source_record_index=episode.record_index,
                suite=suite,
                suite_index=SUITE_TO_INDEX[suite],
                task=episode.instruction,
                perturbation=perturbation,
                perturbation_index=PERTURBATION_TO_INDEX[perturbation],
                source_group=source_group,
                variant_index=variant_index,
                episode_length=episode.episode_length,
                source_path=episode.file_path,
            )
            manifest_dict = asdict(manifest_row)
            master_rows.append(manifest_row)
            if write_master_manifest:
                append_jsonl(master_manifest_path, manifest_dict)
            append_jsonl(dataset.root / "meta" / "episode_manifest.jsonl", manifest_dict)

            episode_counts[perturbation] += 1
            frame_counts[perturbation] += episode.episode_length
            suite_counts[suite] += 1
            task_counts[episode.instruction] += 1
            suite_by_perturbation[perturbation][suite] += 1
            frame_total += episode.episode_length

            converted = len(master_rows)
            if converted <= 3 or (progress_every and converted % progress_every == 0):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"converted {converted:,} episodes / {frame_total:,} frames "
                    f"({elapsed:.1f}s, {frame_total / elapsed:.1f} frames/s); "
                    f"last={perturbation}:{suite}:{output_episode_index}",
                    flush=True,
                )
    finally:
        for dataset in writers.values():
            dataset.stop_image_writer()

    summary = {
        "converted_episodes": len(master_rows),
        "converted_frames": frame_total,
        "episodes_by_perturbation": {
            name: episode_counts[name] for name in PERTURBATIONS
        },
        "frames_by_perturbation": {name: frame_counts[name] for name in PERTURBATIONS},
        "episodes_by_suite": dict(sorted(suite_counts.items())),
        "episodes_by_task": dict(sorted(task_counts.items())),
        "suite_by_perturbation": {
            name: dict(sorted(suite_by_perturbation[name].items()))
            for name in PERTURBATIONS
        },
        "elapsed_seconds_split_conversion": time.monotonic() - started,
    }
    return writers, master_rows, summary


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"category Parquet is missing required column {name!r}")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def scalar_stats(value_min: int, value_max: int, count: int) -> dict[str, np.ndarray]:
    if count <= 0:
        raise ValueError("cannot create scalar stats for an empty episode")
    mean = (value_min + value_max) / 2.0
    std = math.sqrt((count * count - 1) / 12.0) if value_max != value_min else 0.0
    return {
        "min": np.asarray([value_min], dtype=np.int64),
        "max": np.asarray([value_max], dtype=np.int64),
        "mean": np.asarray([mean], dtype=np.float64),
        "std": np.asarray([std], dtype=np.float64),
        "count": np.asarray([count], dtype=np.int64),
    }


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def encode_jpeg_bytes_to_video(
    jpeg_frames: tuple[bytes, ...],
    destination: Path,
    fps: int,
    *,
    codec: str,
    crf: int,
    gop: int,
    preset: str,
) -> None:
    if not jpeg_frames:
        raise ValueError("cannot encode an empty JPEG sequence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    av.logging.set_level(av.logging.ERROR)
    with Image.open(io.BytesIO(jpeg_frames[0])) as first:
        width, height = first.size
    options = {"g": str(gop), "crf": str(crf)}
    if codec == "h264":
        options["preset"] = preset
    with av.open(str(destination), "w") as output:
        stream = output.add_stream(codec, fps, options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for jpeg in jpeg_frames:
            with Image.open(io.BytesIO(jpeg)) as image:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
            packets = stream.encode(frame)
            if packets:
                output.mux(packets)
        packets = stream.encode()
        if packets:
            output.mux(packets)
    if not destination.is_file():
        raise OSError(f"video encoding did not create {destination}")


def configure_video_encoder(*, codec: str, crf: int, gop: int, preset: str) -> None:
    """Configure the encoder called internally by LeRobot ``save_episode``."""
    global VIDEO_ENCODER_CONFIG
    VIDEO_ENCODER_CONFIG = {"codec": codec, "crf": crf, "gop": gop, "preset": preset}
    original = lerobot_dataset_module.encode_video_frames

    def encode_with_requested_codec(
        imgs_dir: Path | str,
        video_path: Path | str,
        fps: int,
        *,
        overwrite: bool = False,
    ) -> None:
        if codec != "h264":
            original(
                imgs_dir,
                video_path,
                fps,
                vcodec=codec,
                crf=crf,
                g=gop,
                overwrite=overwrite,
            )
            return

        # LeRobot's helper uses x264's medium preset. For 2.2M 256x256 frames
        # that is needlessly expensive; the preset changes speed/size, not the
        # CRF quality target or frame semantics.
        image_paths = sorted(
            Path(imgs_dir).glob("frame_*.png"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if not image_paths:
            raise FileNotFoundError(f"no temporary images found in {imgs_dir}")
        destination = Path(video_path)
        destination.parent.mkdir(parents=True, exist_ok=overwrite)
        av.logging.set_level(av.logging.ERROR)
        with Image.open(image_paths[0]) as first:
            width, height = first.size
        with av.open(str(destination), "w") as output:
            stream = output.add_stream(
                "h264",
                fps,
                options={"g": str(gop), "crf": str(crf), "preset": preset},
            )
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    frame = av.VideoFrame.from_image(image.convert("RGB"))
                packets = stream.encode(frame)
                if packets:
                    output.mux(packets)
            packets = stream.encode()
            if packets:
                output.mux(packets)
        if not destination.is_file():
            raise OSError(f"video encoding did not create {destination}")

    lerobot_dataset_module.encode_video_frames = encode_with_requested_codec


def open_dataset_views(output_root: Path) -> dict[str, DatasetView]:
    views = {}
    for perturbation in PERTURBATIONS:
        root = output_root / DATASET_NAMES[perturbation]
        views[perturbation] = DatasetView(
            root=root,
            meta=LeRobotDatasetMetadata(DATASET_NAMES[perturbation], root=root),
        )
    return views


def parallel_conversion_worker(
    rlds_dir: Path,
    index_manifest: Path,
    output_root: Path,
    perturbation: str,
    fps: int,
    max_episodes: int | None,
    verify_crc: bool,
    direct_jpeg: bool,
    image_writer_threads: int,
    image_writer_processes: int,
    progress_every: int,
    video_codec: str,
    video_crf: int,
    video_gop: int,
    video_preset: str,
) -> tuple[str, list[ManifestRow], dict[str, Any]]:
    """Child-process entry point for one provenance group."""
    configure_video_encoder(
        codec=video_codec,
        crf=video_crf,
        gop=video_gop,
        preset=video_preset,
    )
    index_rows = load_index_manifest(index_manifest)
    _, rows, summary = convert_splits(
        rlds_dir,
        index_rows,
        output_root,
        fps=fps,
        max_episodes=max_episodes,
        verify_crc=verify_crc,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        direct_jpeg=direct_jpeg,
        progress_every=progress_every,
        only_perturbation=perturbation,
        write_master_manifest=False,
    )
    return perturbation, rows, summary


def merge_parallel_summaries(
    results: list[tuple[str, list[ManifestRow], dict[str, Any]]],
    *,
    wall_seconds: float,
) -> tuple[list[ManifestRow], dict[str, Any]]:
    rows = sorted(
        (row for _, worker_rows, _ in results for row in worker_rows),
        key=lambda row: row.source_episode_id,
    )
    frame_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    suite_by_perturbation: dict[str, Counter[str]] = defaultdict(Counter)
    worker_elapsed = {}
    for perturbation, _, worker in results:
        frame_counts.update(worker["frames_by_perturbation"])
        episode_counts.update(worker["episodes_by_perturbation"])
        suite_counts.update(worker["episodes_by_suite"])
        task_counts.update(worker["episodes_by_task"])
        for name, counts in worker["suite_by_perturbation"].items():
            suite_by_perturbation[name].update(counts)
        worker_elapsed[perturbation] = worker["elapsed_seconds_split_conversion"]
    summary = {
        "converted_episodes": len(rows),
        "converted_frames": sum(row.episode_length for row in rows),
        "episodes_by_perturbation": {
            name: episode_counts[name] for name in PERTURBATIONS
        },
        "frames_by_perturbation": {name: frame_counts[name] for name in PERTURBATIONS},
        "episodes_by_suite": dict(sorted(suite_counts.items())),
        "episodes_by_task": dict(sorted(task_counts.items())),
        "suite_by_perturbation": {
            name: dict(sorted(suite_by_perturbation[name].items()))
            for name in PERTURBATIONS
        },
        "elapsed_seconds_split_conversion": wall_seconds,
        "worker_elapsed_seconds": worker_elapsed,
        "parallel_perturbation_workers": len(results),
    }
    return rows, summary



def build_all_dataset(
    output_root: Path,
    category_datasets: dict[str, DatasetView],
    master_rows: list[ManifestRow],
    *,
    fps: int,
    progress_every: int,
) -> tuple[Path, dict[str, Any]]:
    all_root = output_root / DATASET_NAMES["all"]
    all_meta = LeRobotDatasetMetadata.create(
        repo_id=DATASET_NAMES["all"],
        root=all_root,
        robot_type="panda",
        fps=fps,
        features=feature_spec(),
        use_videos=True,
    )
    ensure_empty_jsonl_metadata(all_root)
    link_modes: Counter[str] = Counter()
    started = time.monotonic()

    for all_episode_index, row in enumerate(sorted(master_rows, key=lambda item: item.source_episode_id)):
        if row.source_episode_id != all_episode_index:
            raise ValueError(
                "all dataset requires a complete source prefix beginning at episode 0; "
                f"found source_episode_id={row.source_episode_id} at output {all_episode_index}"
            )
        category_dataset = category_datasets[row.perturbation]
        category_meta = category_dataset.meta
        category_episode_index = row.output_episode_index
        if all_meta.get_task_index(row.task) is None:
            all_meta.add_task(row.task)
        all_task_index = all_meta.get_task_index(row.task)
        assert all_task_index is not None

        category_parquet = category_dataset.root / category_meta.get_data_file_path(
            category_episode_index
        )
        table = pq.read_table(category_parquet)
        length = row.episode_length
        if table.num_rows != length:
            raise ValueError(
                f"{category_parquet} has {table.num_rows} rows; manifest expects {length}"
            )
        global_frame_start = all_meta.total_frames
        table = replace_column(
            table,
            "episode_index",
            np.full(length, all_episode_index, dtype=np.int64),
        )
        table = replace_column(
            table,
            "index",
            np.arange(global_frame_start, global_frame_start + length, dtype=np.int64),
        )
        table = replace_column(
            table,
            "task_index",
            np.full(length, all_task_index, dtype=np.int64),
        )
        all_parquet = all_root / all_meta.get_data_file_path(all_episode_index)
        all_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, all_parquet, compression="snappy")

        for video_key in category_meta.video_keys:
            source_video = category_dataset.root / category_meta.get_video_file_path(
                category_episode_index, video_key
            )
            destination_video = all_root / all_meta.get_video_file_path(
                all_episode_index, video_key
            )
            link_modes[hardlink_or_copy(source_video, destination_video)] += 1

        episode_stats = copy.deepcopy(category_meta.episodes_stats[category_episode_index])
        episode_stats["episode_index"] = scalar_stats(
            all_episode_index, all_episode_index, length
        )
        episode_stats["index"] = scalar_stats(
            global_frame_start, global_frame_start + length - 1, length
        )
        episode_stats["task_index"] = scalar_stats(all_task_index, all_task_index, length)
        all_meta.save_episode(all_episode_index, length, [row.task], episode_stats)

        all_row = asdict(row)
        all_row["output_episode_index"] = all_episode_index
        append_jsonl(all_root / "meta" / "episode_manifest.jsonl", all_row)
        if all_episode_index < 3 or (
            progress_every and (all_episode_index + 1) % progress_every == 0
        ):
            print(
                f"built all view {all_episode_index + 1:,}/{len(master_rows):,}; "
                f"videos={dict(link_modes)}",
                flush=True,
            )

    summary = {
        "episodes": all_meta.total_episodes,
        "frames": all_meta.total_frames,
        "video_link_mode_counts": dict(sorted(link_modes.items())),
        "video_payload_encoded_again": False,
        "elapsed_seconds_all_view": time.monotonic() - started,
    }
    return all_root, summary


def finalize_sidecars(
    output_root: Path,
    category_datasets: dict[str, DatasetView],
    summary: dict[str, Any],
) -> None:
    for perturbation in PERTURBATIONS:
        if perturbation == "object":
            count = 0
        else:
            count = category_datasets[perturbation].meta.total_episodes
            ensure_empty_jsonl_metadata(category_datasets[perturbation].root)
        root = output_root / DATASET_NAMES[perturbation]
        write_dataset_sidecars(root, key=perturbation, count=count)
    all_root = output_root / DATASET_NAMES["all"]
    write_dataset_sidecars(all_root, key="all", count=summary["all_view"]["episodes"])
    write_json(output_root / "perturbation_mapping.json", mapping_document())
    write_json(output_root / "conversion_summary.json", summary)
    (output_root / "README.md").write_text(
        "# LIBERO-Plus RLDS → LeRobot perturbation splits\n\n"
        "The RLDS was classified before LeRobot conversion. Five labelled source "
        "groups and an `all` union are materialized. `libero_plus_object_lerobot` "
        "is an explicit zero-episode skeleton because the released mixed RLDS has "
        "no object-labelled source group.\n\n"
        "See `conversion_summary.json`, `episode_manifest.jsonl`, and "
        "`perturbation_mapping.json` for counts and provenance.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-dir", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--jobs",
        type=int,
        default=5,
        help="parallel source-group converters; 5 encodes all released groups concurrently",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--reencode-temp-images",
        action="store_true",
        help=(
            "decode JPEGs and let LeRobot rewrite temporary PNGs; slower but follows "
            "the minimal example literally"
        ),
    )
    parser.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1"),
        default="h264",
        help="h264 is substantially faster for this 2.2M-frame conversion",
    )
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-gop", type=int, default=10)
    parser.add_argument(
        "--video-preset",
        default="ultrafast",
        help="x264 preset used when --video-codec=h264 (default: ultrafast)",
    )
    parser.add_argument("--skip-crc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if not 1 <= args.jobs <= 5:
        parser.error("--jobs must be in 1..5")
    if (
        args.reencode_temp_images
        and args.image_writer_threads <= 0
        and args.image_writer_processes <= 0
    ):
        parser.error("at least one image-writer thread/process is required")
    if not 0 <= args.video_crf <= 63:
        parser.error("--video-crf must be in 0..63")
    if args.video_gop <= 0:
        parser.error("--video-gop must be positive")

    rlds_dir = args.rlds_dir.resolve()
    index_manifest = args.index_manifest.resolve()
    output_root = args.output_root.resolve(strict=False)
    if not rlds_dir.is_dir():
        parser.error(f"RLDS directory does not exist: {rlds_dir}")
    if not index_manifest.is_file():
        parser.error(f"index manifest does not exist: {index_manifest}")
    if output_root == rlds_dir or rlds_dir in output_root.parents:
        parser.error("output root must not be the source RLDS directory or its child")

    index_rows = load_index_manifest(index_manifest)
    source_total = sum(load_shard_lengths(rlds_dir))
    if len(index_rows) != source_total:
        parser.error(
            f"index manifest has {len(index_rows)} rows; RLDS metadata expects {source_total}"
        )
    expected_episodes = (
        source_total if args.max_episodes is None else min(source_total, args.max_episodes)
    )
    prepare_output_root(output_root, overwrite=args.overwrite)
    create_empty_object_dataset(output_root, fps=args.fps)
    write_json(output_root / "perturbation_mapping.json", mapping_document())
    write_json(
        output_root / "conversion_state.json",
        {
            "status": "in_progress",
            "rlds_dir": str(rlds_dir),
            "index_manifest": str(index_manifest),
            "expected_episodes": expected_episodes,
            "fps": args.fps,
            "video_codec": args.video_codec,
            "video_crf": args.video_crf,
            "video_gop": args.video_gop,
            "video_preset": args.video_preset,
            "parallel_jobs": args.jobs,
            "source_image_pipeline": (
                "decoded_png_temp_files"
                if args.reencode_temp_images
                else "source_jpeg_in_memory"
            ),
        },
    )

    if args.jobs == 1:
        configure_video_encoder(
            codec=args.video_codec,
            crf=args.video_crf,
            gop=args.video_gop,
            preset=args.video_preset,
        )
        _, master_rows, split_summary = convert_splits(
            rlds_dir,
            index_rows,
            output_root,
            fps=args.fps,
            max_episodes=args.max_episodes,
            verify_crc=not args.skip_crc,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            direct_jpeg=not args.reencode_temp_images,
            progress_every=args.progress_every,
        )
    else:
        parallel_started = time.monotonic()
        worker_args = [
            (
                rlds_dir,
                index_manifest,
                output_root,
                perturbation,
                args.fps,
                args.max_episodes,
                not args.skip_crc,
                not args.reencode_temp_images,
                args.image_writer_threads,
                args.image_writer_processes,
                args.progress_every,
                args.video_codec,
                args.video_crf,
                args.video_gop,
                args.video_preset,
            )
            for perturbation in PERTURBATIONS
            if perturbation != "object"
        ]
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = list(executor.map(parallel_conversion_worker, *zip(*worker_args)))
        master_rows, split_summary = merge_parallel_summaries(
            results,
            wall_seconds=time.monotonic() - parallel_started,
        )
        for row in master_rows:
            append_jsonl(output_root / "episode_manifest.jsonl", asdict(row))
    if len(master_rows) != expected_episodes:
        raise ValueError(
            f"converted {len(master_rows)} episodes, expected {expected_episodes}"
        )
    source_ids = [row.source_episode_id for row in master_rows]
    if source_ids != list(range(expected_episodes)):
        raise ValueError("parallel split results do not cover each source episode exactly once")
    category_datasets = open_dataset_views(output_root)
    all_root, all_summary = build_all_dataset(
        output_root,
        category_datasets,
        master_rows,
        fps=args.fps,
        progress_every=args.progress_every,
    )
    del all_root
    summary = {
        "schema_version": 1,
        "source_rlds_dir": str(rlds_dir),
        "source_index_manifest": str(index_manifest),
        "fps": args.fps,
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "video_gop": args.video_gop,
        "video_preset": args.video_preset,
        "parallel_jobs": args.jobs,
        "source_image_pipeline": (
            "decoded_png_temp_files"
            if args.reencode_temp_images
            else "source_jpeg_in_memory"
        ),
        "full_release": args.max_episodes is None,
        "split_conversion": split_summary,
        "all_view": all_summary,
        "datasets": {key: DATASET_NAMES[key] for key in (*PERTURBATIONS, "all")},
        "mapping": mapping_document(),
    }
    finalize_sidecars(output_root, category_datasets, summary)
    write_json(
        output_root / "conversion_state.json",
        {
            "status": "complete",
            "converted_episodes": len(master_rows),
            "converted_frames": split_summary["converted_frames"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
