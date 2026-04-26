"""
build_labelled_manifest.py
--------------------------
Convert unlabelled manifest + annotations.csv into a labelled manifest for finetuning.

Maps spectrogram files back to their source audio segments, finds overlapping
annotations, and assigns species labels.

Usage:
    python build_labelled_manifest.py \
        --unlabelled_manifest ./new_processed_data/manifest.json \
        --annotations ../annotations.csv \
        --output ./labelled_manifest.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict


# Constants must match preprocess.py
CLIP_DURATION = 3.0
HOP_DURATION = 1.5
SAMPLE_RATE = 48_000

def model_size_from_ckpt(ckpt_path: str) -> Optional[int]:
    """Infer model size (number of classes) from checkpoint filename, if possible."""
    stem = Path(ckpt_path).stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def parse_spectrogram_path(spec_path: str) -> Optional[tuple[str, int]]:
    """
    Extract source filename and segment index from a spectrogram path.
    
    Example: "new_processed_data\\spectrograms\\PER_001_S01_20190116_100007Z_0000.npy"
    Returns: ("PER_001_S01_20190116_100007Z.flac", 0)
    """
    stem = Path(spec_path).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    
    base_name = parts[0]
    try:
        seg_idx = int(parts[1])
    except ValueError:
        return None
    
    return (base_name, seg_idx)


def get_segment_time_range(seg_idx: int) -> tuple[float, float]:
    """Calculate startTime and endTime (in seconds) for a segment index."""
    start_sec = seg_idx * HOP_DURATION
    end_sec = start_sec + CLIP_DURATION
    return (start_sec, end_sec)


def annotation_overlaps(ann_start: float, ann_end: float, seg_start: float, seg_end: float, min_overlap: float = 0.5) -> bool:
    """Check if annotation overlaps with segment by at least min_overlap seconds."""
    overlap = min(ann_end, seg_end) - max(ann_start, seg_start)
    return overlap >= min_overlap


def build_labelled_manifest(
    unlabelled_manifest_path: str,
    annotations_path: str,
    output_path: str,
) -> None:
    """
    Build a labelled manifest by matching spectrograms to annotations.
    """
    # Load unlabelled manifest
    with open(unlabelled_manifest_path, "r") as f:
        unlabelled = json.load(f)
    spec_paths = unlabelled.get("spectrograms", [])
    print(f"Loaded {len(spec_paths)} spectrograms from {unlabelled_manifest_path}")
    
    # Load annotations
    annotations_by_file = defaultdict(list)
    with open(annotations_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["Filename"]
            # Remove file extension for matching (e.g., "file.flac" → "file")
            filename_stem = Path(filename).stem
            try:
                start = float(row["Start Time (s)"])
                end = float(row["End Time (s)"])
                species_code = row["Species eBird Code"].strip()
                if species_code and species_code != "????":
                    annotations_by_file[filename_stem].append({
                        "start": start,
                        "end": end,
                        "species": species_code,
                    })
            except (ValueError, KeyError):
                continue
    
    print(f"Loaded {len(annotations_by_file)} annotated files")
    
    # Map spectrograms to labels
    samples = []
    class_set = set()
    matched_count = 0
    unmatched_count = 0
    
    for spec_path in spec_paths:
        parsed = parse_spectrogram_path(spec_path)
        if not parsed:
            continue
        
        base_name, seg_idx = parsed
        seg_start, seg_end = get_segment_time_range(seg_idx)
        
        # Try to find matching annotation
        found_label = None
        if base_name in annotations_by_file:
            # Find best (most overlapping) annotation
            best_overlap = 0
            for ann in annotations_by_file[base_name]:
                if annotation_overlaps(ann["start"], ann["end"], seg_start, seg_end):
                    overlap = min(ann["end"], seg_end) - max(ann["start"], seg_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        found_label = ann["species"]
        
        if found_label:
            samples.append({
                "path": spec_path,
                "label": found_label,
            })
            class_set.add(found_label)
            matched_count += 1
        else:
            unmatched_count += 1
    
    classes = sorted(list(class_set))
    
    print(f"\nMatching results:")
    print(f"  Matched: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    print(f"  Total classes: {len(classes)}")
    
    # Write labelled manifest
    labelled = {
        "samples": samples,
        "classes": classes,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labelled, f, indent=2)
    
    print(f"\nLabelled manifest saved → {output_path}")
    print(f"  {len(samples)} labelled samples")
    print(f"  {len(classes)} classes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build labelled manifest for finetuning from annotations"
    )
    p.add_argument(
        "--unlabelled_manifest",
        required=True,
        help="Path to unlabelled manifest (from preprocess.py)"
    )
    p.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.csv"
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for labelled manifest JSON"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_labelled_manifest(
        args.unlabelled_manifest,
        args.annotations,
        args.output,
    )
