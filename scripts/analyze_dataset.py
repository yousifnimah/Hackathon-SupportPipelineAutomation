"""Validate and describe the Iraqi Arabic support-ticket dataset.

Usage:
    python scripts/analyze_dataset.py \
        --dataset data/iraqi_support_dataset_3000.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CATEGORIES = (
    "failed_transfer",
    "wrong_recipient",
    "payment_pending",
    "login_problem",
    "card_issue",
    "account_issue",
    "agent_dispute",
    "other",
)
PRIORITIES = ("high", "medium", "low")
ENTITY_FIELDS = ("amount", "transaction_id", "recipient", "account_id", "date")
REQUIRED_TICKET_FIELDS = {"id", "category", "priority", "messages", "expected"}
REQUIRED_MESSAGE_FIELDS = {"msg_id", "seq", "text"}
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_TEST_REMAINDER_FRACTION = 0.85
ARABIC_INDIC_DIGIT_RE = re.compile(r"[\u0660-\u0669\u06f0-\u06f9]")
WESTERN_DIGIT_RE = re.compile(r"[0-9]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


class DatasetValidationError(ValueError):
    """Raised when a dataset record does not match the expected schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the support-ticket dataset and generate statistics/figures."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Input JSON file")
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=Path("artifacts/dataset_statistics.json"),
        help="Machine-readable JSON summary",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/dataset_figures"),
        help="Directory for PNG and SVG figures",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Initial sequential training fraction (repository default: 0.70)",
    )
    parser.add_argument(
        "--test-remainder-fraction",
        type=float,
        default=DEFAULT_TEST_REMAINDER_FRACTION,
        help=(
            "Fraction of the post-training remainder assigned to test "
            "(repository default: 0.85)"
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"Dataset is not valid JSON: {error}") from error
    if not isinstance(records, list) or not records:
        raise DatasetValidationError("Dataset root must be a non-empty JSON array")
    return records


def validate_dataset(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    seen_ticket_ids: dict[int, int] = {}
    seen_message_ids: dict[int, str] = {}

    for index, ticket in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(ticket, dict):
            errors.append(f"{prefix}: expected an object")
            continue

        missing_ticket_fields = REQUIRED_TICKET_FIELDS - set(ticket)
        if missing_ticket_fields:
            errors.append(
                f"{prefix}: missing fields {sorted(missing_ticket_fields)}"
            )
            continue

        if not isinstance(ticket["id"], int) or isinstance(ticket["id"], bool):
            errors.append(f"{prefix}.id: expected an integer")
        elif ticket["id"] in seen_ticket_ids:
            errors.append(
                f"{prefix}.id: duplicate value {ticket['id']}; first seen at "
                f"record[{seen_ticket_ids[ticket['id']]}]"
            )
        else:
            seen_ticket_ids[ticket["id"]] = index
        if ticket["category"] not in CATEGORIES:
            errors.append(f"{prefix}.category: invalid value {ticket['category']!r}")
        if ticket["priority"] not in PRIORITIES:
            errors.append(f"{prefix}.priority: invalid value {ticket['priority']!r}")

        messages = ticket["messages"]
        if not isinstance(messages, list) or not messages:
            errors.append(f"{prefix}.messages: expected a non-empty array")
        else:
            sequences: list[int] = []
            for message_index, message in enumerate(messages):
                message_prefix = f"{prefix}.messages[{message_index}]"
                if not isinstance(message, dict):
                    errors.append(f"{message_prefix}: expected an object")
                    continue
                missing_message_fields = REQUIRED_MESSAGE_FIELDS - set(message)
                if missing_message_fields:
                    errors.append(
                        f"{message_prefix}: missing fields "
                        f"{sorted(missing_message_fields)}"
                    )
                    continue
                if not isinstance(message["msg_id"], int) or isinstance(
                    message["msg_id"], bool
                ):
                    errors.append(f"{message_prefix}.msg_id: expected an integer")
                elif message["msg_id"] in seen_message_ids:
                    errors.append(
                        f"{message_prefix}.msg_id: duplicate value "
                        f"{message['msg_id']}; first seen at "
                        f"{seen_message_ids[message['msg_id']]}"
                    )
                else:
                    seen_message_ids[message["msg_id"]] = message_prefix
                if not isinstance(message["seq"], int) or isinstance(
                    message["seq"], bool
                ):
                    errors.append(f"{message_prefix}.seq: expected an integer")
                else:
                    sequences.append(message["seq"])
                if not isinstance(message["text"], str) or not message["text"].strip():
                    errors.append(f"{message_prefix}.text: expected non-empty text")
            if sorted(sequences) != list(range(len(messages))):
                errors.append(
                    f"{prefix}.messages: seq values must be contiguous from zero"
                )

        expected = ticket["expected"]
        if not isinstance(expected, dict):
            errors.append(f"{prefix}.expected: expected an object")
        elif set(expected) != set(ENTITY_FIELDS):
            errors.append(
                f"{prefix}.expected: expected exactly {list(ENTITY_FIELDS)}; "
                f"got {sorted(expected)}"
            )
        else:
            amount = expected["amount"]
            if amount is not None and (
                not isinstance(amount, int) or isinstance(amount, bool)
            ):
                errors.append(f"{prefix}.expected.amount: expected integer or null")
            for field in ENTITY_FIELDS[1:]:
                value = expected[field]
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    errors.append(
                        f"{prefix}.expected.{field}: expected non-empty string or null"
                    )

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
        raise DatasetValidationError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            f"{preview}{suffix}"
        )


def percentage(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def count_summary(counts: Counter[str], labels: tuple[str, ...], total: int) -> dict:
    return {
        label: {"count": counts[label], "percentage": percentage(counts[label], total)}
        for label in labels
    }


def reproduce_repository_split(
    records: list[dict[str, Any]],
    train_fraction: float,
    test_remainder_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    if not 0 < test_remainder_fraction < 1:
        raise ValueError("--test-remainder-fraction must be between zero and one")

    train_end = int(len(records) * train_fraction)
    remainder = records[train_end:]
    test_end = int(len(remainder) * test_remainder_fraction)
    splits = {
        "train": records[:train_end],
        "test": remainder[:test_end],
        "validation": remainder[test_end:],
    }
    if any(not split_records for split_records in splits.values()):
        raise ValueError("Configured split produced an empty partition")
    return splits


def summarize_splits(
    splits: dict[str, list[dict[str, Any]]], total: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "method": (
            "Sequential repository-order slicing without shuffling or stratification"
        ),
        "notebook_formula": (
            "train = first 70%; test = first 85% of remainder; "
            "validation = rest of remainder"
        ),
        "splits": {},
    }
    for split_name in ("train", "validation", "test"):
        split_records = splits[split_name]
        split_total = len(split_records)
        category_counts = Counter(record["category"] for record in split_records)
        priority_counts = Counter(record["priority"] for record in split_records)
        summary["splits"][split_name] = {
            "count": split_total,
            "percentage_of_dataset": percentage(split_total, total),
            "first_ticket_id": split_records[0]["id"],
            "last_ticket_id": split_records[-1]["id"],
            "categories": count_summary(
                category_counts, CATEGORIES, split_total
            ),
            "priorities": count_summary(priority_counts, PRIORITIES, split_total),
        }
    return summary


def duplicate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ticket_ids = [ticket["id"] for ticket in records]
    all_messages = [
        message for ticket in records for message in ticket["messages"]
    ]
    message_ids = [message["msg_id"] for message in all_messages]
    message_texts = [message["text"] for message in all_messages]
    normalized_texts = [" ".join(text.lower().split()) for text in message_texts]

    def repeated_values(values: list[Any]) -> list[dict[str, Any]]:
        counter = Counter(values)
        return [
            {"value": value, "count": count}
            for value, count in counter.items()
            if count > 1
        ]

    duplicate_ticket_ids = repeated_values(ticket_ids)
    duplicate_message_ids = repeated_values(message_ids)
    exact_text_counts = Counter(message_texts)
    normalized_text_counts = Counter(normalized_texts)

    return {
        "duplicate_ticket_id_values": len(duplicate_ticket_ids),
        "tickets_with_repeated_ids_beyond_first": sum(
            item["count"] - 1 for item in duplicate_ticket_ids
        ),
        "duplicate_message_id_values": len(duplicate_message_ids),
        "messages_with_repeated_ids_beyond_first": sum(
            item["count"] - 1 for item in duplicate_message_ids
        ),
        "exact_duplicate_text_groups": sum(
            count > 1 for count in exact_text_counts.values()
        ),
        "messages_in_exact_duplicate_groups": sum(
            count for count in exact_text_counts.values() if count > 1
        ),
        "exact_duplicate_messages_beyond_first": sum(
            count - 1 for count in exact_text_counts.values() if count > 1
        ),
        "normalized_duplicate_text_groups": sum(
            count > 1 for count in normalized_text_counts.values()
        ),
        "normalized_duplicate_messages_beyond_first": sum(
            count - 1 for count in normalized_text_counts.values() if count > 1
        ),
        "normalization": "Unicode-preserving lowercase plus collapsed whitespace",
    }


def calculate_statistics(
    records: list[dict[str, Any]],
    dataset_path: Path,
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    total = len(records)
    category_counts = Counter(ticket["category"] for ticket in records)
    priority_counts = Counter(ticket["priority"] for ticket in records)
    message_counts = [len(ticket["messages"]) for ticket in records]
    combined_texts = [
        "\n".join(message["text"] for message in ticket["messages"])
        for ticket in records
    ]
    text_lengths = [len(text) for text in combined_texts]

    category_priority = pd.crosstab(
        pd.Categorical(
            [ticket["category"] for ticket in records], categories=CATEGORIES
        ),
        pd.Categorical(
            [ticket["priority"] for ticket in records], categories=PRIORITIES
        ),
        dropna=False,
    ).reindex(index=CATEGORIES, columns=PRIORITIES, fill_value=0)

    entity_statistics: dict[str, dict[str, float | int]] = {}
    for field in ENTITY_FIELDS:
        present = sum(ticket["expected"][field] is not None for ticket in records)
        entity_statistics[field] = {
            "present_count": present,
            "present_percentage": percentage(present, total),
            "missing_or_null_count": total - present,
            "missing_or_null_percentage": percentage(total - present, total),
        }

    single_message = sum(count == 1 for count in message_counts)
    multi_message = sum(count > 1 for count in message_counts)
    arabic_indic = sum(bool(ARABIC_INDIC_DIGIT_RE.search(text)) for text in combined_texts)
    western_digits = sum(bool(WESTERN_DIGIT_RE.search(text)) for text in combined_texts)
    latin_letters = sum(bool(LATIN_LETTER_RE.search(text)) for text in combined_texts)

    statistics_payload: dict[str, Any] = {
        "dataset_path": dataset_path.as_posix(),
        "schema_validation": "passed",
        "total_tickets": total,
        "total_messages": sum(message_counts),
        "category_count": len(category_counts),
        "categories": count_summary(category_counts, CATEGORIES, total),
        "priorities": count_summary(priority_counts, PRIORITIES, total),
        "category_priority_counts": {
            category: {
                priority: int(category_priority.loc[category, priority])
                for priority in PRIORITIES
            }
            for category in CATEGORIES
        },
        "messages_per_ticket": {
            "single_message_ticket_count": single_message,
            "single_message_ticket_percentage": percentage(single_message, total),
            "multi_message_ticket_count": multi_message,
            "multi_message_ticket_percentage": percentage(multi_message, total),
            "average": round(statistics.fmean(message_counts), 3),
            "median": statistics.median(message_counts),
            "minimum": min(message_counts),
            "maximum": max(message_counts),
            "distribution": {
                str(count): frequency
                for count, frequency in sorted(Counter(message_counts).items())
            },
        },
        "combined_text_length_characters": {
            "definition": (
                "Character count after joining a ticket's message texts with one newline"
            ),
            "average": round(statistics.fmean(text_lengths), 3),
            "median": statistics.median(text_lengths),
            "minimum": min(text_lengths),
            "maximum": max(text_lengths),
        },
        "entity_availability": entity_statistics,
        "text_character_prevalence": {
            "arabic_indic_or_eastern_arabic_digits": {
                "ticket_count": arabic_indic,
                "ticket_percentage": percentage(arabic_indic, total),
            },
            "western_digits": {
                "ticket_count": western_digits,
                "ticket_percentage": percentage(western_digits, total),
            },
            "latin_letters": {
                "ticket_count": latin_letters,
                "ticket_percentage": percentage(latin_letters, total),
                "note": (
                    "Latin-letter presence is measurable but does not by itself prove "
                    "linguistic code-switching."
                ),
            },
        },
        "duplicates": duplicate_summary(records),
        "repository_split": summarize_splits(splits, total),
        "token_length_statistics": (
            "not calculated; MODEL_NAME is configured via .env, but tokenizer/model "
            "files are external to the versioned repository"
        ),
    }

    ticket_frame = pd.DataFrame(
        {
            "id": [ticket["id"] for ticket in records],
            "category": [ticket["category"] for ticket in records],
            "priority": [ticket["priority"] for ticket in records],
            "message_count": message_counts,
            "text_length_characters": text_lengths,
        }
    )
    return statistics_payload, ticket_frame, category_priority


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4b5563",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "svg.hashsalt": "dataset-figures",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)


def annotate_bars(axis: plt.Axes, orientation: str = "vertical") -> None:
    for patch in axis.patches:
        if orientation == "horizontal":
            value = int(round(patch.get_width()))
            axis.annotate(
                f"{value:,}",
                (patch.get_width(), patch.get_y() + patch.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        else:
            value = int(round(patch.get_height()))
            axis.annotate(
                f"{value:,}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )


def generate_figures(
    ticket_frame: pd.DataFrame,
    category_priority: pd.DataFrame,
    statistics_payload: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    category_counts = (
        ticket_frame["category"]
        .value_counts()
        .reindex(CATEGORIES, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(category_counts.index, category_counts.values, color="#2563eb")
    axis.invert_yaxis()
    axis.set_title("Category Distribution")
    axis.set_xlabel("Tickets")
    axis.set_ylabel("Category")
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    axis.set_xlim(0, category_counts.max() * 1.14)
    annotate_bars(axis, orientation="horizontal")
    save_figure(fig, output_dir, "01_category_distribution")

    priority_counts = (
        ticket_frame["priority"]
        .value_counts()
        .reindex(PRIORITIES, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(priority_counts.index, priority_counts.values, color="#0f766e")
    axis.set_title("Priority Distribution")
    axis.set_xlabel("Priority")
    axis.set_ylabel("Tickets")
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_ylim(0, priority_counts.max() * 1.13)
    annotate_bars(axis)
    save_figure(fig, output_dir, "02_priority_distribution")

    fig, axis = plt.subplots(figsize=(7.8, 6.2))
    image = axis.imshow(category_priority.values, cmap="Blues", aspect="auto")
    axis.set_title("Category × Priority Distribution")
    axis.set_xlabel("Priority")
    axis.set_ylabel("Category")
    axis.set_xticks(range(len(PRIORITIES)), labels=PRIORITIES)
    axis.set_yticks(range(len(CATEGORIES)), labels=CATEGORIES)
    for row in range(len(CATEGORIES)):
        for column in range(len(PRIORITIES)):
            value = int(category_priority.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > category_priority.values.max() * 0.55 else "#111827",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Tickets", shrink=0.82)
    save_figure(fig, output_dir, "03_category_priority_distribution")

    availability = statistics_payload["entity_availability"]
    entity_percentages = [availability[field]["present_percentage"] for field in ENTITY_FIELDS]
    entity_counts = [availability[field]["present_count"] for field in ENTITY_FIELDS]
    fig, axis = plt.subplots(figsize=(8.2, 4.9))
    bars = axis.bar(ENTITY_FIELDS, entity_percentages, color="#7c3aed")
    axis.set_title("Entity Availability")
    axis.set_xlabel("Entity field")
    axis.set_ylabel("Tickets with a non-null value (%)")
    axis.set_ylim(0, max(1, max(entity_percentages) * 1.18))
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=20)
    for bar, count, value in zip(bars, entity_counts, entity_percentages):
        axis.annotate(
            f"{count:,}\n({value:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    save_figure(fig, output_dir, "04_entity_availability")

    fig, axis = plt.subplots(figsize=(8.2, 4.9))
    axis.hist(
        ticket_frame["text_length_characters"],
        bins=30,
        color="#d97706",
        edgecolor="white",
        linewidth=0.5,
    )
    axis.axvline(
        ticket_frame["text_length_characters"].median(),
        color="#991b1b",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {ticket_frame['text_length_characters'].median():.0f}",
    )
    axis.set_title("Combined Ticket Text-Length Distribution")
    axis.set_xlabel("Characters per ticket")
    axis.set_ylabel("Tickets")
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    save_figure(fig, output_dir, "05_ticket_text_length_distribution")

    split_order = ("train", "validation", "test")
    display_names = ("Train", "Validation", "Test")
    split_counts = [len(splits[name]) for name in split_order]
    split_category_counts = pd.DataFrame(
        {
            name: Counter(record["category"] for record in splits[name])
            for name in split_order
        }
    ).reindex(CATEGORIES, fill_value=0).T.fillna(0).astype(int)
    split_category_percentages = split_category_counts.div(
        split_category_counts.sum(axis=1), axis=0
    ) * 100

    fig, (count_axis, mix_axis) = plt.subplots(
        1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [0.8, 2.2]}
    )
    count_axis.bar(display_names, split_counts, color="#0369a1")
    count_axis.set_title("Split Size")
    count_axis.set_xlabel("Split")
    count_axis.set_ylabel("Tickets")
    count_axis.grid(axis="y")
    count_axis.set_axisbelow(True)
    count_axis.set_ylim(0, max(split_counts) * 1.14)
    annotate_bars(count_axis)

    color_max = max(15, float(split_category_percentages.values.max()))
    image = mix_axis.imshow(
        split_category_percentages.values,
        cmap="YlGnBu",
        aspect="auto",
        vmin=0,
        vmax=color_max,
    )
    mix_axis.set_title("Category Composition Within Each Split")
    mix_axis.set_xlabel("Category")
    mix_axis.set_ylabel("Split")
    mix_axis.set_xticks(
        range(len(CATEGORIES)), labels=CATEGORIES, rotation=35, ha="right"
    )
    mix_axis.set_yticks(range(len(split_order)), labels=display_names)
    for row in range(len(split_order)):
        for column in range(len(CATEGORIES)):
            count = int(split_category_counts.iloc[row, column])
            proportion = split_category_percentages.iloc[row, column]
            mix_axis.text(
                column,
                row,
                f"{count}\n{proportion:.1f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if proportion > color_max * 0.5
                    else "#111827"
                ),
                fontsize=10,
            )
    fig.colorbar(image, ax=mix_axis, label="Within-split percentage", shrink=0.8)
    fig.suptitle("Repository Train / Validation / Test Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "06_split_distribution")


def print_summary(statistics_payload: dict[str, Any], figures_dir: Path) -> None:
    messages = statistics_payload["messages_per_ticket"]
    lengths = statistics_payload["combined_text_length_characters"]
    print("Dataset validation: passed")
    print(f"Tickets: {statistics_payload['total_tickets']:,}")
    print(f"Messages: {statistics_payload['total_messages']:,}")
    print(
        "Single/multi-message tickets: "
        f"{messages['single_message_ticket_count']:,}/"
        f"{messages['multi_message_ticket_count']:,}"
    )
    print(
        "Average/median combined length: "
        f"{lengths['average']:.1f}/{lengths['median']:.1f} characters"
    )
    split_summary = statistics_payload["repository_split"]["splits"]
    print(
        "Train/validation/test: "
        f"{split_summary['train']['count']:,}/"
        f"{split_summary['validation']['count']:,}/"
        f"{split_summary['test']['count']:,}"
    )
    print(f"Figures: {figures_dir}")


def main() -> None:
    args = parse_args()
    records = load_dataset(args.dataset)
    validate_dataset(records)
    splits = reproduce_repository_split(
        records, args.train_fraction, args.test_remainder_fraction
    )
    statistics_payload, ticket_frame, category_priority = calculate_statistics(
        records, args.dataset, splits
    )

    generate_figures(
        ticket_frame,
        category_priority,
        statistics_payload,
        splits,
        args.figures_dir,
    )

    args.statistics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.statistics_output.open("w", encoding="utf-8") as handle:
        json.dump(statistics_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print_summary(statistics_payload, args.figures_dir)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, DatasetValidationError, ValueError) as error:
        sys.exit(f"error: {error}")
