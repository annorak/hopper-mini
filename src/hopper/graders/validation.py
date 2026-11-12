import hashlib
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from hopper.config import Settings
from hopper.graders import (
    Catalog,
    JevGrader,
    ThresholdValidation,
    Validation,
    definition_fingerprint,
    threshold_pairs,
)
from hopper.graders.jev import JevClient, canonical_state
from hopper.labels import Label, read_labels
from hopper.ledger import Ledger


def _metrics(
    labels: list[Label], probabilities: list[float], no: float, yes: float
) -> ThresholdValidation:
    agreement = predicted_no = correct_no = human_no = 0
    for evidence, probability in zip(labels, probabilities, strict=True):
        predicted = "yes" if probability >= yes else "no" if probability <= no else "unsure"
        agreement += predicted == evidence.label
        predicted_no += predicted == "no"
        human_no += evidence.label == "no"
        correct_no += predicted == evidence.label == "no"
    return ThresholdValidation(
        no_threshold=no,
        yes_threshold=yes,
        agreement=agreement / len(labels) if labels else None,
        precision_no=correct_no / predicted_no if predicted_no else None,
        recall_no=correct_no / human_no if human_no else None,
    )


async def _judge_labels(client: JevClient, grader: JevGrader, labels: list[Label]) -> list[float]:
    probabilities = []
    for evidence in labels:
        judgment = await client.judge(evidence.state, (grader,))
        if judgment.input_hash != evidence.input_hash:
            raise ValueError(
                f"Validation input changed for {grader.name} v{grader.version}; "
                "the saved human label cannot grade a different state"
            )
        probabilities.append(judgment.answers[0].p_yes)
    return probabilities


async def validate_graders(
    settings: Settings,
    graders: list[JevGrader],
    labels_dir: Path,
    rps: float = 5,
    usd_per_hour: Decimal = Decimal("1"),
) -> list[tuple[JevGrader, Validation]]:
    catalog = Catalog(settings.data_dir)
    reports = []
    client = None
    async with AsyncExitStack() as resources:
        for grader in graders:
            labels = read_labels(labels_dir, grader)
            if labels and client is None:
                ledger = Ledger(settings)
                ledger.initialize()
                client = await resources.enter_async_context(
                    JevClient(ledger, rps=rps, usd_per_hour=usd_per_hour)
                )
            probabilities = await _judge_labels(client, grader, labels) if labels else []
            sweep = tuple(
                _metrics(labels, probabilities, no, yes) for no, yes in threshold_pairs(grader)
            )
            current = next(
                row
                for row in sweep
                if (row.no_threshold, row.yes_threshold)
                == (grader.no_threshold, grader.yes_threshold)
            )
            evidence = [
                {"state": item.state, "input_hash": item.input_hash, "label": item.label}
                for item in sorted(labels, key=lambda item: item.input_hash)
            ]
            report = Validation(
                labels=len(labels),
                agreement=current.agreement,
                precision_no=current.precision_no,
                recall_no=current.recall_no,
                threshold_sweep=sweep,
                validated_at=datetime.now(UTC).isoformat(),
                definition_hash=definition_fingerprint(grader),
                labels_hash=hashlib.sha256(
                    canonical_state({"labels": evidence}).encode()
                ).hexdigest(),
            )
            reports.append((catalog.save_validation(grader, report), report))
    return reports
