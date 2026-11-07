import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from hopper import corpus


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "sources"
    for directory in ("specs", "drawings", "addenda"):
        (root / directory).mkdir(parents=True)
    specs = {
        identifier: {
            f"1.{number}": f"Execute the {identifier} work package with approved materials."
            for number in range(1, 8)
        }
        for identifier in corpus.SPEC_IDS
    }
    precedence = "When drawings conflict with specifications, specifications govern."
    specs["SPEC-01"]["1.1"] = precedence
    drawing_notes = {
        identifier: [
            {"number": number, "text": "Coordinate the work with approved contract documents."}
            for number in range(1, 7)
        ]
        for identifier in corpus.DRAWING_IDS
    }
    conflicts = []
    for index, drawing_id in enumerate(("M-101", "M-102", "M-501", "E-101", "I-101", "S-101")):
        spec_id = f"SPEC-{index + 2:02d}"
        spec_text = (
            "Provide a ball valve on the discharge header." if index == 0
            else f"Use the specified material for installation conflict {index + 1}."
        )
        drawing_text = (
            "Provide a butterfly valve on the discharge header." if index == 0
            else f"Use the alternate material for installation conflict {index + 1}."
        )
        specs[spec_id]["1.1"] = spec_text
        drawing_notes[drawing_id][0]["text"] = drawing_text
        conflicts.append({
            "id": f"CONFLICT-{index + 1:02d}", "topic": f"installation conflict {index + 1}",
            "spec": f"{spec_id}/1.1", "drawing": f"{drawing_id}/1",
            "spec_quote": spec_text, "drawing_quote": drawing_text,
            "resolution_quote": spec_text,
        })
    addenda = {identifier: {} for identifier in corpus.ADDENDUM_IDS}
    changes = []
    for index in range(9):
        spec_id = f"SPEC-{index + 8:02d}"
        addendum_id, item = f"ADD-{index // 3 + 1}", str(index % 3 + 1)
        before, after = "Provide the base finish.", "Provide the revised finish."
        specs[spec_id]["1.1"] = before
        addenda[addendum_id][item] = after
        changes.append({
            "id": f"{addendum_id}/{item}", "section": f"{spec_id}/1.1",
            "topic": f"finish revision {index + 1}", "before": before, "after": after,
        })
    gaps = []
    for index in range(10):
        spec_id, number = f"SPEC-{17 + index // 7:02d}", f"1.{index % 7 + 1}"
        text = f"The engineer must decide the final arrangement for condition {index + 1}."
        specs[spec_id][number] = text
        gaps.append({
            "id": f"GAP-{index + 1:02d}", "section": f"{spec_id}/{number}",
            "topic": f"engineer decision {index + 1}", "quote": text,
        })
    calculations = []
    for index in range(25):
        inputs = []
        for side, spec_id in enumerate(("SPEC-19", "SPEC-20")):
            value = index + 2 + side
            quote = f"The {['supply', 'return'][side]} quantity for this check is {value} units."
            number = f"1.{index % 7 + 1}"
            specs[spec_id][number] += "\n" + quote
            inputs.append({
                "name": ["supply", "return"][side], "citation": f"{spec_id}/{number}",
                "quote": quote, "value": value,
            })
        calculations.append({
            "id": f"CALC-{index + 1:02d}", "topic": f"quantity check {index + 1}",
            "operation": ("add", "subtract", "multiply", "divide")[index % 4],
            "inputs": inputs, "unit": "units", "tolerance": 0.01,
        })
    for directory, documents in (("specs", specs), ("addenda", addenda)):
        for identifier, sections in documents.items():
            text = f"# {identifier} Contract requirements\n\n" + "\n\n".join(
                f"## {number} Requirement\n\n{body}" for number, body in sections.items()
            ) + "\n"
            (root / directory / f"{identifier}.md").write_text(text)
    for identifier, notes in drawing_notes.items():
        write_yaml(root / "drawings" / f"{identifier}.yaml", {
            "id": identifier, "title": "Contract installation requirements", "notes": notes,
        })
    write_yaml(root / "cases.yaml", {
        "precedence": "SPEC-01/1.1", "conflicts": conflicts, "changes": changes,
        "gaps": gaps, "calculations": calculations,
    })
    return root


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_load_corpus_counts_and_readable_documents(source_root):
    loaded = corpus.load_corpus(source_root)
    assert len(loaded.version) == 64
    assert len(loaded.documents) == 35
    assert len(loaded.sections) == 140 + 72 + 9
    assert len(loaded.drawings) == 12
    assert loaded.sections["M-101/1"] == loaded.drawings["M-101"].notes[0]
    assert "butterfly valve" in loaded.documents["M-101"]
    assert loaded.sections["ADD-1/1"].kind == "addendum"
    assert "cases" not in loaded.documents


def test_source_version_changes_with_source_and_metadata_bytes(source_root):
    first = corpus.load_corpus(source_root).version
    spec = source_root / "specs" / "SPEC-08.md"
    spec.write_bytes(spec.read_bytes() + b"\n")
    second = corpus.load_corpus(source_root).version
    assert first != second
    cases = source_root / "cases.yaml"
    cases.write_bytes(cases.read_bytes() + b"\n# metadata comment\n")
    third = corpus.load_corpus(source_root).version
    assert second != third
    assert corpus.load_corpus(source_root).version == third


def test_version_frames_paths_and_contents_independently():
    assert corpus._source_version({"a": b"bc"}) != corpus._source_version({"ab": b"c"})
    assert corpus._source_version({"b": b"2", "a": b"1"}) == corpus._source_version({
        "a": b"1", "b": b"2",
    })


@pytest.mark.parametrize(("quote", "expected"), [
    ("The duty pump count is 2.", ["2"]),
    ("The tank volume is 96 m3.", ["96"]),
    ("The filling rate is 0.024 m3/s.", ["0.024"]),
    ("The platform area is 35 m2.", ["35"]),
    ("A stated quantity of 1,200 units.", ["1,200"]),
    ("A stated quantity of 1.2e3 units.", ["1.2e3"]),
])
def test_numeric_quotes_ignore_unit_exponents_and_sentence_punctuation(quote, expected):
    assert corpus.NUMBER.findall(quote) == expected


def test_search_returns_both_valve_sides_and_all_document_kinds(source_root):
    loaded = corpus.load_corpus(source_root)
    results = corpus.search_corpus(loaded, "butterfly valve discharge")
    assert {"SPEC-02/1.1", "M-101/1"} <= {result["id"] for result in results}
    assert results[0]["id"] == "M-101/1"
    assert all(result["score"] > 0 for result in results)
    assert results == corpus.search_corpus(loaded, "butterfly valve discharge")
    assert corpus.search_corpus(loaded, "revised finish")[0]["id"].startswith("ADD-")
    assert corpus.search_corpus(loaded, "?!") == []
    assert len(corpus.search_corpus(loaded, "contract", limit=1)) == 1
    with pytest.raises(ValueError, match="positive integer"):
        corpus.search_corpus(loaded, "valve", limit=True)


def test_section_and_note_counts_use_allowed_ranges(source_root):
    spec = source_root / "specs" / "SPEC-08.md"
    text, moved = spec.read_text().rsplit("## 1.7", 1)
    spec.write_text(text)
    other = source_root / "specs" / "SPEC-09.md"
    other.write_text(other.read_text() + "\n## 1.8" + moved)
    path = source_root / "drawings" / "G-001.yaml"
    drawing = yaml.safe_load(path.read_text())
    note = drawing["notes"].pop()
    write_yaml(path, drawing)
    path = source_root / "drawings" / "C-101.yaml"
    drawing = yaml.safe_load(path.read_text())
    drawing["notes"].append({**note, "number": 7})
    write_yaml(path, drawing)
    assert len(corpus.load_corpus(source_root).sections) == 221


@pytest.mark.parametrize(("kind", "field", "value", "message"), [
    ("conflicts", "spec_quote", "Invented requirement.", "exact quote"),
    ("conflicts", "spec", "../../outside", "existing section"),
    ("conflicts", "drawing", "SPEC-02/1.1", "drawing section"),
    ("conflicts", "resolution_quote", "Drawings govern.", "exact quote"),
    ("changes", "before", "Wrong base value.", "exact quote"),
    ("changes", "after", "Wrong amended value.", "exact quote"),
    ("gaps", "section", "SPEC-99/1.1", "existing section"),
    ("calculations", "tolerance", float("nan"), "finite number"),
    ("calculations", "operation", "power", "unsupported calculation"),
])
def test_malformed_cases_are_rejected(source_root, kind, field, value, message):
    path = source_root / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    cases[kind][0][field] = value
    write_yaml(path, cases)
    with pytest.raises(ValueError, match=message):
        corpus.load_corpus(source_root)


@pytest.mark.parametrize("value", [True, float("inf"), 999, "2"])
def test_numeric_values_must_match_single_source_number(source_root, value):
    path = source_root / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    cases["calculations"][0]["inputs"][0]["value"] = value
    write_yaml(path, cases)
    with pytest.raises(ValueError, match="finite number|exactly its input value"):
        corpus.load_corpus(source_root)


def test_numeric_quote_cannot_hide_extra_values(source_root):
    path = source_root / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    item = cases["calculations"][0]["inputs"][0]
    original = item["quote"]
    item["quote"] += " Repeat with 2 units."
    section = source_root / "specs" / "SPEC-19.md"
    section.write_text(section.read_text().replace(original, item["quote"], 1))
    write_yaml(path, cases)
    with pytest.raises(ValueError, match="exactly its input value"):
        corpus.load_corpus(source_root)


@pytest.mark.parametrize("surprise", ["missing", "extra", "symlink", "duplicate_yaml", "bad_id"])
def test_source_surprises_are_rejected(source_root, surprise):
    drawing = source_root / "drawings" / "G-001.yaml"
    if surprise == "missing":
        drawing.unlink()
    elif surprise == "extra":
        (source_root / "unexpected.md").write_text("Unexpected source")
    elif surprise == "symlink":
        drawing.unlink()
        drawing.symlink_to(source_root / "drawings" / "C-101.yaml")
    elif surprise == "duplicate_yaml":
        drawing.write_text(drawing.read_text() + "\nid: G-001\n")
    else:
        value = yaml.safe_load(drawing.read_text())
        value["id"] = "../outside"
        write_yaml(drawing, value)
    with pytest.raises(ValueError):
        corpus.load_corpus(source_root)


def test_duplicate_spec_section_is_rejected(source_root):
    path = source_root / "specs" / "SPEC-08.md"
    path.write_text(path.read_text() + "\n## 1.7 Duplicate section\n\nRepeated body.\n")
    with pytest.raises(ValueError, match="Duplicate section ID"):
        corpus.load_corpus(source_root)


def test_build_produces_byte_identical_real_pngs_and_manifest(source_root, tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    summary = corpus.build_corpus(first_dir, source_root)
    assert summary == corpus.build_corpus(second_dir, source_root)
    assert summary == {
        "corpus_version": corpus.load_corpus(source_root).version,
        "documents": 35, "specs": 20, "spec_sections": 140, "drawings": 12,
        "drawing_notes": 72, "addenda": 3, "addendum_items": 9,
        "conflicts": 6, "changes": 9, "gaps": 10, "calculations": 25,
    }
    expected = {path.relative_to(first_dir): path.read_bytes()
                for path in (first_dir / "corpus").rglob("*") if path.is_file()}
    actual = {path.relative_to(second_dir): path.read_bytes()
              for path in (second_dir / "corpus").rglob("*") if path.is_file()}
    assert expected == actual
    assert summary == corpus.build_corpus(first_dir, source_root)
    assert expected == {path: (first_dir / path).read_bytes() for path in expected}
    manifest = json.loads((first_dir / "corpus" / "manifest.json").read_text())
    for sheet in manifest["drawings"]:
        path = first_dir / "corpus" / sheet["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sheet["sha256"]
        with Image.open(path) as image:
            image.load()
            assert image.size == (1600, 1000)
            assert image.mode == "RGB"
            assert image.info == {}
    assert list((first_dir / "tmp").iterdir()) == []


def test_publication_uses_tmp_fsyncs_and_manifest_last(source_root, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    replacements, syncs = [], []
    replace, fsync = corpus.os.replace, corpus.os.fsync

    def record_sync(descriptor):
        fsync(descriptor)
        syncs.append(descriptor)

    def record_replace(source, destination):
        assert Path(source).parent == data_dir / "tmp"
        assert syncs
        assert Path(source).read_bytes()
        replacements.append(Path(destination))
        replace(source, destination)

    monkeypatch.setattr(corpus.os, "fsync", record_sync)
    monkeypatch.setattr(corpus.os, "replace", record_replace)
    corpus.build_corpus(data_dir, source_root)
    assert len(replacements) == 13
    assert replacements[-1] == data_dir / "corpus" / "manifest.json"
    assert all(path.suffix == ".png" for path in replacements[:-1])
    assert len(syncs) >= len(replacements) * 3


def test_render_overflow_fails_before_publishing(source_root, tmp_path):
    data_dir = tmp_path / "data"
    corpus.build_corpus(data_dir, source_root)
    previous = {path: path.read_bytes() for path in (data_dir / "corpus").rglob("*")
                if path.is_file()}
    path = source_root / "drawings" / "P-101.yaml"
    drawing = yaml.safe_load(path.read_text())
    drawing["notes"][-1]["text"] = "Coordinate the installation. " * 100
    write_yaml(path, drawing)
    with pytest.raises(ValueError, match="overflows"):
        corpus.build_corpus(data_dir, source_root)
    assert all(path.read_bytes() == content for path, content in previous.items())


def test_failed_atomic_replace_cleans_temporary_file(source_root, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    def fail_replace(source, destination):
        raise OSError("injected publication failure")

    monkeypatch.setattr(corpus.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        corpus.build_corpus(data_dir, source_root)
    assert not (data_dir / "corpus" / "manifest.json").exists()
    assert list((data_dir / "corpus" / "drawings").iterdir()) == []
    assert list((data_dir / "tmp").iterdir()) == []
