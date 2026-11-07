import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import PIL
import yaml
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "corpus" / "v1"
SPEC_IDS = tuple(f"SPEC-{number:02d}" for number in range(1, 21))
DRAWING_IDS = (
    "G-001", "C-101", "S-101", "M-101", "M-102", "M-501",
    "E-101", "E-601", "I-101", "I-501", "H-101", "P-101",
)
ADDENDUM_IDS = ("ADD-1", "ADD-2", "ADD-3")
SHEET_SIZE = (1600, 1000)
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\d|\.\d)"
)


@dataclass(frozen=True)
class Section:
    id: str
    document_id: str
    kind: str
    title: str
    text: str


@dataclass(frozen=True)
class Drawing:
    id: str
    title: str
    notes: tuple[Section, ...]


@dataclass(frozen=True)
class Corpus:
    version: str
    sections: dict[str, Section]
    drawings: dict[str, Drawing]
    cases: dict
    documents: dict[str, str]


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    pairs = loader.construct_pairs(node, deep=deep)
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError(f"YAML mapping keys must be unique strings: {key!r}")
        result[key] = value
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _read_yaml(text: str, name: str):
    try:
        return yaml.load(text, Loader=_UniqueLoader)
    except (yaml.YAMLError, ValueError) as error:
        raise ValueError(f"Invalid YAML in {name}: {error}") from error


def _record(value, fields: set[str], name: str) -> dict:
    if type(value) is not dict or value.keys() != fields:
        raise ValueError(f"{name} must contain exactly {', '.join(sorted(fields))}")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _number(value, name: str) -> Decimal:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return number


def _source_files(source_root: Path) -> dict[str, bytes]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"Corpus source root must be a real directory: {source_root}")
    expected = {
        *(f"specs/{document_id}.md" for document_id in SPEC_IDS),
        *(f"drawings/{document_id}.yaml" for document_id in DRAWING_IDS),
        *(f"addenda/{document_id}.md" for document_id in ADDENDUM_IDS),
        "cases.yaml",
    }
    files = {}
    for directory, names, filenames in os.walk(source_root, followlinks=False):
        for name in sorted(names + filenames):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"Corpus source must not contain links or special files: {path}")
            relative = path.relative_to(source_root).as_posix()
            if stat.S_ISDIR(mode):
                if relative not in {"specs", "drawings", "addenda"}:
                    raise ValueError(f"Unexpected corpus source directory: {relative}")
            elif relative not in expected:
                raise ValueError(f"Unexpected corpus source file: {relative}")
            else:
                files[relative] = path.read_bytes()
    if files.keys() != expected:
        missing = ", ".join(sorted(expected - files.keys()))
        raise ValueError(f"Missing corpus source files: {missing}")
    return dict(sorted(files.items()))


def _source_version(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256(b"hopper-corpus-v1\0")
    for path, content in sorted(files.items()):
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _markdown_sections(document_id: str, kind: str, text: str) -> list[Section]:
    number = r"\d+\.\d+" if kind == "spec" else r"\d+"
    heading = re.compile(rf"^## ({number}) ([^\n]+)$", re.MULTILINE)
    matches = list(heading.finditer(text))
    if not matches or len(matches) != len(re.findall(r"^##(?:\s|$)", text, re.MULTILINE)):
        raise ValueError(f"{document_id} needs numbered, nonempty section headings")
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_id = f"{document_id}/{match[1]}"
        body = _text(text[match.end():end].strip(), section_id)
        sections.append(Section(section_id, document_id, kind, match[2].strip(), body))
    return sections


def _drawing(document_id: str, text: str) -> Drawing:
    source = _record(_read_yaml(text, document_id), {"id", "title", "notes"}, document_id)
    if source["id"] != document_id:
        raise ValueError(f"Drawing ID must match filename: {document_id}")
    title = _text(source["title"], f"{document_id} title")
    notes = source["notes"]
    if type(notes) is not list or not 5 <= len(notes) <= 10:
        raise ValueError(f"{document_id} must have 5 to 10 notes")
    sections = []
    for note in notes:
        _record(note, {"number", "text"}, f"{document_id} note")
        if type(note["number"]) is not int or note["number"] <= 0:
            raise ValueError(f"{document_id} note numbers must be positive integers")
        section_id = f"{document_id}/{note['number']}"
        sections.append(Section(
            section_id, document_id, "drawing", f"{title}, note {note['number']}",
            _text(note["text"], section_id),
        ))
    if len({section.id for section in sections}) != len(sections):
        raise ValueError(f"Duplicate note number in {document_id}")
    return Drawing(document_id, title, tuple(sections))


def _citation(corpus: Corpus, value, kind: str, name: str) -> Section:
    if not isinstance(value, str) or value not in corpus.sections:
        raise ValueError(f"{name} must reference an existing section")
    section = corpus.sections[value]
    if section.kind != kind:
        raise ValueError(f"{name} must reference a {kind} section")
    return section


def _quote(section: Section, value, name: str) -> str:
    quote = _text(value, name)
    if quote not in section.text:
        raise ValueError(f"{name} is not an exact quote from {section.id}")
    return quote


def _case_list(cases: dict, name: str, fields: set[str], count: int | None) -> list[dict]:
    items = cases[name]
    if type(items) is not list or (count is not None and len(items) != count):
        raise ValueError(f"{name} must contain {count if count is not None else 'a list of'} cases")
    identifiers = set()
    for item in items:
        _record(item, fields, f"{name} case")
        case_id = _text(item["id"], f"{name} case ID")
        _text(item["topic"], f"{case_id} topic")
        if case_id in identifiers:
            raise ValueError(f"Duplicate {name} case ID: {case_id}")
        identifiers.add(case_id)
    return items


def _validate_cases(corpus: Corpus) -> None:
    cases = _record(
        corpus.cases, {"precedence", "conflicts", "changes", "gaps", "calculations"}, "cases.yaml"
    )
    precedence = _citation(corpus, cases["precedence"], "spec", "precedence")
    if precedence.document_id != "SPEC-01":
        raise ValueError("The precedence section must belong to SPEC-01")
    conflicts = _case_list(cases, "conflicts", {
        "id", "topic", "spec", "drawing", "spec_quote", "drawing_quote", "resolution_quote",
    }, 6)
    for case in conflicts:
        spec = _citation(corpus, case["spec"], "spec", f"{case['id']} spec")
        drawing = _citation(corpus, case["drawing"], "drawing", f"{case['id']} drawing")
        _quote(spec, case["spec_quote"], f"{case['id']} spec_quote")
        _quote(drawing, case["drawing_quote"], f"{case['id']} drawing_quote")
        _quote(spec, case["resolution_quote"], f"{case['id']} resolution_quote")
    changes = _case_list(cases, "changes", {"id", "section", "topic", "before", "after"}, None)
    for case in changes:
        addendum = _citation(corpus, case["id"], "addendum", f"{case['id']} addendum")
        section = _citation(corpus, case["section"], "spec", f"{case['id']} section")
        _quote(section, case["before"], f"{case['id']} before")
        _quote(addendum, case["after"], f"{case['id']} after")
    changed_sections = {case["section"] for case in changes}
    if len(changed_sections) < 8:
        raise ValueError("Addenda must change at least 8 distinct spec sections")
    if changed_sections & {case["spec"] for case in conflicts}:
        raise ValueError("Addenda must not modify a base conflict section")
    for case in _case_list(cases, "gaps", {"id", "section", "topic", "quote"}, 10):
        section = corpus.sections.get(case["section"]) if isinstance(case["section"], str) else None
        if section is None:
            raise ValueError(f"{case['id']} gap must reference an existing section")
        _quote(section, case["quote"], f"{case['id']} quote")
    for case in _case_list(cases, "calculations", {
        "id", "topic", "operation", "inputs", "unit", "tolerance",
    }, 25):
        _text(case["unit"], f"{case['id']} unit")
        if not isinstance(case["operation"], str) or case["operation"] not in {
            "add", "subtract", "multiply", "divide",
        }:
            raise ValueError(f"{case['id']} has an unsupported calculation operation")
        if _number(case["tolerance"], f"{case['id']} tolerance") < 0:
            raise ValueError(f"{case['id']} tolerance must be nonnegative")
        if type(case["inputs"]) is not list or len(case["inputs"]) != 2:
            raise ValueError(f"{case['id']} needs exactly two calculation inputs")
        values = []
        for item in case["inputs"]:
            _record(item, {"name", "citation", "quote", "value"}, f"{case['id']} input")
            _text(item["name"], f"{case['id']} input name")
            citation = item["citation"]
            if not isinstance(citation, str) or citation not in corpus.sections:
                raise ValueError(f"{case['id']} input must reference an existing section")
            quote = _quote(corpus.sections[citation], item["quote"], f"{case['id']} input quote")
            numbers = NUMBER.findall(quote)
            value = _number(item["value"], f"{case['id']} input value")
            if len(numbers) != 1 or Decimal(numbers[0].replace(",", "")) != value:
                raise ValueError(f"{case['id']} numeric quote must contain exactly its input value")
            values.append(value)
        if case["operation"] == "divide" and values[1] == 0:
            raise ValueError(f"{case['id']} cannot divide by zero")


def load_corpus(source_root: Path = DEFAULT_SOURCE_ROOT) -> Corpus:
    files = _source_files(Path(source_root))
    try:
        sources = {path: content.decode("utf-8") for path, content in files.items()}
    except UnicodeDecodeError as error:
        raise ValueError("Corpus source files must contain UTF-8 text") from error
    sections = {}
    documents = {}
    for kind, directory, identifiers in (
        ("spec", "specs", SPEC_IDS), ("addendum", "addenda", ADDENDUM_IDS),
    ):
        for document_id in identifiers:
            text = sources[f"{directory}/{document_id}.md"]
            documents[document_id] = text
            for section in _markdown_sections(document_id, kind, text):
                if section.id in sections:
                    raise ValueError(f"Duplicate section ID: {section.id}")
                sections[section.id] = section
    drawings = {}
    for document_id in DRAWING_IDS:
        drawing = _drawing(document_id, sources[f"drawings/{document_id}.yaml"])
        drawings[document_id] = drawing
        documents[document_id] = f"# {document_id} {drawing.title}\n\n" + "\n\n".join(
            f"## {section.id.rsplit('/', 1)[1]}\n{section.text}" for section in drawing.notes
        )
        sections.update((section.id, section) for section in drawing.notes)
    counts = Counter(section.kind for section in sections.values())
    if not 120 <= counts["spec"] <= 180:
        raise ValueError("Corpus must have 120 to 180 spec sections")
    if not 60 <= counts["drawing"] <= 120:
        raise ValueError("Corpus must have 60 to 120 drawing notes")
    corpus = Corpus(
        _source_version(files), sections, drawings,
        _read_yaml(sources["cases.yaml"], "cases.yaml"), documents,
    )
    _validate_cases(corpus)
    return corpus


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def search_corpus(corpus: Corpus, query: str, limit: int = 10) -> list[dict]:
    if not isinstance(query, str):
        raise ValueError("Search query must be text")
    if type(limit) is not int or limit < 1:
        raise ValueError("Search limit must be a positive integer")
    terms = set(_tokens(query))
    if not terms:
        return []
    records = [
        (section, Counter(_tokens(f"{section.title} {section.text}")))
        for section in corpus.sections.values()
    ]
    if not records:
        return []
    average_length = sum(sum(tokens.values()) for _, tokens in records) / len(records)
    frequencies = Counter(term for _, tokens in records for term in terms if term in tokens)
    results = []
    for section, tokens in records:
        length = sum(tokens.values())
        score = 0.0
        for term in sorted(terms & tokens.keys()):
            inverse = math.log(1 + (len(records) - frequencies[term] + 0.5)
                               / (frequencies[term] + 0.5))
            frequency = tokens[term]
            score += inverse * (frequency * 2.5) / (
                frequency + 1.5 * (0.25 + 0.75 * length / average_length)
            )
        if score > 0:
            results.append({
                "id": section.id, "title": section.title, "text": section.text, "score": score,
            })
    return sorted(results, key=lambda result: (-result["score"], result["id"]))[:limit]


def _wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = ""
        for word in words:
            if draw.textlength(word, font=font) > width:
                raise ValueError("Drawing text contains a word wider than its note panel")
            candidate = f"{line} {word}".strip()
            if line and draw.textlength(candidate, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def _render_drawing(drawing: Drawing) -> bytes:
    if PIL.__version__ != "12.0.0":
        raise ValueError("Corpus rendering requires Pillow 12.0.0 for reproducible PNGs")
    image = Image.new("RGB", SHEET_SIZE, "#f7f7f2")
    draw = ImageDraw.Draw(image)
    ink, blue, pale = "#172c3c", "#244e68", "#e8eef0"
    fonts = {size: ImageFont.load_default(size=size) for size in (17, 20, 24, 30, 40)}
    draw.rectangle((36, 30, 1564, 970), outline=ink, width=3)
    draw.rectangle((36, 30, 1564, 142), fill=ink)
    draw.text((65, 48), "BAYVIEW PUMP STATION UPGRADE", font=fonts[20], fill="#c9dce5")
    title_lines = _wrapped_lines(draw, drawing.title, fonts[30], 1235)
    if len(title_lines) != 1:
        raise ValueError(f"{drawing.id} title overflows the drawing title band")
    draw.text((65, 83), drawing.title, font=fonts[30], fill="white")
    draw.text((1370, 74), drawing.id, font=fonts[30], fill="white")
    rows = math.ceil(len(drawing.notes) / 2)
    row_height = 654 // rows
    for index, section in enumerate(drawing.notes):
        column, row = divmod(index, rows)
        left, top = 64 + column * 750, 168 + row * row_height
        right, bottom = left + 722, top + row_height - 16
        draw.rectangle((left, top, right, bottom), fill="white", outline="#b5c5cc", width=1)
        draw.rectangle((left, top, left + 81, top + 31), fill=pale)
        note_number = section.id.rsplit("/", 1)[1]
        draw.text((left + 12, top + 6), f"NOTE {note_number}", font=fonts[17], fill=blue)
        lines = _wrapped_lines(draw, section.text, fonts[24], 690)
        line_height = 29
        if top + 43 + len(lines) * line_height > bottom - 8:
            raise ValueError(f"{section.id} text overflows its drawing note panel")
        for line_index, line in enumerate(lines):
            draw.text((left + 16, top + 39 + line_index * line_height), line,
                      font=fonts[24], fill=ink)
    draw.line((36, 842, 1564, 842), fill=ink, width=2)
    draw.line((1080, 842, 1080, 970), fill=ink, width=2)
    draw.line((1320, 842, 1320, 970), fill=ink, width=2)
    draw.text((64, 858), "CONTRACT DRAWING NOTES", font=fonts[17], fill=blue)
    draw.text((64, 890), "Read with specifications and all issued addenda.",
              font=fonts[24], fill=ink)
    draw.text((64, 931), "FICTIONAL PROJECT  |  REPRODUCIBLE REFERENCE SHEET",
              font=fonts[17], fill=blue)
    draw.text((1100, 858), "ISSUE", font=fonts[17], fill=blue)
    draw.text((1100, 891), "CORPUS V1", font=fonts[24], fill=ink)
    draw.text((1340, 858), "SHEET", font=fonts[17], fill=blue)
    draw.text((1340, 891), drawing.id, font=fonts[40], fill=ink)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Corpus output directory must not be a symbolic link: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Corpus output directory is not a directory: {path}")
        return
    _ensure_directory(path.parent)
    path.mkdir(exist_ok=True)
    _fsync_directory(path.parent)


def _publish(path: Path, content: bytes, temporary_dir: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="corpus-", suffix=".tmp", dir=temporary_dir)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        _fsync_directory(temporary_dir)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_corpus(data_dir: Path, source_root: Path = DEFAULT_SOURCE_ROOT) -> dict:
    corpus = load_corpus(source_root)
    rendered = {drawing.id: _render_drawing(drawing) for drawing in corpus.drawings.values()}
    kinds = Counter(section.kind for section in corpus.sections.values())
    counts = {
        "documents": len(corpus.documents), "specs": len(SPEC_IDS),
        "spec_sections": kinds["spec"], "drawings": len(corpus.drawings),
        "drawing_notes": kinds["drawing"], "addenda": len(ADDENDUM_IDS),
        "addendum_items": kinds["addendum"],
        **{name: len(corpus.cases[name])
           for name in ("conflicts", "changes", "gaps", "calculations")},
    }
    manifest = {
        "corpus_version": corpus.version,
        "counts": counts,
        "drawings": [
            {"id": identifier, "path": f"drawings/{identifier}.png",
             "sha256": hashlib.sha256(content).hexdigest()}
            for identifier, content in sorted(rendered.items())
        ],
    }
    data_dir = Path(data_dir)
    temporary_dir, drawing_dir = data_dir / "tmp", data_dir / "corpus" / "drawings"
    _ensure_directory(temporary_dir)
    _ensure_directory(drawing_dir)
    for identifier, content in rendered.items():
        _publish(drawing_dir / f"{identifier}.png", content, temporary_dir)
    _publish(
        data_dir / "corpus" / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"), temporary_dir,
    )
    return {"corpus_version": corpus.version, **counts}
