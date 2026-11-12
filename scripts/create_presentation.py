"""Create a self-contained research presentation from local pilot results."""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "escalation_probes_pilot.pptx"

EMU_PER_INCH = 914400
SLIDE_WIDTH = 13.333 * EMU_PER_INCH
SLIDE_HEIGHT = 7.5 * EMU_PER_INCH


def emu(value: float) -> int:
    """Convert inches to English Metric Units."""
    return int(value * EMU_PER_INCH)


def read_summary() -> list[dict[str, str]]:
    """Read selected-layer summary rows for the three models."""
    rows = []
    for model in ("qwen2.5-3b", "olmo-2-1b", "smollm2-1.7b"):
        with (TABLES / f"summary_{model}.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row["probe_kind"] == "mlp" and row["layer_role"] == "selected")
    return rows


def read_token_accuracy() -> dict[str, float]:
    """Read selected-layer MLP test token accuracy for each model."""
    accuracies = {}
    for model in ("qwen2.5-3b", "olmo-2-1b", "smollm2-1.7b"):
        with (TABLES / f"probe_single_turn_{model}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["probe_kind"] == "mlp" and row["layer_role"] == "selected" and row["split"] == "test":
                    accuracies[model] = float(row["token_accuracy"])
    return accuracies


def text_box(text: str, x: float, y: float, width: float, height: float, size: int = 20, bold: bool = False) -> str:
    """Return a PowerPoint text-box shape."""
    runs = []
    for line in text.splitlines() or [""]:
        runs.append(
            f"<a:p><a:r><a:rPr lang=\"en-US\" sz=\"{size * 100}\" b=\"{str(bold).lower()}\"/>"
            f"<a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang=\"en-US\"/></a:p>"
        )
    return (
        f"<p:sp><p:nvSpPr><p:cNvPr id=\"{next_id()}\" name=\"TextBox\"/>"
        f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"{emu(x)}\" y=\"{emu(y)}\"/>"
        f"<a:ext cx=\"{emu(width)}\" cy=\"{emu(height)}\"/></a:xfrm></p:spPr>"
        f"<p:txBody><a:bodyPr/><a:lstStyle/>{''.join(runs)}</p:txBody></p:sp>"
    )


_shape_id = 1


def next_id() -> int:
    """Return a unique shape identifier for the current slide."""
    global _shape_id
    _shape_id += 1
    return _shape_id


def image_shape(image_name: str, rel_id: str) -> str:
    """Return a centered image shape."""
    return (
        f"<p:pic><p:nvPicPr><p:cNvPr id=\"{next_id()}\" name=\"{escape(image_name)}\"/>"
        f"<p:cNvPicPr/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed=\"{rel_id}\"/>"
        f"<a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm>"
        f"<a:off x=\"{emu(0.45)}\" y=\"{emu(1.05)}\"/><a:ext cx=\"{emu(12.4)}\" cy=\"{emu(5.95)}\"/>"
        f"</a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr></p:pic>"
    )


def slide_xml(title: str, body: str = "", image: str | None = None, rel_id: str | None = None) -> str:
    """Build one slide XML document."""
    global _shape_id
    _shape_id = 1
    shapes = [
        "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
    ]
    shapes.append(text_box(title, 0.45, 0.25, 12.4, 0.55, size=26, bold=True))
    if image and rel_id:
        shapes.append(image_shape(image, rel_id))
    if body:
        shapes.append(text_box(body, 0.7, 1.15 if not image else 6.8, 12.0, 5.8 if not image else 0.45, size=16))
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
        "<p:cSld><p:spTree>" + "".join(shapes) + "</p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


def rels(targets: list[tuple[str, str, str]]) -> str:
    """Build a relationships document."""
    items = [
        f"<Relationship Id=\"{rid}\" Type=\"{kind}\" Target=\"{target}\"/>"
        for rid, kind, target in targets
    ]
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        + "".join(items)
        + "</Relationships>"
    )


def content_types(slide_count: int, image_count: int) -> str:
    """Build the package content-types document."""
    overrides = [
        "<Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>",
        "<Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>",
        "<Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>",
        "<Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>",
    ]
    overrides.extend(
        f"<Override PartName=\"/ppt/slides/slide{i}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"
        for i in range(1, slide_count + 1)
    )
    defaults = [
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
        "<Default Extension=\"png\" ContentType=\"image/png\"/>",
    ]
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        + "".join(defaults + overrides)
        + "</Types>"
    )


def main() -> None:
    """Create the 14-slide pilot presentation."""
    rows = read_summary()
    token_accuracy = read_token_accuracy()
    model_names = {
        "qwen2.5-3b": "Qwen2.5-3B-Instruct",
        "olmo-2-1b": "OLMo-2-1B-Instruct",
        "smollm2-1.7b": "SmolLM2-1.7B-Instruct",
    }
    result_lines = ["Model | MLP test token accuracy | Probe accuracy | Judge accuracy"]
    for row in rows:
        result_lines.append(
            f"{model_names[row['model']]} | {token_accuracy[row['model']]:.1%} | "
            f"{float(row['probe_accuracy']):.1%} | {float(row['judge_accuracy']):.1%}"
        )

    slides = [
        ("Testing Representation Probes for Multi-Turn Safety Failures", "Three-model pilot replication\\nQwen2.5 · OLMo-2 · SmolLM2\\nAI safety / interpretability study", None),
        ("Motivation and research questions", "• Do single-turn representation signals transfer to multi-turn escalation?\\n• Does an MLP probe detect harmful multi-turn states?\\n• Does internal monitoring add value over a text-only judge?\\n\\nTarget method: PCA reference space plus MLP probing.", None),
        ("Study design", "200 single-turn records: 100 benign + 100 harmful\\n8 multi-turn records: 4 escalations + 4 matched controls\\nTwo categories: phishing/fraud and non-operational cyber intrusion\\nGrouped split: 140 train / 30 validation / 30 test", None),
        ("Models and pipeline", "Qwen2.5-3B-Instruct — primary\\nOLMo-2-1B-Instruct — secondary\\nSmolLM2-1.7B-Instruct — secondary substitution for gated/unavailable Moonlight\\n\\nCache → select layer → PCA → MLP/linear probe → judge → comparison", None),
        ("What the probes measure", "Probe training uses only single-turn activation tokens.\\nLayer selection uses validation accuracy only.\\nPCA is fitted on training tokens, then multi-turn activations are projected into that fixed space.\\nTeacher forcing keeps the final response tokens fixed across context lengths.", None),
        ("Held-out single-turn performance", "\\n".join(result_lines), None),
        ("PCA reference-space variance", "", "all_models_figure_02_pca_variance.png"),
        ("PCA geometry and multi-turn readouts", "", "all_models_figure_07_pca_scatter.png"),
        ("Context-length effects", "", "all_models_figure_03_context_length.png"),
        ("Per-turn and category trajectories", "", "all_models_figure_08_category_trajectories.png"),
        ("Condition and layer sensitivity", "", "all_models_figure_05_condition_layer.png"),
        ("Internal probe versus text-only judge", "", "all_models_figure_06_probe_vs_judge.png"),
        ("Main findings", "• All three models learn a strong single-turn benign/harmful separation.\\n• Qwen has the strongest selected-layer probe conversation accuracy in this pilot: 75%.\\n• The judge scores 37.5% on the eight-conversation comparison set.\\n• Probe performance varies across model family, condition, and layer.\\n• The evidence supports representation monitoring feasibility, not reliable escalation detection.", None),
        ("Limitations, next steps, and conclusion", "Limitations: four escalation conversations, four controls, prewritten rather than adaptive turns, and a local supplementary judge.\\n\\nOptional next step: rerun ambiguous cases or add seeds before making strong claims.\\n\\nConclusion: the pipeline is complete and reproducible for this pilot; broader claims require a larger, adaptive evaluation.", None),
    ]

    image_slides = {index + 1: image for index, (_, _, image) in enumerate(slides) if image}
    image_names = sorted(set(image_slides.values()))
    image_ids = {name: f"rId{index + 2}" for index, name in enumerate(image_names)}

    presentation = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<p:presentation xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
        "<p:sldMasterIdLst><p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/></p:sldMasterIdLst>"
        "<p:sldIdLst>"
        + "".join(f"<p:sldId id=\"{255 + index}\" r:id=\"rId{index + 3}\"/>" for index in range(len(slides)))
        + "</p:sldIdLst><p:sldSz cx=\"12192000\" cy=\"6858000\" type=\"screen16x9\"/>"
        "<p:notesSz cx=\"6858000\" cy=\"9144000\"/><p:defaultTextStyle/></p:presentation>"
    )

    master = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<p:sldMaster xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
        "<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        "</p:spTree></p:cSld><p:clrMap accent1=\"accent1\" accent2=\"accent2\" bg1=\"lt1\" bg2=\"lt2\" tx1=\"dk1\" tx2=\"dk2\" hlink=\"hlink\" folHlink=\"folHlink\"/></p:sldMaster>"
    )
    layout = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<p:sldLayout xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" type=\"blank\">"
        "<p:cSld name=\"Blank\"><p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>"
        "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    )
    theme = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><a:theme xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" name=\"Default\"><a:themeElements><a:clrScheme name=\"Default\"><a:dk1><a:srgbClr val=\"000000\"/></a:dk1><a:lt1><a:srgbClr val=\"FFFFFF\"/></a:lt1><a:dk2><a:srgbClr val=\"1F2937\"/></a:dk2><a:lt2><a:srgbClr val=\"F9FAFB\"/></a:lt2><a:accent1><a:srgbClr val=\"2563EB\"/></a:accent1><a:accent2><a:srgbClr val=\"059669\"/></a:accent2><a:hlink><a:srgbClr val=\"0563C1\"/></a:hlink><a:folHlink><a:srgbClr val=\"954F72\"/></a:folHlink></a:clrScheme><a:fontScheme name=\"Default\"><a:majorFont><a:latin typeface=\"Aptos Display\"/></a:majorFont><a:minorFont><a:latin typeface=\"Aptos\"/></a:minorFont></a:fontScheme><a:fmtScheme name=\"Default\"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>"

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types(len(slides), len(image_names)))
        archive.writestr("_rels/.rels", rels([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml")]))
        archive.writestr("ppt/presentation.xml", presentation)
        presentation_targets = [
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
        ]
        presentation_targets.extend(
            (f"rId{index + 3}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{index + 1}.xml")
            for index in range(len(slides))
        )
        archive.writestr("ppt/_rels/presentation.xml.rels", rels(presentation_targets))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", master)
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml")]))
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels([("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml")]))
        archive.writestr("ppt/theme/theme1.xml", theme)
        for index, (title, body, image) in enumerate(slides, start=1):
            image_rel = image_ids.get(image)
            archive.writestr(f"ppt/slides/slide{index}.xml", slide_xml(title, body, image, image_rel))
            targets = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
            if image:
                targets.append((image_rel, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{image}"))
            archive.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", rels(targets))
        for image in image_names:
            archive.write(FIGURES / image, f"ppt/media/{image}")

    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
