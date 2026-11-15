"""Compose native ChemDraw SVGs in the audited Figure 5 layout.

Spectrum values, colors, hypotheses and scores are retained. Arial labels
and optionally enlarged, stacked hypothesis views improve paper-width reading.
Selection metadata and interpretation notes belong in the accompanying caption.
The original case package is never modified.
"""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes as C
import ctypes.util
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ROOT = Path(__file__).resolve().parents[2]


def renderer(width: int, height: int):
    # Reuse only the self-contained native SVG renderer. Importing the frozen
    # server plotting module would also import the unrelated PyTorch stack.
    source = ROOT / "tools/analysis/plot_figure5_validation_cases.py"
    module = ast.parse(source.read_text())
    names = {"Rect", "GError", "render_native"}
    body = [n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(body) == len(names)
    namespace = {"C": C, "W": width, "H": height}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["render_native"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/figure5_validation_cases_20261112")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figure5_chemdraw_20261113")
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--stack-hypotheses", action="store_true",
                        help="enlarge proposal structures by stacking the two views per candidate")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to(ROOT / "outputs")
    molecules = out / "molecules"
    payload = json.loads((args.source_dir / "cases.json").read_text())
    root = ET.parse(args.source_dir / "figure5_validation_cases.svg").getroot()
    old_nodes = root.findall(f"{{{SVG}}}svg")
    keys = []
    for case in payload["cases"]:
        for candidate in case["candidates"]:
            key = f"{case['case']}_{candidate['role']}"
            keys.append(f"{key}_molecule")
            keys.extend(f"{key}_fragment_hypothesis_{i+1}" for i in range(len(candidate["displayed_hypotheses"])))
    assert len(keys) == len(old_nodes) == 12
    replacements = []
    views = {}
    for old, key in zip(old_nodes, keys):
        path = molecules / f"{key}.svg"
        new = ET.parse(path).getroot()
        assert not new.findall(f".//{{{SVG}}}image")
        for attr in ("x", "y", "width", "height"):
            new.set(attr, old.attrib[attr])
        for tag in ("title", "desc"):
            node = old.find(f"{{{SVG}}}{tag}")
            if node is not None:
                new.insert(0, copy.deepcopy(node))
        position = list(root).index(old)
        root.remove(old)
        root.insert(position, new)
        views[key] = new
        replacements.append({"asset": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for node in root.iter(f"{{{SVG}}}text"):
        if node.text == "GT only":
            node.text = "Ground truth only"
        elif node.text == "Pair already ranks GT first":
            node.text = "Pair already ranks ground truth first"
        elif node.text in ("Decoy only", "Neither"):
            node.set("x", str(float(node.attrib["x"])+40))
    for node in root.findall(f"{{{SVG}}}line"):
        if node.attrib.get("stroke-width") == "5" and float(node.attrib["x1"]) == 374:
            node.set("x1", str(float(node.attrib["x1"])+40))
            node.set("x2", str(float(node.attrib["x2"])+40))
    # Reserve the image for the two cases; provenance and interpretation
    # belong in the caption. Crop the vacated header/footer bands without
    # moving or rescaling any spectrum or molecular panel.
    for node in list(root):
        if node.tag == f"{{{SVG}}}text" and (
            node.text == "Official-validation explanatory cases"
            or (node.text or "").startswith(("Post hoc", "All highlighted sets", "Neither =", "Pair already ranks"))
        ):
            root.remove(node)
    root.set("viewBox", "0 48 1600 1188")
    root.set("height", f"{185*1188/1600}mm")
    background = root.find(f"{{{SVG}}}rect")
    if background is not None and background.attrib.get("fill") == "white":
        background.set("y", "48")
        background.set("height", "1188")
    for node in root.iter(f"{{{SVG}}}text"):
        if "DejaVu Sans" in node.attrib.get("font-family", ""):
            node.set("font-family", args.font_family)
            # At 31 pc paper width, 26 SVG units are approximately 6 pt.
            node.set("font-size", str(max(26, float(node.attrib["font-size"]))))
            if node.attrib.get("font-weight") == "bold" and (node.text or "").startswith(("A  ", "B  ")):
                node.set("font-size", "36")
            # Arial lacks some Unicode subscript digits. Use ordinary Arial
            # digits with a subscript baseline instead of a fallback typeface.
            if re.search("[₀-₉]", node.text or ""):
                parts = re.split("([₀-₉]+)", node.text)
                node.text = parts[0]
                for index in range(1, len(parts), 2):
                    sub = ET.SubElement(node, f"{{{SVG}}}tspan",
                                        {"baseline-shift": "sub", "font-size": "75%"})
                    sub.text = parts[index].translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
                    sub.tail = parts[index+1]
    if args.stack_hypotheses:
        extra_height = 345
        # Keep each case's spectrum geometry fixed; translate the entire second
        # case, including its rotated y-axis label, to make room for larger views.
        groups = [ET.Element(f"{{{SVG}}}g"),
                  ET.Element(f"{{{SVG}}}g", {"transform": f"translate(0 {extra_height})"})]
        for node in list(root):
            if node is background:
                continue
            y = float(node.attrib.get("y", node.attrib.get("y1", "0")))
            root.remove(node)
            groups[int(y >= 640)].append(node)
        for group in groups:
            root.append(group)
            for node in group.findall(f"{{{SVG}}}rect"):
                if node.attrib.get("height") == "575":
                    node.set("height", str(575 + extra_height))
        for case_index, case in enumerate(payload["cases"]):
            y = 64 + 590 * case_index
            for candidate in case["candidates"]:
                key = f"{case['case']}_{candidate['role']}"
                x = float(views[f"{key}_molecule"].attrib["x"])
                for j, _ in enumerate(candidate["displayed_hypotheses"]):
                    view = views[f"{key}_fragment_hypothesis_{j+1}"]
                    view.attrib.update(x=str(x), y=str(y + 362 + 274*j),
                                       width="430", height="180")
                    for line, original_y in enumerate((497, 527, 554)):
                        matches = [n for n in groups[case_index].findall(f"{{{SVG}}}text")
                                   if float(n.attrib["x"]) == x + 215*j + 105
                                   and float(n.attrib["y"]) == y + original_y]
                        assert len(matches) == 1, (key, j, original_y)
                        matches[0].set("x", str(x + 215))
                        matches[0].set("y", str(y + 566 + 274*j + 28*line))
        height = 1188 + 2*extra_height
        root.set("viewBox", f"0 48 1600 {height}")
        root.set("height", f"{185*height/1600}mm")
        if background is not None:
            background.set("height", str(height))
    target = out / "figure5_chemdraw.svg"
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    dimensions = list(map(int, root.attrib["viewBox"].split()[2:]))
    renderer(*dimensions)(target, target.with_suffix(".pdf"), target.with_suffix(".png"))
    selection = json.loads((args.source_dir / "selection_manifest.json").read_text())
    counts = selection["eligible_counts"]
    caption = f"""Figure 5. Post-hoc official-validation examples, sampled uniformly after
query-ID sorting with seed {selection['rules']['seed']} ({counts['A']} rescue-eligible and
{counts['B']} ambiguity-eligible queries). Both use the supported top-20 branch,
[M+H]+, and structures with at most 30 heavy atoms. (A) Ground truth moves
from Stage 1 rank 2 to final rank 1; the original winner moves from 1 to 4.
The pair reranker already ranks ground truth first, so this rescue does not
isolate a coverage effect. (B) A same-formula decoy ranks first and ground
truth third despite identical nonempty matched-peak sets; this failure is
specific to the example. Peak colors distinguish ground-truth-only,
decoy-only, both and neither. Coverage uses all 20 candidates with the frozen
median correction. Up to two hypotheses per molecule follow decreasing
matched-peak intensity. Highlights mark retained proposal atom sets; Direct
and NL* denote direct-ion and neutral-loss mass tests, and ΔH is the hydrogen
adjustment. These are mass-feasible hypotheses, not confirmed fragments,
losses or mechanisms. Neither means no match in the frozen proposal family.
The cases illustrate system behavior rather than population-level performance.
"""
    (out / "caption.md").write_text(caption)

    # A larger four-structure sheet makes the bond/highlight style easy to review.
    sheet = ET.Element(f"{{{SVG}}}svg", {"viewBox": "0 0 1000 680", "width": "185mm", "height": "125.8mm"})
    ET.SubElement(sheet, f"{{{SVG}}}rect", {"width": "1000", "height": "680", "fill": "white"})
    for index, key in enumerate(k for k in keys if k.endswith("_molecule")):
        x, y = (index % 2) * 500, (index // 2) * 340
        title = key.replace("_molecule", "").replace("_", " ")
        ET.SubElement(sheet, f"{{{SVG}}}text", {"x": str(x+20), "y": str(y+30), "font-family": args.font_family, "font-size": "22", "fill": "#333333"}).text = title
        mol = ET.parse(molecules / f"{key}.svg").getroot()
        mol.attrib.update({"x": str(x+20), "y": str(y+55), "width": "460", "height": "260"})
        sheet.append(mol)
    sheet_path = out / "chemdraw_structures.svg"
    ET.ElementTree(sheet).write(sheet_path, encoding="utf-8", xml_declaration=True)
    renderer(1000, 680)(sheet_path, sheet_path.with_suffix(".pdf"), sheet_path.with_suffix(".png"))
    report = {
        "source_cases_sha256": hashlib.sha256((args.source_dir / "cases.json").read_bytes()).hexdigest(),
        "source_figure_sha256": hashlib.sha256((args.source_dir / "figure5_validation_cases.svg").read_bytes()).hexdigest(),
        "scope": "Native ChemDraw structures and Arial labels; optional stacked proposal views; spectrum values, masks, hypotheses, scores and ranks unchanged",
        "font_family": args.font_family,
        "stacked_hypotheses": args.stack_hypotheses,
        "caption_sha256": hashlib.sha256((out / "caption.md").read_bytes()).hexdigest(),
        "replaced_structure_views": replacements,
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for stem in (target, sheet_path)
                    for p in (stem, stem.with_suffix(".pdf"), stem.with_suffix(".png"))},
    }
    (out / "composition_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Created", target.with_suffix(".pdf"))
    print("Structure comparison sheet:", sheet_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
