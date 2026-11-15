"""Create ChemDraw structure variants from the frozen Figure 5 atom masks.

Run with the Windows Python 3.10/ChemScript installation documented in
docs/maintenance/LOCAL_WSL_CHEMDRAW_SETUP_20261111.md. This changes depiction
only; it never selects cases or recomputes spectra, matches or model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-size", type=float, default=10,
                        help="ChemDraw atom-label size before final figure scaling")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    install = Path(r"C:\Program Files\RevvitySignalsSoftware\ChemDrawApplications_x64")
    assert sys.version_info[:2] == (3, 10) and install.is_dir()
    dll_directory = os.add_dll_directory(str(install))
    sys.path.insert(0, str(install))
    import ChemScript as CS

    CS.Env.SetThrowExceptions(True)
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for case in payload["cases"]:
        for candidate in case["candidates"]:
            key = f"{case['case']}_{candidate['role']}"
            mol = CS.StructureData.LoadData(candidate["molblock"], "chemical/x-mdl-molfile")
            atoms = list(mol.Atoms)
            atom_ids = [a.ID for a in atoms]
            assert [a.Element for a in atoms] == [a["symbol"] for a in candidate["atoms"]]
            bonds = sorted(tuple(sorted((b.Atom1.ID, b.Atom2.ID))) for b in mol.Bonds)
            mol.CleanupStructure(True)
            assert [a.ID for a in mol.Atoms] == atom_ids
            assert sorted(tuple(sorted((b.Atom1.ID, b.Atom2.ID))) for b in mol.Bonds) == bonds
            mol.WriteFile(str(out / f"{key}_cleaned.mol"), "chemical/x-mdl-molfile")
            data = mol.WriteData("text/xml")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            xml = ET.fromstring(data)
            nodes = list(xml.iter("n"))
            assert len(nodes) == len(atoms)
            id_map = {i: str(a.CdxIdAsWrittenToFile) for i, a in enumerate(mol.Atoms)}
            assert set(id_map.values()) == {n.attrib["id"] for n in nodes}
            # Rigid rotation only: fit the long axis horizontally and place
            # the highlighted region to the right. Atom identity is unchanged.
            coords = [tuple(map(float, n.attrib["p"].split())) for n in nodes]
            cx = sum(x for x, y in coords) / len(coords)
            cy = sum(y for x, y in coords) / len(coords)
            xx = sum((x-cx)**2 for x, y in coords)
            yy = sum((y-cy)**2 for x, y in coords)
            xy = sum((x-cx)*(y-cy) for x, y in coords)
            angle = 0.5 * math.atan2(2*xy, xx-yy)
            angle = round(angle / (math.pi/6)) * (math.pi/6)
            rotated = [(math.cos(angle)*(x-cx)+math.sin(angle)*(y-cy),
                        -math.sin(angle)*(x-cx)+math.cos(angle)*(y-cy)) for x, y in coords]
            node_indices = {id_map[i]: i for i in id_map}
            selected = set(candidate["highlighted_parent_union_atom_indices"])
            selected_x = [x for n, (x, y) in zip(nodes, rotated) if node_indices[n.attrib["id"]] in selected]
            other_x = [x for n, (x, y) in zip(nodes, rotated) if node_indices[n.attrib["id"]] not in selected]
            if selected_x and other_x and sum(selected_x)/len(selected_x) < sum(other_x)/len(other_x):
                rotated = [(-x, -y) for x, y in rotated]
            min_x = min(x for x, y in rotated)
            min_y = min(y for x, y in rotated)
            for node, (x, y) in zip(nodes, rotated):
                node.set("p", f"{x-min_x+36:.6f} {y-min_y+36:.6f}")
            xml.set("BoundingBox", f"0 0 {max(x for x, y in rotated)-min_x+72:.6f} {max(y for x, y in rotated)-min_y+72:.6f}")
            # Ordinary bonds stay fine; selected bonds use a modestly wider
            # line without changing the bond display type or stereochemistry.
            xml.attrib.update({
                "BondLength": "18", "LineWidth": "0.9", "BoldWidth": "2.4",
                "BondSpacing": "18", "LabelFont": "3", "LabelSize": f"{args.label_size:g}",
                "LabelFace": "0", "CaptionFont": "3", "CaptionSize": f"{args.label_size:g}",
                "MarginWidth": "1.5", "ShowAtomNumber": "no",
                "ShowTerminalCarbonLabels": "no", "ShowNonTerminalCarbonLabels": "no",
            })
            fonts = xml.find("fonttable")
            if fonts is not None:
                xml.remove(fonts)
            fonts = ET.SubElement(xml, "fonttable")
            ET.SubElement(fonts, "font", {"id": "3", "charset": "iso-8859-1", "name": "Arial"})
            # Color indices 0/1 are implicit, table entries start at 2.
            colors = xml.find("colortable")
            assert colors is not None
            color_index = str(len(list(colors)) + 2)
            rgb = (0.20, 0.43, 0.65) if candidate["role"] == "ground_truth" else (0.76, 0.35, 0.18)
            ET.SubElement(colors, "color", dict(zip(("r", "g", "b"), map(str, rgb))))
            variants = [("molecule", candidate["highlighted_parent_union_atom_indices"])]
            variants += [(f"fragment_hypothesis_{i+1}", h["proposal_atom_indices"])
                         for i, h in enumerate(candidate["displayed_hypotheses"])]
            for suffix, selected in variants:
                doc = ET.fromstring(ET.tostring(xml))
                wanted = {id_map[i] for i in selected}
                for node in doc.iter("n"):
                    if node.attrib["id"] in wanted:
                        node.set("color", color_index)
                for bond in doc.iter("b"):
                    if bond.attrib["B"] in wanted and bond.attrib["E"] in wanted:
                        bond.set("color", color_index)
                        bond.set("LineWidth", "2.2")
                filename = f"{key}_{suffix}.cdxml"
                ET.ElementTree(doc).write(out / filename, encoding="utf-8", xml_declaration=True)
            records.append({
                "key": key, "formula": mol.Formula(), "cleaned_mol": f"{key}_cleaned.mol",
                "original_atom_ids": atom_ids, "original_index_to_cdxml_id": id_map,
                "variants": {suffix: selected for suffix, selected in variants},
            })
    manifest = {
        "renderer": "ChemScript 25.0.2 / ChemDraw Professional",
        "atom_label_size_pt": args.label_size,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "scope": "Depiction only; original case data and atom masks unchanged",
        "style": "0.9 pt ordinary bonds; selected colored bonds 2.2 pt; no filled highlight halos",
        "ordinary_bond_width_pt": 0.9,
        "highlight_bond_width_pt": 2.2,
        "structures": records,
    }
    (out / "chemdraw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared", len(records), "ChemDraw structures and their original atom-mask variants:", out)
    dll_directory.close()


if __name__ == "__main__":
    main()
