#!/usr/bin/env python3
"""Draw Figure 5 from audited cases; RDKit SVG and native librsvg/Cairo vectors.

No model inference, spectrum matching, selection or manuscript writes occur here.
The small fragment SVGs depict retained proposal atom sets on the intact parent;
they do not invent capped or charged fragment species.
"""
from __future__ import annotations
import argparse
import copy
import ctypes as C
import ctypes.util
import html
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from rdkit import Chem,rdBase
from rdkit.Chem.Draw import rdMolDraw2D
from tools.evaluation.official_s7_common import install_no_test_guard,read,write,sha,GUARD_EVENTS

SVG='http://www.w3.org/2000/svg';ET.register_namespace('',SVG)
W,H=1600,1320
# Same method colors and neutral ink as existing plot_jcheminf_figures.py.
COLORS={'ground_truth_only':'#2a78d6','decoy_only':'#eb6834','both':'#1baf7a','neither':'#b5b4ae'}
INK='#0b0b0b';SECONDARY='#52514e';GRID='#e4e3df'
DEFAULT=ROOT/'outputs/figure5_validation_cases_20261112'


def element(parent,tag,attrs=None,text=None):
    node=ET.SubElement(parent,'{'+SVG+'}'+tag,{str(k):str(v) for k,v in (attrs or {}).items()})
    if text is not None:node.text=str(text)
    return node


def text(parent,x,y,value,size=28,weight='normal',fill=INK,anchor='start',extra=None):
    attrs=dict(x=x,y=y,fill=fill,**{'font-size':size,'font-weight':weight,'text-anchor':anchor,'font-family':'DejaVu Sans, sans-serif'})
    attrs.update(extra or {});return element(parent,'text',attrs,value)


def rgb(color):return tuple(int(color[i:i+2],16)/255 for i in (1,3,5))


def molecule_svg(candidate,atoms,width=430,height=220,numbered=False):
    mol=Chem.MolFromMolBlock(candidate['molblock'],sanitize=True,removeHs=False)
    assert mol is not None and mol.GetNumAtoms()==len(candidate['atoms'])
    assert Chem.MolToInchiKey(mol)[:14]==candidate['connectivity']
    for atom,stored in zip(mol.GetAtoms(),candidate['atoms']):
        assert atom.GetIdx()==stored['index'] and atom.GetSymbol()==stored['symbol']
    color=COLORS['ground_truth_only' if candidate['role']=='ground_truth' else 'decoy_only']
    bonds=[b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    drawer=rdMolDraw2D.MolDraw2DSVG(width,height)
    opts=drawer.drawOptions();opts.clearBackground=False;opts.padding=.08
    opts.useBWAtomPalette();opts.bondLineWidth=1.8 if width>300 else 1.5
    opts.fixedFontSize=30 if width>300 else 24
    opts.addAtomIndices=numbered
    opts.continuousHighlight=True;opts.fillHighlights=True
    highlight={i:rgb(color) for i in atoms};bond_colors={i:rgb(color) for i in bonds}
    drawer.DrawMolecule(mol,highlightAtoms=list(atoms),highlightBonds=bonds,
        highlightAtomColors=highlight,highlightBondColors=bond_colors,highlightAtomRadii={i:.25 for i in atoms})
    drawer.FinishDrawing();return drawer.GetDrawingText()


def annotate_svg(source,title,description):
    node=ET.fromstring(source)
    title_node=ET.Element('{'+SVG+'}title');title_node.text=title;node.insert(0,title_node)
    desc=ET.Element('{'+SVG+'}desc');desc.text=description;node.insert(1,desc)
    return ET.tostring(node,encoding='unicode')


def insert_svg(parent,source,x,y,width,height):
    node=ET.fromstring(source);node.set('x',str(x));node.set('y',str(y))
    node.set('width',str(width));node.set('height',str(height))
    parent.append(node)


def spectrum(parent,case,x,y):
    left,top,right,bottom=x+96,y+139,x+600,y+348
    element(parent,'line',dict(x1=left,y1=bottom,x2=right,y2=bottom,stroke=INK,**{'stroke-width':1.8}))
    maxmz=max(max(p['mz'] for p in case['peaks'])*1.04,case['precursor_mz']*1.025)
    step=100 if maxmz>350 else 50 if maxmz>170 else 25
    for tick in range(0,int(maxmz)+1,step):
        px=left+(right-left)*tick/maxmz
        element(parent,'line',dict(x1=px,y1=bottom,x2=px,y2=bottom+7,stroke=SECONDARY,**{'stroke-width':1.4}))
        text(parent,px,bottom+34,tick,25,fill=SECONDARY,anchor='middle')
    for tick in (0,50,100):
        py=bottom-(bottom-top)*tick/100
        element(parent,'line',dict(x1=left,y1=py,x2=right,y2=py,stroke=GRID,**{'stroke-width':1}))
        text(parent,left-14,py+8,tick,25,fill=SECONDARY,anchor='end')
    text(parent,(left+right)/2,bottom+60,'m/z',29,anchor='middle',extra={'font-style':'italic'})
    text(parent,x+20,(top+bottom)/2,'Relative intensity (%)',25,anchor='middle',fill=SECONDARY,
        extra={'transform':f'rotate(-90 {x+20} {(top+bottom)/2})'})
    selected={h['peak_index'] for c in case['candidates'] for h in c['displayed_hypotheses']}
    marks=[]
    for p in case['peaks']:
        px=left+(right-left)*p['mz']/maxmz;py=bottom-(bottom-top)*p['relative_intensity']
        element(parent,'line',dict(x1=px,y1=bottom,x2=px,y2=py,stroke=COLORS[p['category']],
            **{'stroke-width':3.2 if p['category']!='neither' else 2.1}))
        if p['peak_index'] in selected:marks.append((px,py,p['peak_index']))
    # Deterministic two-level labels; no peak locations/intensities are moved.
    last=-1e9;level=0
    for px,py,index in sorted(marks):
        level=1-level if px-last<70 else 0;last=px
        label_y=max(top-9,py-13-30*level)
        text(parent,min(right-20,max(left+20,px)),label_y,f'p{index+1}',24,weight='bold',anchor='middle')
    legend=[('ground_truth_only','GT only'),('decoy_only','Decoy only'),('both','Both'),('neither','Neither')]
    for j,(key,label) in enumerate(legend):
        col=j%2;row=j//2;lx=x+80+260*col;ly=y+488+31*row
        element(parent,'line',dict(x1=lx,y1=ly-7,x2=lx+34,y2=ly-7,stroke=COLORS[key],**{'stroke-width':5}))
        text(parent,lx+44,ly,label,25)
    if case['case']=='A':
        text(parent,x+60,y+448,'Pair already ranks GT first',25,fill=SECONDARY)
    else:text(parent,x+60,y+448,f"Identical masks: {case['matched_peak_counts'][0]} matched peaks",25,fill=SECONDARY)


def candidate_panel(parent,case,candidate,x,y,assets):
    role=candidate['role'];role_label='Ground truth' if role=='ground_truth' else 'Decoy'
    color=COLORS['ground_truth_only' if role=='ground_truth' else 'decoy_only']
    text(parent,x+10,y+29,role_label,33,'bold',color)
    text(parent,x+10,y+58,candidate['formula'].translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')),27,fill=SECONDARY)
    union=candidate['highlighted_parent_union_atom_indices']
    src=molecule_svg(candidate,union,width=430,height=215)
    src=annotate_svg(src,f"Case {case['case']} {role_label}",
        f"Parent structure; union of shown mass-feasible proposal atom sets (zero-based): {union}. Not confirmed fragments.")
    filename=f"{case['case']}_{role}_molecule.svg";(assets/filename).write_text(src)
    numbered=molecule_svg(candidate,[],width=700,height=400,numbered=True)
    (assets/f"{case['case']}_{role}_atom_indices.svg").write_text(numbered)
    insert_svg(parent,src,x,y+67,430,215)
    ranks=candidate['ranks']
    text(parent,x+10,y+310,f"Stage 1 → final: {ranks['ensemble']['frozen_ordinal']} → {ranks['final']['frozen_ordinal']}",29)
    text(parent,x+10,y+345,f"Coverage (top-20): {candidate['scores']['coverage']:.4f}",27)
    hypotheses=candidate['displayed_hypotheses']
    if not hypotheses:
        text(parent,x+10,y+413,'No frozen-family match',25,fill=SECONDARY)
    for j,hypothesis in enumerate(hypotheses):
        px=x+215*j
        mini=molecule_svg(candidate,hypothesis['proposal_atom_indices'],width=210,height=112)
        mini=annotate_svg(mini,'Mass-feasible retained-atom hypothesis',
            f"Case {case['case']} {role_label}; peak index {hypothesis['peak_index']}; {hypothesis['route']}; H adjustment {hypothesis['hydrogen_adjustment']}; signed ppm {hypothesis['signed_ppm_error']}; zero-based retained atoms {hypothesis['proposal_atom_indices']}. Parent context shown; no confirmed ion or loss assignment.")
        filename=f"{case['case']}_{role}_fragment_hypothesis_{j+1}.svg";(assets/filename).write_text(mini)
        insert_svg(parent,mini,px,y+362,210,112)
        peak=case['peaks'][hypothesis['peak_index']]
        text(parent,px+105,y+497,f"p{hypothesis['peak_index']+1}: {peak['mz']:.4f}",25,anchor='middle')
        route='Direct' if hypothesis['route']=='direct' else 'NL*'
        delta=hypothesis['hydrogen_adjustment']
        text(parent,px+105,y+527,f'{route}; ΔH {delta:+d}',23,anchor='middle',fill=SECONDARY)
        text(parent,px+105,y+554,f"{hypothesis['signed_ppm_error']:+.2f} ppm",23,anchor='middle',fill=SECONDARY)


class Rect(C.Structure):_fields_=[('x',C.c_double),('y',C.c_double),('width',C.c_double),('height',C.c_double)]
class GError(C.Structure):_fields_=[('domain',C.c_uint),('code',C.c_int),('message',C.c_char_p)]


def render_native(svg_path,pdf_path,png_path,width_mm=185,scale=2):
    """Use installed system libraries, preserving paths/text in PDF; no pip installs."""
    rsvg=C.CDLL(C.util.find_library('rsvg-2'));cairo=C.CDLL(C.util.find_library('cairo'));gobj=C.CDLL(C.util.find_library('gobject-2.0'))
    vp=C.c_void_p
    rsvg.rsvg_handle_new_from_data.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(vp)];rsvg.rsvg_handle_new_from_data.restype=vp
    rsvg.rsvg_handle_render_document.argtypes=[vp,vp,C.POINTER(Rect),C.POINTER(vp)];rsvg.rsvg_handle_render_document.restype=C.c_int
    cairo.cairo_pdf_surface_create.argtypes=[C.c_char_p,C.c_double,C.c_double];cairo.cairo_pdf_surface_create.restype=vp
    cairo.cairo_image_surface_create.argtypes=[C.c_int,C.c_int,C.c_int];cairo.cairo_image_surface_create.restype=vp
    cairo.cairo_create.argtypes=[vp];cairo.cairo_create.restype=vp
    for name in ('cairo_destroy','cairo_surface_finish','cairo_surface_destroy'):
        getattr(cairo,name).argtypes=[vp];getattr(cairo,name).restype=None
    cairo.cairo_status.argtypes=[vp];cairo.cairo_status.restype=C.c_int
    cairo.cairo_surface_status.argtypes=[vp];cairo.cairo_surface_status.restype=C.c_int
    cairo.cairo_surface_write_to_png.argtypes=[vp,C.c_char_p];cairo.cairo_surface_write_to_png.restype=C.c_int
    gobj.g_object_unref.argtypes=[vp];gobj.g_object_unref.restype=None
    data=svg_path.read_bytes();error=vp();handle=rsvg.rsvg_handle_new_from_data(data,len(data),C.byref(error))
    if not handle:raise RuntimeError(C.cast(error,C.POINTER(GError)).contents.message.decode())
    width_pt=width_mm/25.4*72;height_pt=width_pt*H/W
    surfaces=[('pdf',cairo.cairo_pdf_surface_create(str(pdf_path).encode(),width_pt,height_pt),width_pt,height_pt),
        ('png',cairo.cairo_image_surface_create(0,W*scale,H*scale),W*scale,H*scale)]
    for mode,surface,width,height in surfaces:
        assert cairo.cairo_surface_status(surface)==0
        cr=cairo.cairo_create(surface);viewport=Rect(0,0,width,height);err=vp()
        ok=rsvg.rsvg_handle_render_document(handle,cr,C.byref(viewport),C.byref(err))
        if not ok:raise RuntimeError(C.cast(err,C.POINTER(GError)).contents.message.decode())
        assert cairo.cairo_status(cr)==0
        if mode=='png':assert cairo.cairo_surface_write_to_png(surface,str(png_path).encode())==0
        cairo.cairo_destroy(cr);cairo.cairo_surface_finish(surface);cairo.cairo_surface_destroy(surface)
    gobj.g_object_unref(handle)
    assert b'/Subtype /Image' not in pdf_path.read_bytes(),'Unexpected raster image in vector PDF'
    return dict(renderer='system librsvg + Cairo, via ctypes; no environment upgrades',
        libraries={k:C.util.find_library(k) for k in ('rsvg-2','cairo','gobject-2.0')},
        pdf_width_mm=width_mm,pdf_height_mm=width_mm*H/W,preview_pixels=[W*scale,H*scale],
        pdf_contains_image_xobjects=False)


def main():
    install_no_test_guard();assert rdBase.rdkitVersion=='2023.09.6'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,default=DEFAULT/'cases.json')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output_dir.resolve()
    assert out.is_relative_to(ROOT/'outputs'),'Figure generation is confined to outputs/, never manuscript/'
    out.mkdir(parents=True,exist_ok=True);assets=out/'molecules';assets.mkdir(exist_ok=True)
    payload=read(args.cases);assert payload['rdkit_version']=='2023.09.6'
    svg=ET.Element('{'+SVG+'}svg',dict(width='185mm',height=f'{185*H/W}mm',viewBox=f'0 0 {W} {H}',version='1.1'))
    element(svg,'rect',dict(x=0,y=0,width=W,height=H,fill='white'))
    text(svg,36,36,'Official-validation explanatory cases',30,'bold')
    text(svg,W-35,35,'Post hoc · fixed seed 20261112',24,fill=SECONDARY,anchor='end')
    for i,case in enumerate(payload['cases']):
        y=64+i*590
        element(svg,'rect',dict(x=18,y=y-8,width=W-36,height=575,rx=10,fill='none',stroke=GRID,**{'stroke-width':1.5}))
        title='System rescue' if case['case']=='A' else 'Unresolved coverage ambiguity'
        text(svg,40,y+29,f"{case['case']}  {title}",32,'bold')
        text(svg,40,y+64,case['identifier'],24,fill=SECONDARY)
        text(svg,40,y+95,f"[M+H]+ · precursor {case['precursor_mz']:.4f}",24,fill=SECONDARY)
        spectrum(svg,case,34,y)
        for candidate,x in zip(case['candidates'],(686,1140)):
            candidate_panel(svg,case,candidate,x,y,assets)
    # Additional footer space below the second row's hypothesis captions.
    text(svg,40,H-54,'All highlighted sets and Direct/NL assignments are mass-feasible hypotheses, not confirmed fragments or losses.',23,fill=SECONDARY)
    text(svg,40,H-22,'Neither = no match in the frozen proposal family. NL*: neutral-loss test, retained atoms highlighted. B is unresolved only here.',23,fill=SECONDARY)
    source=ET.tostring(svg,encoding='unicode');assert '<image' not in source
    whole=out/'figure5_validation_cases.svg';whole.write_text(source+'\n')
    rendering=render_native(whole,out/'figure5_validation_cases.pdf',out/'figure5_validation_cases.png')
    write(out/'rendering_manifest.json',dict(status='COMPLETE',rdkit_version=rdBase.rdkitVersion,
        source={'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},
        cases_sha256=sha(args.cases),layout='Two rows; measured spectrum + GT and decoy parent/proposal views',
        colors=COLORS,renderer=rendering,proposal_view_semantics='Retained atom sets on intact parent; no capped/charged fragment structure invented',
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.suffix in ('.svg','.pdf','.png')},
        forbidden_test_file_operations=GUARD_EVENTS))
    print('RENDERED',whole,'PDF/PNG',rendering,flush=True)


if __name__=='__main__':main()
