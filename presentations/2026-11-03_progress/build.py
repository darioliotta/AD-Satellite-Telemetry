#!/usr/bin/env python3
"""Build a compact English results deck from the original saved experiments.

Dependencies: numpy, matplotlib, Pillow, python-pptx, lxml.
Run from the installed presentation folder: python build.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/toast-presentation-mpl")
if Path("/tmp/toast-presentation-deps").is_dir():
    sys.path.insert(0, "/tmp/toast-presentation-deps")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.opc.package import Part
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from lxml import etree


RED = "#9B0014"
INK = "#252525"
GRAY = "#737373"
PALE = "#FCF4F5"
GRID = "#E7E7E7"
DATASETS = ("OPS-SAT", "SMAP", "MSL")
BASE = {"OPS-SAT": (10, 8), "SMAP": (10, 50), "MSL": (50, 110)}
SMALL = {"OPS-SAT": (3, 2), "SMAP": (3, 13), "MSL": (13, 28)}
JOINT = {"OPS-SAT": [(10, 8), (8, 6), (5, 4), (3, 2)],
         "SMAP": [(10, 50), (8, 38), (5, 25), (3, 13)],
         "MSL": [(50, 110), (38, 83), (25, 55), (13, 28)]}
ORDER = ("DSN_1k", "SMAP", "MSL", "SWaT", "SMD", "WADI")
PAPER = {"DSN_1k": (.9247305754501705, .8043825114944156),
         "SMAP": (.8424058108497716, .8490943365297927),
         "MSL": (.8900594107919398, .8195600136783313),
         "SWaT": (.3290405792926449, .8239396071908388),
         "SMD": (.7129061265981155, .8508932548126886),
         "WADI": (.5084310441094361, .27884117951370924)}
W, H = 13.333, 7.5

plt.rcParams.update({
    "font.family": "Arial", "font.size": 17,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.titlesize": 20, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A0A0A0", "axes.linewidth": .8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "path",
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_results(project):
    path = project / "tranad_plus/reports/plot_ablation_grid.py"
    spec = importlib.util.spec_from_file_location("original_ablation_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs, cells, provenance = module.load_reports(project / "results")
    assert len(runs) == 270 and len(cells) == 90
    best, selected = {}, []
    for dataset in ORDER:
        for method in ("pot", "bf"):
            candidates = []
            for source in (project / "results/legacy" / dataset).rglob("*_" + method + ".pkl"):
                with source.open("rb") as stream:
                    values = pickle.load(stream)
                row = dict(dataset=dataset, method=method,
                           source=str(source.relative_to(project)), source_sha256=sha256(source),
                           **{k: float(values[method + "_" + k])
                              for k in ("f1", "precision", "recall", "ROC/AUC")})
                candidates.append(row)
            if dataset == "SWaT":
                target = PAPER[dataset][0 if method == "pot" else 1]
                choice = min(candidates, key=lambda r: (abs(r["f1"] - target), r["source"]))
            else:
                choice = max(candidates, key=lambda r: (r["f1"], r["source"]))
            best[dataset, method] = choice
            selected.append(choice)
    return cells, best, selected, provenance


def canvas():
    return plt.figure(figsize=(W, H), layout=None)


def read_swat20(project):
    root=project/"results/legacy/SWaT/_native_pickles"
    records=[]
    for seed in (1,2,3):
        name=f"papercheck_SWaT_bf_w20_ff816_l4_bs2048_s{seed}"
        row={"seed":seed,"identifier":name}
        for method in ("pot","bf"):
            source=root/(name+"_"+method+".pkl")
            with source.open("rb") as stream:
                values=pickle.load(stream)
            for key in ("f1","precision","recall","ROC/AUC"):
                row[method+"_"+key]=float(values[method+"_"+key])
            row[method+"_source"]=str(source.relative_to(project))
            row[method+"_sha256"]=sha256(source)
        records.append(row)
    return max(records,key=lambda r:r["bf_f1"]),records


def plain_axis(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def title(fig, text):
    fig.text(.055, .95, text, fontsize=29, color=RED, weight="bold", va="top")


def table(fig, headers, rows, widths, *, top=.79, row_height=.092,
          fontsize=22, left=.055, groups=1, left_columns=(0,)):
    ax = plain_axis(fig)
    total = sum(widths)
    ax.add_patch(Rectangle((left, top-row_height), total, row_height,
                           facecolor=RED, edgecolor="none"))
    x = left
    for header, width in zip(headers, widths):
        ax.text(x+.013, top-row_height/2, header, color="white", fontsize=fontsize-2,
                weight="bold", va="center", ha="left")
        x += width
    for i, row in enumerate(rows):
        y = top-row_height*(i+2)
        background = "white" if (i//groups) % 2 else PALE
        ax.add_patch(Rectangle((left, y), total, row_height,
                               facecolor=background, edgecolor="none"))
        x = left
        for j, (value, width) in enumerate(zip(row, widths)):
            left_aligned = j in left_columns
            ax.text(x+.013 if left_aligned else x+width-.018,
                    y+row_height/2, str(value), fontsize=fontsize, color=INK,
                    va="center", ha="left" if left_aligned else "right")
            x += width
        if groups > 1 and (i+1) % groups == 0:
            ax.plot([left, left+total], [y, y], color="#DDDDDD", lw=.8)


def model_figure():
    fig = canvas()
    ax = plain_axis(fig)
    def box(x, y, w, h, label, sub="", accent=True):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=.012",
                                   facecolor=PALE if accent else "#F5F5F5",
                                   edgecolor=RED if accent else "#9B9B9B",lw=1.1))
        ax.text(x+w/2,y+h*.65,label,ha="center",va="center",fontsize=23,
                color=RED if accent else INK,weight="bold")
        if sub:ax.text(x+w/2,y+h*.30,sub,ha="center",va="center",fontsize=17,color=GRAY)
    def line(a,b,arrow=True):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>" if arrow else "-",
                                    mutation_scale=15,lw=1.5,color=GRAY))
    box(.055,.37,.15,.20,"Window","W × f",False)
    box(.28,.60,.21,.16,"Encoder","shared weights")
    box(.57,.60,.18,.16,"Decoder 1",accent=False)
    box(.28,.25,.21,.16,"Encoder","same weights")
    box(.57,.25,.18,.16,"Decoder 2",accent=False)
    box(.82,.40,.13,.20,"Score","MSE",True)
    line((.205,.52),(.245,.52),False);line((.245,.52),(.245,.68),False)
    line((.245,.68),(.278,.68));line((.205,.43),(.245,.43),False)
    line((.245,.43),(.245,.33),False);line((.245,.33),(.278,.33))
    line((.49,.68),(.568,.68));line((.49,.33),(.568,.33))
    line((.65,.599),(.65,.49),False);line((.65,.49),(.385,.49),False)
    line((.385,.49),(.385,.412))
    ax.text(.52,.515,"(D1 − X)²",fontsize=20,color=INK,ha="center")
    line((.75,.68),(.79,.68),False);line((.79,.68),(.79,.55),False)
    line((.79,.55),(.818,.55));line((.75,.33),(.79,.33),False)
    line((.79,.33),(.79,.445),False);line((.79,.445),(.818,.445))
    ax.text(.76,.16,"α D1 + (1 − α) D2",fontsize=22,ha="center",color=INK)
    return fig


def datasets_figure():
    fig=canvas()
    rows=[["OPS-SAT","ESA CubeSat","1","Segments","10","8"],
          ["SMAP","Earth observation","25","Time points","10","50"],
          ["MSL","Mars rover","55","Time points","50","110"]]
    table(fig,["Dataset","Mission","Inputs","Labels","W0","FF0"],rows,
          [.14,.27,.10,.18,.10,.10],row_height=.108,fontsize=23,left_columns=(0,1,3))
    for x, number, label in [(.19,"30","configurations / dataset"),
                             (.50,"3","seeds"),(.81,"50","epochs")]:
        fig.text(x,.235,number,fontsize=38,color=RED,ha="center",weight="bold")
        fig.text(x,.17,label,fontsize=18,color=GRAY,ha="center")
    return fig


def benchmark_figure(best):
    fig=canvas()
    rows=[]
    for dataset in ORDER:
        p,b=PAPER[dataset]
        rows.append([dataset.replace("_"," "),f"{100*p:.2f}",
                     f"{100*best[dataset,'pot']['f1']:.2f}",f"{100*b:.2f}",
                     f"{100*best[dataset,'bf']['f1']:.2f}"])
    table(fig,["Dataset","Paper POT","Our POT","Paper Best-F1","Our Best-F1"],
          rows,[.18,.17,.17,.20,.17],top=.82,row_height=.088,fontsize=23)
    return fig


def legend(fig):
    handles=[Line2D([],[],color=INK,lw=2.2,marker="o",ms=5,label="POT"),
             Line2D([],[],color=RED,lw=2.6,marker="o",ms=5,label="Best-F1 (dense)")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.96,.955),
               ncol=2,frameon=False,fontsize=16,columnspacing=1.5,handlelength=2.0)


def axes3(fig):
    axes=fig.subplots(1,3)
    fig.subplots_adjust(left=.075,right=.965,top=.79,bottom=.16,wspace=.32)
    return axes


def trend_figure(cells, lookup, direction):
    fig=canvas();axes=axes3(fig);legend(fig)
    for ax,dataset in zip(axes,DATASETS):
        w0,h0=BASE[dataset]
        if direction=="window":
            pairs=sorted([(c["window"],h0) for c in cells
                          if c["dataset"]==dataset and c["feedforward"]==h0])
            xs=[w for w,h in pairs]
            heading=f"{dataset} · FF {h0}";xlabel="Window size"
        elif direction=="ff":
            pairs=sorted([(w0,c["feedforward"]) for c in cells
                          if c["dataset"]==dataset and c["window"]==w0],key=lambda p:p[1])
            xs=[h for w,h in pairs]
            heading=f"{dataset} · W {w0}";xlabel="FF width"
        else:
            pairs=JOINT[dataset];xs=np.arange(4)
            heading=dataset;xlabel="W / FF"
        lower,upper=100.,0.
        for prefix,color in (("pot",INK),("dense",RED)):
            group=[lookup[dataset,*p] for p in pairs]
            means=np.array([100*c[prefix+"_f1_mean"] for c in group])
            sd=np.array([100*c[prefix+"_f1_std"] for c in group])
            ax.fill_between(xs,means-sd,means+sd,color=color,alpha=.10,linewidth=0)
            ax.errorbar(xs,means,yerr=sd,color=color,lw=2.2,marker="o",ms=5,
                        capsize=3,elinewidth=1.1)
            lower=min(lower,float((means-sd).min()))
            upper=max(upper,float((means+sd).max()))
        ax.set(title=heading,xlabel=xlabel,ylabel="F1 (%)",
               ylim=(max(0,np.floor((lower-2)/5)*5),min(100,np.ceil((upper+2)/5)*5)))
        ax.set_xticks(xs)
        if direction=="joint":
            ax.set_xticklabels([f"{w}/{h}" for w,h in pairs],fontsize=14)
        ax.grid(axis="y",color=GRID,lw=.7);ax.set_axisbelow(True)
    return fig


def heatmap_figure(cells,lookup,dataset):
    fig=canvas()
    axes=fig.subplots(1,2)
    fig.subplots_adjust(left=.09,right=.965,top=.79,bottom=.16,wspace=.28)
    windows=sorted({c["window"] for c in cells if c["dataset"]==dataset})
    widths=sorted({c["feedforward"] for c in cells if c["dataset"]==dataset})
    palette=LinearSegmentedColormap.from_list("Padova",["#FFF9FA","#F7DDE2","#DA8F9D",RED])
    for ax,prefix,label in zip(axes,("pot","dense"),("POT","Best-F1 (dense)")):
        values=np.array([[100*lookup[dataset,w,h][prefix+"_f1_mean"]
                          for w in windows] for h in widths])
        image=ax.pcolormesh(np.arange(len(windows)+1)-.5,np.arange(len(widths)+1)-.5,
                            values,cmap=palette,shading="flat",rasterized=False,
                            edgecolors="white",linewidth=.7)
        for j,h in enumerate(widths):
            for i,w in enumerate(windows):
                rgb=image.cmap(image.norm(values[j,i]))
                lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                ax.text(i,j,f"{values[j,i]:.1f}",ha="center",va="center",fontsize=19,
                        color=INK if lum>.55 else "white")
        ax.set(title=label,xlabel="Window size",ylabel="FF width",
               xticks=range(len(windows)),xticklabels=windows,
               yticks=range(len(widths)),yticklabels=widths)
        ax.tick_params(length=0,pad=8)
        colorbar=fig.colorbar(image,ax=ax,fraction=.035,pad=.025)
        colorbar.ax.tick_params(labelsize=14,length=0)
        colorbar.outline.set_visible(False)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
    return fig


def cost_figure(cells,lookup):
    fig=canvas();axes=axes3(fig)
    for ax,dataset in zip(axes,DATASETS):
        part=[c for c in cells if c["dataset"]==dataset]
        base=lookup[dataset,*BASE[dataset]];small=lookup[dataset,*SMALL[dataset]]
        x=[100*c["macs_mean"]/base["macs_mean"] for c in part]
        y=[100*c["dense_f1_mean"] for c in part]
        ax.scatter(x,y,color="#D78B98",s=65,edgecolor="white",lw=.6,alpha=.8)
        ax.scatter(100,100*base["dense_f1_mean"],marker="*",color=INK,s=210,zorder=5)
        ax.scatter(100*small["macs_mean"]/base["macs_mean"],100*small["dense_f1_mean"],
                   marker="D",color=RED,s=85,zorder=5)
        ax.set(title=dataset,xlabel="MACs (% baseline)",ylabel="Dense F1 (%)")
        ax.grid(color=GRID,lw=.7);ax.set_axisbelow(True)
    handles=[Line2D([],[],ls="",marker="*",color=INK,ms=14,label="Baseline"),
             Line2D([],[],ls="",marker="D",color=RED,ms=8,label="Smallest")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.96,.955),
               frameon=False,ncol=2,fontsize=16)
    return fig


def compact_figure(lookup):
    fig=canvas();rows=[]
    for dataset in DATASETS:
        for i,(label,pair) in enumerate((("Baseline",BASE[dataset]),("Compact",SMALL[dataset]))):
            c=lookup[dataset,*pair]
            rows.append([dataset if i==0 else "",label,str(pair[0]),str(pair[1]),
                         f"{int(c['parameters_mean']):,}",f"{int(c['macs_mean']):,}",
                         f"{100*c['dense_f1_mean']:.2f}"])
    table(fig,["Dataset","Model","W","FF","Parameters","MACs","Dense F1"],
          rows,[.14,.16,.07,.08,.17,.14,.13],top=.82,row_height=.088,fontsize=21,
          groups=2,left_columns=(0,1))
    return fig


def metrics_figure(lookup):
    fig=canvas();rows=[]
    for dataset in DATASETS:
        for i,(label,pair) in enumerate((("Baseline",BASE[dataset]),("Compact",SMALL[dataset]))):
            c=lookup[dataset,*pair]
            rows.append([dataset if i==0 else "",label,*[f"{100*c[k+'_mean']:.2f}"
                         for k in ("dense_f1","dense_precision","dense_recall","dense_auc")]])
    table(fig,["Dataset","Model","Dense F1","Precision (%)","Recall (%)","PA AUC (%)"],
          rows,[.15,.17,.14,.16,.14,.13],top=.82,row_height=.088,fontsize=21,
          groups=2,left_columns=(0,1))
    return fig


def architecture_proposals_figure():
    fig=canvas()
    ax=plain_axis(fig)
    proposals=[
        (.72,"Shared decoder","One set of decoder weights · two passes"),
        (.49,"Single pass","One encoder–decoder pass · no feedback"),
        (.26,"Encoder only","Encoder + reconstruction head"),
    ]
    for y,label,description in proposals:
        ax.plot(.073,y+.025,"o",color=RED,markersize=8)
        ax.text(.105,y+.025,label,fontsize=27,color=RED,weight="bold",va="center")
        ax.text(.105,y-.055,description,fontsize=22,color=GRAY,va="center")
    return fig


def add_svg(slide, svg, png, package):
    pic=slide.shapes.add_picture(str(png),0,0,width=Inches(W),height=Inches(H))
    part=Part(package.next_partname("/ppt/media/vector%d.svg"),"image/svg+xml",
              package,svg.read_bytes())
    rid=slide.part.relate_to(part,RT.IMAGE)
    blip=pic._pic.blipFill.blip
    ext_list=OxmlElement("a:extLst");blip.append(ext_list)
    ext=OxmlElement("a:ext");ext.set("uri","{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
    node=etree.SubElement(ext,"{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip",
                          nsmap={"asvg":"http://schemas.microsoft.com/office/drawing/2016/SVG/main"})
    node.set(qn("r:embed"),rid);ext_list.append(ext)


def build(project,out):
    out.mkdir(parents=True,exist_ok=True)
    figures=out/"figures";figures.mkdir(exist_ok=True)
    data=out/"data";data.mkdir(exist_ok=True)
    work=out/".render";work.mkdir(exist_ok=True)
    cells,best,selected,provenance=read_results(project)
    swat20,swat20_records=read_swat20(project)
    lookup={(c["dataset"],c["window"],c["feedforward"]):c for c in cells}
    csv_write(data/"ablation.csv",cells);csv_write(data/"benchmarks.csv",selected)
    csv_write(data/"swat20.csv",swat20_records)
    pages=[
        ("01_tranad","TranAD+",lambda:model_figure()),
        ("02_datasets","Datasets & study",lambda:datasets_figure()),
        ("03_benchmarks","Benchmarks · F1 (%)",lambda:benchmark_figure(best)),
        ("04_window","Window size",lambda:trend_figure(cells,lookup,"window")),
        ("05_feedforward","Feed-forward width",lambda:trend_figure(cells,lookup,"ff")),
        ("06_joint","Window + FF",lambda:trend_figure(cells,lookup,"joint")),
        ("07_opssat","OPS-SAT",lambda:heatmap_figure(cells,lookup,"OPS-SAT")),
        ("08_smap","SMAP",lambda:heatmap_figure(cells,lookup,"SMAP")),
        ("09_msl","MSL",lambda:heatmap_figure(cells,lookup,"MSL")),
        ("10_compute","Accuracy vs compute",lambda:cost_figure(cells,lookup)),
        ("11_compact","Compact models",lambda:compact_figure(lookup)),
        ("12_metrics","Precision · Recall · AUC",lambda:metrics_figure(lookup)),
        ("13_architectures","Architecture proposals",lambda:architecture_proposals_figure()),
    ]
    prs=Presentation();prs.slide_width=Inches(W);prs.slide_height=Inches(H)
    prs.core_properties.author="Dario Liotta"
    prs.core_properties.title="TranAD+ — results discussion"
    with PdfPages(out/"Results.pdf",metadata={"Title":"TranAD+ — results discussion","Author":"Dario Liotta"}) as pdf:
        for number,(stem,heading,factory) in enumerate(pages,1):
            print(f"[{number}/{len(pages)}] {heading}",flush=True)
            fig=factory()
            # SVG is placed in PowerPoint; the fallback is 4000 × 2250 pixels.
            svg=figures/(stem+".svg");png=work/(stem+".png")
            fig.savefig(svg)
            fig.savefig(png,dpi=300)
            slide=prs.slides.add_slide(prs.slide_layouts[6])
            add_svg(slide,svg,png,prs.part.package)
            heading_box=slide.shapes.add_textbox(Inches(.055*W),Inches(.05*H),
                                                Inches(8.2),Inches(.60))
            frame=heading_box.text_frame
            frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
            p=frame.paragraphs[0];p.text=heading;p.font.name="Arial";p.font.size=Pt(29)
            p.font.bold=True;p.font.color.rgb=RGBColor.from_string(RED[1:])
            title(fig,heading)
            pdf.savefig(fig)
            fig.savefig(figures/(stem+".pdf"))
            plt.close(fig)
    prs.save(out/"Results.pptx")
    shutil.rmtree(work)
    manifest={"language":"English","slides":len(pages),"colour":RED,
              "selection":"Saved ordinary legacy maxima by dataset and method, except SWaT: closest to the paper for each method. One row per dataset.",
              "ablation":"Mean and sample SD of 3 seeds; 50 epochs; PA; dense BF in ablation; original BF in benchmark.",
              "sources":provenance,"slides_content":[p[1] for p in pages],
              "swat20_selected":swat20,
              "brand_source":"https://www.unipd.it/",
              "builder_sha256":sha256(__file__)}
    (data/"sources.json").write_text(json.dumps(manifest,indent=2))
    (out/"README.md").write_text(f"# Results discussion\n\nEnglish · {len(pages)} slides · Padova red #9B0014.\n\n"
        "Results.pdf: vector presentation.\nResults.pptx: SVG figures with high-resolution fallbacks.\n"
        "figures/: standalone Python-generated SVG/PDF figures.\ndata/: displayed values and source records.\n\n"
        "Rebuild: python build.py\n")
    print("Complete:",out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    build(args.project_root.resolve(),args.output_dir.resolve())
