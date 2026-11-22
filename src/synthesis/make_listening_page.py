"""Build a standalone listening page per language: one row per sentence, one audio player per system (ground truth
first), plus the reference voices. Paths are relative, so open audios/<Language>/index.html directly or serve the folder:
    python src/synthesis/make_listening_page.py --language Yoruba
    cd audios/Yoruba && python -m http.server 8765      # then open http://localhost:8765/ (forward the port from VS Code)
"""
import argparse, glob, html, os
import pandas as pd

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
ap = argparse.ArgumentParser(); ap.add_argument("--language", required=True)
ap.add_argument("--exclude", default="", help="comma-separated system folders to leave out (e.g. a system whose synthesis is incomplete)")
args = ap.parse_args(); exclude = set(filter(None, args.exclude.split(",")))
root = f"{REPO}/audios/{args.language}"
gt = pd.read_csv(f"{root}/ground-truth/metadata.csv", dtype=str).sort_values("clip_id")
systems = {}                                                           # name -> {clip_id: (relpath, label)}
for m in sorted(glob.glob(f"{root}/*/metadata.csv")):
    name = m.split("/")[-2]
    if name in ("ground-truth", "reference-voices") or name in exclude: continue
    d = pd.read_csv(m, dtype=str)
    systems[name] = {r.clip_id: (os.path.relpath(f"{REPO}/{r.audio_path}", root), f"{r.voice_id} · {float(r.duration_s):.1f} s") for r in d.itertuples()}
refs = pd.read_csv(f"{root}/reference-voices/metadata.csv", dtype=str) if os.path.exists(f"{root}/reference-voices/metadata.csv") else None

def player(rel, label):
    """The ?v= query string is the file's mtime: clip ids are reused when the ground truth is re-sampled, and without it the
    browser keeps playing the cached old audio under the new text."""
    v = int(os.path.getmtime(os.path.join(root, rel)))
    return f'<audio controls preload="none" src="{html.escape(rel)}?v={v}"></audio><div class="lab">{html.escape(label)}</div>'

cols = ["ground-truth"] + list(systems)
rows = []
for r in gt.itertuples():
    cells = [player(os.path.relpath(f"{REPO}/{r.audio_path}", root), f"{r.speaker_id} · {r.gender} · {float(r.duration_s):.1f} s")]
    cells += [player(*systems[s][r.clip_id]) if r.clip_id in systems[s] else "<span class=miss>missing</span>" for s in systems]
    rows.append(f"<tr><th class=id>{r.clip_id}<br><small>{html.escape(r.dataset)} · {html.escape(r.domain)}</small>"
                f"<div class=txt>{html.escape(r.text)}</div></th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
ref_html = "" if refs is None else "<h2>Reference voices (cloning models)</h2><table><tr>" + "".join(
    f"<td>{player(os.path.relpath(f'{REPO}/{x.audio_path}', root), f'{x.ref_id} · {x.gender} · {x.age}')}<div class=txt>{html.escape(x.text)}</div></td>"
    for x in refs.itertuples()) + "</tr></table>"
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.language)} listening page</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}} h1{{margin:0 0 4px}} .sub{{color:#666;margin-bottom:16px}}
table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 8px;vertical-align:top;text-align:left}}
thead th{{position:sticky;top:0;background:#fff;z-index:1}} th.id{{min-width:260px;max-width:340px;font-weight:600}}
.txt{{font-weight:400;font-size:13px;margin-top:4px;white-space:normal}} .lab{{font-size:11px;color:#666}} audio{{width:210px;height:32px}} .miss{{color:#b00}}
tr:nth-child(even) td{{background:#f3f3f3}}
</style></head><body>
<h1>{html.escape(args.language)}</h1><div class=sub>{len(gt)} sentences × {len(cols)} systems (ground truth + {len(systems)} TTS). Each cell: player, voice id, duration.</div>
<table><thead><tr><th class=id>sentence</th>{"".join(f"<th>{html.escape(c)}</th>" for c in cols)}</tr></thead><tbody>{"".join(rows)}</tbody></table>
{ref_html}
</body></html>"""
open(f"{root}/index.html", "w").write(page)
print(f"{root}/index.html: {len(gt)} sentences × {len(cols)} systems")
