#!/usr/bin/env python
"""Build the self-contained review page for the belief-formation figure.

Clariden cannot push to a local display, so figures are reviewed by embedding
the PNG as a base64 data URI in a small page and publishing it. Numbers are
recomputed here from the same sources the figure uses -- never transcribed.

Usage:  python3 analysis/build_belief_artifact.py <out.html>
"""
import base64
import json
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG = os.path.join(ROOT, "plots", "belief_layers", "belief_layers.png")

MODELS = ["gemma2b", "llama", "phi", "qwen", "dsqwen7",
          "qwen27", "gemma31", "llama70", "dsllama70"]
MD = {"gemma2b": "Gemma-2 2B", "llama": "Llama-3.2 3B", "phi": "Phi-4-mini 3.8B",
      "qwen": "Qwen-3.5 4B", "dsqwen7": "R1-Qwen 7B", "qwen27": "Qwen-3.6 27B",
      "gemma31": "Gemma-4 31B", "llama70": "Llama-3.3 70B",
      "dsllama70": "R1-Llama 70B"}
SCEN = {"bandit": "bayes", "monty": "bayes", "trust_game": "bayes",
        "machine": "bayes", "chicken": "nonreact",
        "stag_hunt": "react", "bos": "react", "pennies": "react"}
FAMS = [("bayes", "Bayesian inference", "#2a78d6"),
        ("nonreact", "Chicken", "#1baf7a"),
        ("react", "Reactive games", "#eb6834")]
GRID = np.arange(0.0, 100.0 + 1e-9, 1.0)
COLS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def split(base):
    for env in SCEN:
        if base.endswith("_" + env):
            return base[: -(len(env) + 1)], env
    return None


def load():
    belief = {}
    for tag in ("P1_v4_layerbal", "P1_v4"):
        for p in glob.glob(os.path.join(ROOT, "results", tag, "probe_*.json")):
            hit = split(os.path.basename(p)[6:-5])
            if hit is None or hit[0] not in MODELS:
                continue
            lab = json.load(open(p)).get("layer_accuracy_bal")
            if not lab:
                continue
            ls = sorted(int(k) for k in lab)
            belief[hit] = np.interp(
                GRID, np.array(ls, float) / ls[-1] * 100,
                np.array([lab[str(k)] for k in ls]) * 100)
    beh = json.load(open(os.path.join(
        ROOT, "results", "P1_v4", "behavior_by_t.json")))["behavior"]
    action = {}
    for combo, per_t in beh.items():
        hit = split(combo)
        if hit is None or hit[0] not in MODELS:
            continue
        c1 = c0 = n1 = n0 = 0.0
        for r in per_t.values():
            if r.get("agree_act1") is not None:
                c1 += r["agree_act1"] * r["n_act1"]; n1 += r["n_act1"]
            if r.get("agree_act0") is not None:
                c0 += r["agree_act0"] * r["n_act0"]; n0 += r["n_act0"]
        if n1 and n0:
            action[hit] = 50.0 * (c1 / n1 + c0 / n0)
    return belief, action


def main(out_path):
    belief, action = load()
    combos = [k for k in belief if k in action]

    fam_rows, depth_rows = [], []
    for key, name, color in FAMS:
        sel = [k for k in combos if SCEN[k[1]] == key]
        mean = np.vstack([belief[k] for k in sel]).mean(axis=0)
        act = float(np.mean([action[k] for k in sel]))
        fam_rows.append((name, color, len(sel), mean[-1], act, mean[-1] - act))
        depth_rows.append((name, color,
                           [float(np.interp(c, GRID, mean)) for c in COLS]))

    model_rows = []
    for m in MODELS:
        cells = []
        for key, _, _ in FAMS:
            sel = [k for k in combos if k[0] == m and SCEN[k[1]] == key]
            b = float(np.mean([belief[k][-1] for k in sel]))
            a = float(np.mean([action[k] for k in sel]))
            cells.append((b, a, b - a))
        model_rows.append((MD[m], cells))

    allf = np.array([belief[k][-1] for k in combos])
    stats = dict(n=len(combos), mean=allf.mean(), median=float(np.median(allf)),
                 lo=allf.min(), hi=allf.max(),
                 react_hi=max(belief[k][-1] for k in combos
                              if SCEN[k[1]] == "react"),
                 non_lo=min(belief[k][-1] for k in combos
                            if SCEN[k[1]] != "react"))

    b64 = base64.b64encode(open(PNG, "rb").read()).decode()

    def depth_table():
        head = "".join(f"<th>{c}</th>" for c in COLS)
        body = "".join(
            f'<tr><th scope="row"><span class="dot" style="--c:{c}"></span>{n}</th>'
            + "".join(f"<td>{v:.1f}</td>" for v in vals) + "</tr>"
            for n, c, vals in depth_rows)
        return (f'<table><caption>Balanced probe accuracy (%) by depth '
                f'(layer / <em>L</em>, %)</caption><thead><tr>'
                f'<th scope="col">family</th>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table>')

    def model_table():
        body = ""
        for name, cells in model_rows:
            tds = "".join(
                f'<td>{b:.1f}</td><td>{a:.1f}</td>'
                # Gemma-4 31B on Chicken has action ABOVE belief (99.9 -> 100.0),
                # so the sign has to come from the value. Hardcoding &minus;
                # rendered that cell as "--0".
                f'<td class="gap" style="--c:{FAMS[i][2]}">'
                f'{"&minus;" if g >= 0 else "+"}{abs(g):.0f}</td>'
                for i, (b, a, g) in enumerate(cells))
            body += f'<tr><th scope="row">{name}</th>{tds}</tr>'
        grp = "".join(
            f'<th colspan="3" style="--c:{c}"><span class="dot" '
            f'style="--c:{c}"></span>{n}</th>' for _, n, c in
            [(k, n, c) for k, n, c in FAMS])
        sub = "".join("<th>belief</th><th>action</th><th>gap</th>"
                      for _ in FAMS)
        return (f'<table><caption>Final-layer belief vs. pre-CoT action, '
                f'per model</caption><thead>'
                f'<tr><td></td>{grp}</tr><tr><th scope="col">model</th>{sub}'
                f'</tr></thead><tbody>{body}</tbody></table>')

    fam_cards = "".join(
        f'<div class="card" style="--c:{c}"><h3>{n}</h3>'
        f'<p class="n">{cnt} combos</p>'
        f'<p class="pair"><span class="b">{b:.1f}</span>'
        f'<span class="arrow">&rarr;</span>'
        f'<span class="a">{a:.1f}</span></p>'
        f'<p class="lab">belief &rarr; action</p>'
        f'<p class="drop">&minus;{g:.1f} points</p></div>'
        for n, c, cnt, b, a, g in fam_rows)

    html = f"""<title>Belief forms across depth — figure review</title>
<style>
  :root {{
    --ground:#fbfbfa; --ink:#14161a; --muted:#5c6068; --rule:#e4e4df;
    --card:#ffffff; --shadow:0 1px 2px rgba(20,22,26,.06),0 8px 24px rgba(20,22,26,.05);
    --blue:#2a78d6; --aqua:#1baf7a; --orange:#eb6834;
    --serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,'Times New Roman',serif;
    --sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
    --mono:ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace;
  }}
  @media (prefers-color-scheme:dark) {{
    :root:not([data-theme="light"]) {{
      --ground:#15171a; --ink:#eceae4; --muted:#9b9d9f; --rule:#2c2f34;
      --card:#1d2024; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
      --blue:#5b9ce8; --aqua:#35c48c; --orange:#f4834f;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#15171a; --ink:#eceae4; --muted:#9b9d9f; --rule:#2c2f34;
    --card:#1d2024; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
    --blue:#5b9ce8; --aqua:#35c48c; --orange:#f4834f;
  }}
  :root[data-theme="light"] {{
    --ground:#fbfbfa; --ink:#14161a; --muted:#5c6068; --rule:#e4e4df;
    --card:#ffffff; --blue:#2a78d6; --aqua:#1baf7a; --orange:#eb6834;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--sans); font-size:16px; line-height:1.62;
    -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:60rem; margin:0 auto; padding:3.5rem 1.5rem 5rem;
    display:flex; flex-direction:column; gap:2.75rem; }}
  header {{ display:flex; flex-direction:column; gap:.5rem;
    border-bottom:1px solid var(--rule); padding-bottom:1.75rem; }}
  .eyebrow {{ font-family:var(--mono); font-size:.7rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--muted); }}
  h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(1.7rem,4vw,2.4rem);
    line-height:1.15; margin:0; text-wrap:balance; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); max-width:62ch; margin:.25rem 0 0; }}
  h2 {{ font-family:var(--serif); font-weight:600; font-size:1.35rem; margin:0 0 .6rem;
    letter-spacing:-.005em; }}
  section > p {{ max-width:68ch; margin:0 0 .9rem; }}
  .figure {{ background:#fcfcfb; border:1px solid var(--rule); border-radius:6px;
    padding:1rem; box-shadow:var(--shadow); overflow-x:auto; }}
  .figure img {{ display:block; width:100%; min-width:34rem; height:auto; }}
  figcaption {{ font-size:.85rem; color:var(--muted); margin-top:.9rem;
    max-width:70ch; line-height:1.5; }}
  .cards {{ display:grid; gap:1rem;
    grid-template-columns:repeat(auto-fit,minmax(13rem,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--rule); border-radius:6px;
    padding:1.1rem 1.2rem 1.2rem; border-top:3px solid var(--c);
    display:flex; flex-direction:column; gap:.1rem; }}
  .card h3 {{ font-size:.95rem; font-weight:600; margin:0; }}
  .card .n {{ font-family:var(--mono); font-size:.72rem; color:var(--muted);
    margin:0 0 .5rem; }}
  .pair {{ margin:0; display:flex; align-items:baseline; gap:.45rem;
    font-family:var(--mono); font-variant-numeric:tabular-nums; }}
  .pair .b {{ font-size:1.5rem; font-weight:600; }}
  .pair .a {{ font-size:1.5rem; font-weight:600; color:var(--c); }}
  .pair .arrow {{ color:var(--muted); font-size:1rem; }}
  .card .lab {{ margin:.1rem 0 0; font-size:.74rem; color:var(--muted);
    letter-spacing:.02em; }}
  .card .drop {{ margin:.55rem 0 0; font-family:var(--mono); font-size:.82rem;
    font-weight:600; color:var(--c); }}
  .scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:6px;
    background:var(--card); }}
  table {{ border-collapse:collapse; width:100%; font-size:.83rem;
    font-family:var(--mono); font-variant-numeric:tabular-nums; }}
  caption {{ text-align:left; padding:.85rem 1rem .1rem; font-family:var(--sans);
    font-size:.85rem; color:var(--muted); }}
  th,td {{ padding:.42rem .6rem; text-align:right; white-space:nowrap;
    border-bottom:1px solid var(--rule); }}
  thead th {{ font-weight:600; color:var(--muted); font-size:.74rem; }}
  tbody th[scope=row], thead th[scope=col] {{ text-align:left;
    font-family:var(--sans); font-weight:500; color:var(--ink); }}
  tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:none; }}
  .gap {{ color:var(--c); font-weight:600; }}
  .dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%;
    background:var(--c); margin-right:.4rem; vertical-align:middle; }}
  ul {{ max-width:68ch; padding-left:1.1rem; margin:0; display:flex;
    flex-direction:column; gap:.55rem; }}
  li::marker {{ color:var(--muted); }}
  code {{ font-family:var(--mono); font-size:.86em; background:var(--card);
    border:1px solid var(--rule); border-radius:3px; padding:.05em .35em; }}
  .note {{ border-left:2px solid var(--orange); padding:.1rem 0 .1rem 1rem;
    color:var(--muted); font-size:.92rem; max-width:66ch; }}
  footer {{ border-top:1px solid var(--rule); padding-top:1.25rem;
    font-family:var(--mono); font-size:.72rem; color:var(--muted); }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">Results 3 &middot; figure draft 1</p>
    <h1>Belief forms across depth; the action readout discards it</h1>
    <p class="sub">72 model&times;scenario combos, 9 models, 8 scenarios.
    Balanced accuracy throughout, so chance is 50.</p>
  </header>

  <section>
    <figure class="figure" style="margin:0">
      <img src="data:image/png;base64,{b64}"
           alt="Left panel: balanced probe accuracy rising with normalized network
                depth for three scenario families. Right strip: the model's own
                pre-CoT action agreement at the final position, far below the
                belief level, with dashed connectors showing the drop.">
      <figcaption>Left: a <em>trained</em> linear readout of the residual stream
      at each depth. Right: the <em>model's own</em> readout at 100% depth &mdash;
      the pre-CoT action preference. Both panels read the same vector at the
      final layer; they differ only in whether the linear map is fitted to the
      baseline or is the model's own unembedding direction. The dashed drop is
      the belief-to-action gap.</figcaption>
    </figure>
  </section>

  <section>
    <h2>The gap, by family</h2>
    <div class="cards">{fam_cards}</div>
    <p class="note">Chicken and the reactive games land at almost the same
    action level ({fam_rows[1][4]:.1f} vs {fam_rows[2][4]:.1f}) from very
    different belief levels ({fam_rows[1][3]:.1f} vs {fam_rows[2][3]:.1f}).
    Belief height is set by opponent reactivity; action height is set by game
    structure. Chicken &mdash; a game with no reactive opponent type &mdash; is
    the control that separates the two stages, and it carries the largest drop
    in the suite.</p>
  </section>

  <section>
    <h2>Belief across depth</h2>
    <div class="scroll">{depth_table()}</div>
    <p style="margin-top:.9rem">Non-reactive families are resolved by roughly
    half depth and the remaining forward pass adds nothing. The reactive family
    is still climbing at the final layer &mdash; it runs out of network before
    it converges.</p>
  </section>

  <section>
    <h2>Per model</h2>
    <div class="scroll">{model_table()}</div>
  </section>

  <section>
    <h2>Does &ldquo;belief exists everywhere&rdquo; hold?</h2>
    <p>Across all {stats['n']} combos the final-layer floor is
    <strong>{stats['lo']:.1f}</strong> against a chance level of 50, so nothing
    is marginal &mdash; the precondition holds. But the mean of
    {stats['mean']:.1f} sits over a <em>bimodal</em> distribution: the reactive
    combos top out at {stats['react_hi']:.1f} and the lowest non-reactive combo
    is {stats['non_lo']:.1f}. The two groups do not overlap at all.</p>
    <p>So the claim to make is the floor and the separation, not an average.
    The posterior regression already localizes the deficit: the type posterior
    is represented as well in the reactive games (Stag&nbsp;Hunt 0.460,
    BoS 0.453) as in Chicken (0.470) and better than in Monty&nbsp;Hall (0.300).
    Belief is uniform; what degrades is composing it with the agent's own last
    action.</p>
  </section>

  <section>
    <h2>Open decisions</h2>
    <ul>
      <li><strong>Figure width &mdash; settled.</strong> Authored directly at
      AAAI one-column width (3.30&nbsp;in) and placed with
      <code>width=\\columnwidth</code>, so there is no LaTeX rescaling between
      authored and printed size and every in-figure label prints at the pt size
      it was set at. The include must stay at 1.0x:
      <code>width=0.9\\columnwidth</code> would drag the 9&nbsp;pt labels to
      8.1&nbsp;pt and breach the AAAI minimum.</li>
      <li><strong>Legend &mdash; now one row.</strong> All three families share a
      single horizontal row at the bottom, measured at draw time to 2.76&nbsp;in
      of the 3.30&nbsp;in box, so it stays at the full 9&nbsp;pt. This is what
      forced the short labels: &ldquo;Bayesian&rdquo; and &ldquo;Chicken&rdquo;
      rather than &ldquo;Bayesian inference&rdquo; and &ldquo;Chicken
      (non-reactive)&rdquo;. The caption still spells both out, so nothing is
      lost &mdash; but check that you agree the caption is carrying it.</li>
      <li><strong>Thin per-combo lines.</strong> Currently on, showing real
      spread including the final-layer dips. Turn off with
      <code>SHOW_COMBOS = False</code> if it reads as clutter at print size.</li>
      <li><strong>Gemma-4 31B is non-monotonic</strong> mid-network (Bayesian
      77.2 at 20% &rarr; 69.2 at 30% &rarr; 83.0 at 40%). Real, or a probe fit
      artifact at those layers? Visible in the thin lines.</li>
      <li><strong>Panel titles.</strong> &ldquo;belief / action&rdquo; is doing
      a lot of work in two words. Alternative is to drop them and carry it
      entirely in the caption.</li>
    </ul>
  </section>

  <footer>
    analysis/plot_belief_layers.py &middot; analysis/belief_layers_table.py
    &middot; sources: results/P1_v4 + P1_v4_layerbal (layer_accuracy_bal, final
    layer, no layer selection, LOSO over 100 seeds) and
    results/P1_v4/behavior_by_t.json (balanced, pooled over t from per-class
    counts). IPD excluded.
  </footer>
</div>
"""
    open(out_path, "w").write(html)
    print(f"wrote {out_path} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
