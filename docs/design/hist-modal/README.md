# History modal — UX rework mockups

Source artboards for the draft cheat sheet's **Hist** modal rework. Each
`*.dc.html` is one artboard; `canvas.json` lays them out on a design canvas.
Colors use the app's real dark-theme tokens (`--card`, `--border`, position
accents) so they map straight onto the live modal. Player numbers are
illustrative placeholders — the layouts are the deliverable.

| File | Direction | Idea |
| --- | --- | --- |
| `Current.dc.html` | Baseline | What ships today, for contrast. |
| `Main.dc.html` | A · Verdict-first | Plain-English verdict + one "profile vs. price" bar with an edge callout; comps/trends collapse away. |
| `Scannable.dc.html` | B · Scannable | Everything on one screen — hero, finish ladder, diverging price bar, inline trends. |
| `Guided.dc.html` | C · Guided cards | Friendliest / mobile — titled, iconed cards with big touch targets. |

Shared fixes: a real verdict on top, one comparison instead of six competing
numbers, and progressive disclosure for comps / trends / profile.

The live modal is rendered by `renderHistPanel` in `static/cheat_sheet.js`;
its styles live in `dashboard_services/pages/cheat_sheet_page.py`.
