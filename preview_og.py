"""
Run from the fantasy-dashboard directory:
    python3 preview_og.py
Opens /tmp/og_preview.png when done.
"""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(BASE, "static", "fonts")

def _font(name, size):
    try:
        return ImageFont.truetype(os.path.join(font_dir, name), size)
    except Exception:
        return ImageFont.load_default()

f_huge  = _font("Inter-Bold.ttf",     78)
f_large = _font("Inter-Bold.ttf",     38)
f_med   = _font("Inter-SemiBold.ttf", 28)
f_small = _font("Inter-Regular.ttf",  24)
f_label = _font("Inter-Regular.ttf",  20)
f_pill  = _font("Inter-SemiBold.ttf", 20)

C_BG     = (2,   6,   23)
C_SURFACE= (15,  23,  42)
C_ACCENT = (56,  189, 248)
C_TEXT   = (229, 231, 235)
C_MUTED  = (148, 163, 184)
C_SUBTLE = (71,  85,  105)
C_GREEN  = (34,  197, 94)
C_RED    = (239, 68,  68)
C_BORDER = (31,  41,  55)

# ── Edit these to preview different data ──────────────────────────────────────
week_label  = "Week 7 Recap"
league_name = "The Gridiron Gang"
season      = 2024
high_name, high_pts = "Dynasty Kings",     148.32
low_name,  low_pts  = "Pocket Protectors", 71.80
matchups = [
    {"w": "Dynasty Kings",     "w_pts": 148.3, "l": "Gridiron Ghosts",   "l_pts": 85.9,  "margin": 62.4},
    {"w": "Blitz Brigade",     "w_pts": 131.7, "l": "Endzone Elite",     "l_pts": 104.2, "margin": 27.5},
    {"w": "Redzone Rebels",    "w_pts": 119.8, "l": "Pocket Protectors", "l_pts": 71.8,  "margin": 48.0},
    {"w": "Super Cena 09",     "w_pts": 112.4, "l": "09 Orton Fan Club", "l_pts": 98.1,  "margin": 14.3},
    {"w": "Thunderdome",       "w_pts": 108.9, "l": "Gridiron Ghosts",   "l_pts": 95.3,  "margin": 13.6},
]
# ─────────────────────────────────────────────────────────────────────────────

W, H = 1200, 630
img  = Image.new("RGB", (W, H), color=C_BG)
draw = ImageDraw.Draw(img)

for x in range(0, W, 64):
    draw.line([(x, 0), (x, H)], fill=(15, 23, 42), width=1)
for y in range(0, H, 64):
    draw.line([(0, y), (W, y)], fill=(15, 23, 42), width=1)

draw.rectangle([0, 0, W, 5], fill=C_ACCENT)

PAD = 72

try:
    logo = Image.open(os.path.join(BASE, "static", "Website_Logo_dark.png")).convert("RGBA")
    logo_h = 90
    logo_w = int(logo.width * logo_h / logo.height)
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
    img.paste(logo, (PAD, 38), logo)
except Exception as e:
    draw.text((PAD, 58), "BR Fantasy", fill=C_ACCENT, font=f_pill)
    print(f"Logo load failed: {e}")

draw.text((PAD, 152), week_label,  fill=C_TEXT,  font=f_huge)
draw.text((PAD, 256), league_name, fill=C_MUTED, font=f_large)
draw.rounded_rectangle([PAD, 314, PAD + 64, 318], radius=2, fill=C_ACCENT)

def stat_row(y, label, name, value, color):
    draw.text((PAD, y),       label,     fill=C_SUBTLE, font=f_label)
    draw.text((PAD, y + 24),  name[:22], fill=C_TEXT,   font=f_med)
    bb = draw.textbbox((0, 0), value, font=f_med)
    draw.text((580 - (bb[2] - bb[0]), y + 24), value, fill=color, font=f_med)

stat_row(336,       "HIGH SCORE",  high_name, f"{high_pts:.2f}", C_GREEN)
if matchups:
    best = matchups[0]
    stat_row(336 + 76,  "BIGGEST WIN", best["w"], f"+{best['margin']:.1f}", C_ACCENT)
stat_row(336 + 152, "LOW SCORE",   low_name,  f"{low_pts:.2f}",  C_RED)

CX, CY, CW, CH = 680, 56, 456, 510
draw.rounded_rectangle([CX, CY, CX + CW, CY + CH], radius=16, fill=C_SURFACE)
draw.text((CX + 24, CY + 20), "Scoreboard", fill=C_TEXT,  font=f_med)
draw.text((CX + 24, CY + 52), f"Week {week_label.split()[1] if 'Week' in week_label else ''}", fill=C_MUTED, font=f_label)
draw.rectangle([CX, CY + 80, CX + CW, CY + 81], fill=C_BORDER)

ry, row_h = CY + 94, 82
for i, m in enumerate(matchups[:5]):
    if ry + row_h > CY + CH - 10:
        break
    draw.text((CX + 20, ry),      m["w"][:18], fill=C_TEXT,  font=f_small)
    draw.text((CX + 20, ry + 28), m["l"][:18], fill=C_MUTED, font=f_small)
    for val, col, yoff in [(f"{m['w_pts']:.1f}", C_GREEN, 0), (f"{m['l_pts']:.1f}", C_MUTED, 28)]:
        bb = draw.textbbox((0, 0), val, font=f_small)
        draw.text((CX + CW - 20 - (bb[2]-bb[0]), ry + yoff), val, fill=col, font=f_small)
    draw.text((CX + 20, ry + 54), f"+{m['margin']:.1f}", fill=C_ACCENT, font=f_label)
    ry += row_h
    if i < len(matchups) - 1 and ry < CY + CH - 10:
        draw.rectangle([CX + 16, ry - 4, CX + CW - 16, ry - 3], fill=C_BORDER)

draw.rectangle([0, H - 50, W, H], fill=C_SURFACE)
draw.text((PAD, H - 32), "brfantasy.com", fill=C_MUTED, font=f_label)
draw.text((PAD + 170, H - 32), "·  AI-powered weekly recap", fill=C_SUBTLE, font=f_label)
season_str = f"Season {season}"
bb = draw.textbbox((0, 0), season_str, font=f_label)
draw.text((W - PAD - (bb[2]-bb[0]), H - 32), season_str, fill=C_MUTED, font=f_label)

out = "/tmp/og_preview.png"
img.save(out)
print(f"Saved → {out}")
subprocess.run(["open", out])
