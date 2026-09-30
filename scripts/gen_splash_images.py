"""Generate iOS PWA launch images (apple-touch-startup-image) from the BR logo.

iOS shows a blank white screen from icon tap until first HTML paint unless
the page declares per-device startup images. These PNGs mirror the in-page
#appSplash exactly (same BR logo art, same backgrounds: #f8fafc light,
#020617 dark) so the native launch image hands off to the pulsing in-page
splash seamlessly.

The in-page splash logo is 170 CSS px wide, so each launch image draws the
logo at 170 * device-pixel-ratio px wide, centered. Images must be EXACT
device pixels or iOS ignores them.

Run:  python3 scripts/gen_splash_images.py
Writes: static/splash/splash-<W>x<H>.png and static/splash/splash-<W>x<H>-dark.png
Prints: the <link> tags for app.py (kept in sync with
tests/test_pwa_launch_images.py). The hrefs deliberately carry no ?v=
query string: some iOS versions fail to match startup images whose href
has one, and the images are content-stable.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
OUT_DIR = STATIC / "splash"

LIGHT_BG = (248, 250, 252)  # #f8fafc, matches #appSplash background
DARK_BG = (2, 6, 23)  # #020617, matches html[data-theme="dark"] #appSplash
LIGHT_LOGO = STATIC / "BR_Logo.png"
DARK_LOGO = STATIC / "BR_Logo_dark.png"

# (device-width CSS px, device-height CSS px, -webkit-device-pixel-ratio)
SIZES = [
    (430, 932, 3),  # iPhone 14 Pro Max / 15 Plus / 15 Pro Max / 16 Plus
    (393, 852, 3),  # iPhone 14 Pro / 15 / 15 Pro / 16
    (390, 844, 3),  # iPhone 12 / 13 / 14
    (428, 926, 3),  # iPhone 12 Pro Max / 13 Pro Max / 14 Plus
    (414, 896, 3),  # iPhone XS Max / 11 Pro Max
    (375, 812, 3),  # iPhone X / XS / 11 Pro / 12 mini / 13 mini
    (360, 780, 3),  # iPhone 12 mini / 13 mini (alt reporting)
    (414, 896, 2),  # iPhone XR / 11
    (375, 667, 2),  # iPhone 6/7/8 / SE (2nd/3rd gen)
    (414, 736, 3),  # iPhone 6+/7+/8 Plus
    (1024, 1366, 2),  # iPad Pro 12.9"
    (834, 1194, 2),  # iPad Pro 11" / iPad Air (4th/5th gen)
]

LOGO_CSS_WIDTH = 170  # #appSplash img width in app.py


def render(logo: Image.Image, width: int, height: int, ratio: int, bg) -> Image.Image:
    canvas = Image.new("RGB", (width, height), bg)
    logo_w = LOGO_CSS_WIDTH * ratio
    scale = logo_w / logo.width
    logo_h = round(logo.height * scale)
    resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
    x = (width - logo_w) // 2
    y = (height - logo_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    light_logo = Image.open(LIGHT_LOGO).convert("RGBA")
    dark_logo = Image.open(DARK_LOGO).convert("RGBA")
    entries = []
    for css_w, css_h, ratio in SIZES:
        w, h = css_w * ratio, css_h * ratio
        for logo, bg, suffix in (
            (light_logo, LIGHT_BG, ""),
            (dark_logo, DARK_BG, "-dark"),
        ):
            name = f"splash-{w}x{h}{suffix}.png"
            img = render(logo, w, h, ratio, bg)
            img.save(OUT_DIR / name, optimize=True)
            entries.append((css_w, css_h, ratio, suffix, name))
            print(f"wrote static/splash/{name} ({w}x{h})")
    print("\nLink tags for app.py:\n")
    for css_w, css_h, ratio, suffix, name in entries:
        scheme = "dark" if suffix else "light"
        media = (
            f"(prefers-color-scheme: {scheme}) and "
            f"(device-width: {css_w}px) and (device-height: {css_h}px) "
            f"and (-webkit-device-pixel-ratio: {ratio}) and (orientation: portrait)"
        )
        print(f'<link rel="apple-touch-startup-image" media="{media}" '
              f'href="/static/splash/{name}">')


if __name__ == "__main__":
    main()
