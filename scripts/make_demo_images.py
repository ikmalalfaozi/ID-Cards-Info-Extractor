"""Membuat gambar contoh SINTETIS di demo_input/ (tiga dokumen fiktif, difoto miring di atas latar).

Bukan dokumen resmi dan tidak meniru fitur keamanannya: semua data fiktif, tiap kartu bertanda "SPESIMEN / DATA FIKTIF",
tanpa lambang negara, foto berupa siluet. Hasilnya deterministik (seed tetap) sehingga bisa dibuat ulang:

    python scripts/make_demo_images.py            # menulis demo_input/contoh_{ktp,sim,paspor}.jpg
    python scripts/make_demo_images.py --out DIR
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARD_W, CARD_H = 1012, 638          # rasio kartu ID-1 (85,6 x 54 mm), dipakai KTP dan SIM
PASSPORT_W, PASSPORT_H = 1000, 704  # halaman data paspor (125 x 88 mm)

CARDS = {
    "ktp": dict(
        size=(CARD_W, CARD_H), bg=(214, 232, 247), band=(33, 91, 163),
        header=["PROVINSI CONTOH", "KOTA SPESIMEN"], title="KARTU TANDA PENDUDUK",
        top=("NIK", "9999 0000 0000 0001"),
        rows=[("Nama", "BUDI CONTOH"), ("Tempat/Tgl Lahir", "KOTA SPESIMEN, 01-01-1990"),
              ("Jenis Kelamin", "LAKI-LAKI        Gol. Darah : O"), ("Alamat", "JL. CONTOH NO. 1"),
              ("RT/RW", "001/002"), ("Kel/Desa", "DESA FIKTIF"), ("Kecamatan", "KEC. SPESIMEN"),
              ("Agama", "ISLAM"), ("Status Perkawinan", "BELUM KAWIN"), ("Pekerjaan", "PELAJAR/MAHASISWA"),
              ("Kewarganegaraan", "WNI"), ("Berlaku Hingga", "SEUMUR HIDUP")],
        photo=(700, 150, 960, 470)),
    "sim": dict(
        size=(CARD_W, CARD_H), bg=(244, 226, 226), band=(176, 48, 56),
        header=["REPUBLIK CONTOH"], title="SURAT IZIN MENGEMUDI",
        top=("No. SIM", "9999-0000-000001"),
        rows=[("Nama", "SITI SPESIMEN"), ("Tempat/Tgl Lahir", "KOTA FIKTIF, 02-02-1992"),
              ("Tinggi", "160"), ("Jenis Kelamin", "PEREMPUAN"), ("Alamat", "JL. CONTOH NO. 2, KOTA FIKTIF"),
              ("Pekerjaan", "KARYAWAN SWASTA"), ("Gol. SIM", "A"), ("Berlaku s/d", "02-02-2030")],
        photo=(700, 170, 960, 490)),
    "paspor": dict(
        size=(PASSPORT_W, PASSPORT_H), bg=(232, 240, 226), band=(54, 105, 62),
        header=["REPUBLIK CONTOH"], title="PASPOR  /  PASSPORT",
        top=("No. Paspor", "X 0000001"),
        rows=[("Jenis / Type", "P"), ("Kode Negara", "CTH"), ("Nama", "ANDI SAMPLE"),
              ("Kewarganegaraan", "CONTOH"), ("Tgl Lahir", "03 MAR 1988"), ("Jenis Kelamin", "L / M"),
              ("Tempat Lahir", "KOTA SPESIMEN"), ("Tgl Pengeluaran", "01 JAN 2023"),
              ("Tgl Habis Berlaku", "01 JAN 2033"), ("No. Registrasi", "X-0000-000001"),
              ("Kantor Penerbit", "KANTOR FIKTIF")],
        photo=(70, 190, 330, 530)),
}


def font(size, bold=False):
    """DejaVu bila ada (Linux, matplotlib); jika tidak, font bawaan Pillow."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [name]
    try:
        import matplotlib
        candidates.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name))
    except ImportError:
        pass
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_document(spec):
    w, h = spec["size"]
    img = Image.new("RGB", (w, h), spec["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 120], fill=spec["band"])
    y = 14
    for line in spec["header"]:
        d.text((w // 2, y), line, font=font(24, True), fill="white", anchor="ma")
        y += 30
    d.text((w // 2, 120 - 34), spec["title"], font=font(26, True), fill="white", anchor="ma")

    x0 = 40 if spec["photo"][0] > 400 else 360
    label, value = spec["top"]
    d.text((x0, 140), label, font=font(22), fill=(40, 40, 40))
    d.text((x0 + 190, 136), value, font=font(30, True), fill=(10, 10, 10))
    y = 190
    step = min(36, (h - y - 50) // len(spec["rows"]))
    for label, value in spec["rows"]:
        d.text((x0, y), label, font=font(20), fill=(60, 60, 60))
        d.text((x0 + 250, y), f": {value}", font=font(20, True), fill=(10, 10, 10))
        y += step

    px0, py0, px1, py1 = spec["photo"]                       # foto: siluet abu-abu, bukan wajah
    d.rectangle([px0, py0, px1, py1], fill=(200, 205, 212), outline=(120, 120, 120), width=3)
    cx, cy = (px0 + px1) // 2, py0 + (py1 - py0) // 3
    r = (px1 - px0) // 5
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(140, 145, 152))
    d.ellipse([cx - 2 * r, cy + r + 10, cx + 2 * r, cy + 5 * r], fill=(140, 145, 152))
    d.rectangle([px0 + 3, py1 - 36, px1 - 3, py1 - 3], fill=spec["band"])
    d.text(((px0 + px1) // 2, py1 - 20), "FOTO FIKTIF", font=font(16, True), fill="white", anchor="mm")

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))          # tanda air diagonal
    ld = ImageDraw.Draw(layer)
    ld.text((w // 2, h // 2), "SPESIMEN / DATA FIKTIF", font=font(64, True), fill=(150, 0, 0, 70), anchor="mm")
    layer = layer.rotate(25, resample=Image.BICUBIC)
    img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    ImageDraw.Draw(img).rectangle([0, 0, w - 1, h - 1], outline=(90, 90, 90), width=4)
    return np.array(img)


def photograph(doc, seed, angle, skew):
    """Letakkan dokumen miring (perspektif) di atas latar meja bertekstur, lalu beri sedikit blur dan noise."""
    rng = np.random.default_rng(seed)
    out_w, out_h = 1280, 960
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    wood = 120 + 25 * np.sin(yy / 9.0 + 6 * np.sin(xx / 300.0)) + rng.normal(0, 6, (out_h, out_w))
    bg = np.stack([wood * 0.95, wood * 0.72, wood * 0.5], axis=-1).clip(0, 255).astype(np.uint8)

    h, w = doc.shape[:2]
    scale = 0.7 * out_w / w
    cw, ch = w * scale, h * scale
    c = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]])
    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    dst = c @ rot.T + np.array([out_w / 2, out_h / 2])
    dst += np.array([[skew, 0], [-skew, 0], [skew * 0.6, 0], [-skew * 0.6, 0]]) * 3
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(doc, M, (out_w, out_h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (out_w, out_h), flags=cv2.INTER_NEAREST)
    shadow = cv2.GaussianBlur(np.roll(mask, (14, 18), axis=(0, 1)), (0, 0), 14)[..., None] / 255.0 * 0.45
    img = bg * (1 - shadow) + 0.0
    img = np.where(mask[..., None] > 0, warped, img)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8) + rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "demo_input"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    poses = {"ktp": (0, -9, 22), "sim": (1, 7, -18), "paspor": (2, -4, 14)}
    for name, spec in CARDS.items():
        seed, angle, skew = poses[name]
        image = photograph(draw_document(spec), seed, angle, skew)
        path = os.path.join(args.out, f"contoh_{name}.jpg")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


if __name__ == "__main__":
    main()
