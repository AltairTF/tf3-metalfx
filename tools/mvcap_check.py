#!/usr/bin/env python3
"""Check recorded motion vectors (mvcap folder) without RenderDoc or numpy (pure stdlib).

For each pair of consecutive recorded frames (A = frame N, B = frame N+1) it warps A's colour into
B using B's motion vectors (mv points from B's pixel to where it was in A) and compares:
  err_zero  mean |B - A|                  (no motion compensation)
  err_mv    mean |B - A warped by mv|     (with the layer's motion vectors)
Correct vectors: err_mv clearly below err_zero whenever the camera moved.
It also writes PNGs per frame: colour (R11G11B10 float, the upscaler input), motion (grey = still, red/cyan = x, green/magenta = y), depth.

  python3 tools/mvcap_check.py <mvcap folder> [--png]
"""
import json
import os
import struct
import sys
import zlib
from array import array


def load_u32(path):
    a = array("I")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def half(u16):
    return struct.unpack("<e", struct.pack("<H", u16))[0]


HALF = [half(i) for i in range(65536)]


def decode_mv(words):
    xs = array("f", (HALF[w & 0xFFFF] for w in words))
    ys = array("f", (HALF[w >> 16] for w in words))
    return xs, ys


def uf(bits, mant):
    """Unsigned small float (B10G11R11 packed): 5-bit exponent, `mant`-bit mantissa."""
    e, m = bits >> mant, bits & ((1 << mant) - 1)
    if e == 0:
        return m / (1 << mant) * 2.0 ** -14
    if e == 31:
        return 65504.0
    return (1.0 + m / (1 << mant)) * 2.0 ** (e - 15)


UF11 = [uf(i, 6) for i in range(2048)]
UF10 = [uf(i, 5) for i in range(1024)]


def rgb_r11g11b10(w):
    return UF11[w & 0x7FF], UF11[(w >> 11) & 0x7FF], UF10[w >> 22]


def luma_r11g11b10(words):
    out = array("f")
    for w in words:
        r, g, b = rgb_r11g11b10(w)
        out.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return out


def write_png(path, w, h, rgb_rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rgb_rows)
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def load_frame(folder, base):
    meta = json.load(open(os.path.join(folder, base + ".json")))
    w, h = meta["file_w"], meta["file_h"]
    color = load_u32(os.path.join(folder, base + ".color.bin"))
    mvx, mvy = decode_mv(load_u32(os.path.join(folder, base + ".mvfinal.bin")))
    depth = array("f")
    with open(os.path.join(folder, base + ".depth.bin"), "rb") as f:
        depth.frombytes(f.read())
    return meta, w, h, color, luma_r11g11b10(color), mvx, mvy, depth


def sample(lum, w, h, x, y):
    x = min(max(x, 0.0), w - 1.001)
    y = min(max(y, 0.0), h - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    i = y0 * w + x0
    a = lum[i] + (lum[i + 1] - lum[i]) * fx
    b = lum[i + w] + (lum[i + w + 1] - lum[i + w]) * fx
    return a + (b - a) * fy


def main():
    folder = sys.argv[1]
    png = "--png" in sys.argv
    bases = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".json"))
    frames = {}
    for b in bases:
        frames[b] = load_frame(folder, b)
        meta, w, h, color, lum, mvx, mvy, depth = frames[b]
        n = len(mvx)
        moving = sum(1 for i in range(n) if abs(mvx[i]) + abs(mvy[i]) > 0.25)
        mean = sum(abs(mvx[i]) + abs(mvy[i]) for i in range(n)) / n
        print(f"{b}: frame {meta['frame']} jitter={meta['jitter_px']} mv_enabled={meta['mv_enabled']} "
              f"mean|mv|={mean:.3f}px(render) moving(>0.25px)={100.0 * moving / n:.1f}%")
        if png:
            step = meta["file_step"]
            rows, mrows, drows = [], [], []
            for y in range(h):
                r, m, d = bytearray(), bytearray(), bytearray()
                for x in range(w):
                    cr, cg, cb = rgb_r11g11b10(color[y * w + x])
                    r += bytes(int(min(max(v, 0.0), 1.0) ** (1 / 2.2) * 255) for v in (cr, cg, cb))
                    vx, vy = mvx[y * w + x] * 0.05, mvy[y * w + x] * 0.05
                    m += bytes((int(min(max(128 + 127 * vx, 0), 255)), int(min(max(128 + 127 * vy, 0), 255)),
                                int(min(max(128 - 64 * (vx + vy), 0), 255))))
                    dv = int(min(max(depth[y * w + x], 0.0), 1.0) ** 32 * 255)
                    d += bytes((dv, dv, dv))
                rows.append(r); mrows.append(m); drows.append(d)
            write_png(os.path.join(folder, b + ".color.png"), w, h, rows)
            write_png(os.path.join(folder, b + ".mv.png"), w, h, mrows)
            write_png(os.path.join(folder, b + ".depth.png"), w, h, drows)
    # consecutive pairs: B.frame == A.frame + 1
    by_frame = {frames[b][0]["frame"]: b for b in bases}
    for f, b in sorted(by_frame.items()):
        a = by_frame.get(f - 1)
        if not a:
            continue
        metaA, w, h, _, lumA, mvxA, mvyA, _ = frames[a]
        meta, _, _, _, lumB, mvx, mvy, _ = frames[b]
        step = meta["file_step"]  # files keep every other render pixel; mv is in render pixels
        # display-like luma (tone-mapped HDR clamped, gamma) so bright pixels don't dominate
        gA = [min(max(v, 0.0), 1.0) ** (1 / 2.2) for v in lumA]
        gB = [min(max(v, 0.0), 1.0) ** (1 / 2.2) for v in lumB]
        jA, jB = metaA["jitter_px"], meta["jitter_px"]
        # Only textured pixels: flat ground matches anywhere and hides wrong vectors.
        pts = []
        for y in range(12, h - 12, 2):
            for x in range(12, w - 12, 2):
                i = y * w + x
                if abs(gB[i + 1] - gB[i - 1]) + abs(gB[i + w] - gB[i - w]) > 0.06:
                    pts.append(i)
        if len(pts) < 200:
            print(f"pair {a} -> {b}: too little texture ({len(pts)} px)")
            continue

        def err(fx, fy, jit):
            # B(p) shows the point whose unjittered position is p - jB; it was at (p - jB + mv) unjittered
            # in A, i.e. at p - jB + mv + jA in A's jittered image (jit = sign of that correction).
            e = 0.0
            for i in pts:
                x, y = i % w, i // w
                sx = x + (fx(i, 0) + jit * (jA[0] - jB[0])) / step
                sy = y + (fy(i, 1) + jit * (jA[1] - jB[1])) / step
                e += abs(gB[i] - sample(gA, w, h, sx, sy))
            return e / len(pts)

        zero = lambda i, k: 0.0
        own = lambda i, k: (mvx if k == 0 else mvy)[i]
        lag = lambda i, k: (mvxA if k == 0 else mvyA)[i]  # A's vectors: what a one-frame lag would give
        half_ = lambda i, k: 0.5 * own(i, k)
        double_ = lambda i, k: 2.0 * own(i, k)
        res = {"zero": err(zero, zero, 1), "mv": err(own, own, 1), "mv_nojit": err(own, own, 0),
               "mv_-jit": err(own, own, -1), "lagA": err(lag, lag, 1), "mv*0.5": err(half_, half_, 1),
               "mv*2": err(double_, double_, 1)}
        best = min(res, key=res.get)
        verdict = "GOOD" if res["mv"] < 0.7 * res["zero"] and best in ("mv", "mv_nojit", "mv_-jit") else \
                  ("still" if res["zero"] < 0.01 else "BAD")
        print(f"pair {a} -> {b} ({len(pts)} px): " + " ".join(f"{k}={v:.4f}" for k, v in res.items()) +
              f"  best={best} {verdict}")


if __name__ == "__main__":
    main()
