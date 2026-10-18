#!/usr/bin/env python3
"""Handwritten digit collector.

Shows a target digit (0-9) and a 24x24 grid. Draw the digit with the mouse,
save it, and you get the next one. Images are saved as 24x24 grayscale PNGs
(white ink on black, MNIST-style) into dataset/<digit>/.

Controls:
  Left mouse      draw
  Right mouse     erase
  Enter / Space   save and go to next digit
  Backspace / C   clear grid
  S               skip this digit (no save)
  B               toggle soft brush (grayscale edges) / hard pixel brush
  Ctrl+Z          undo last save
  Esc             quit
"""

import argparse
import random
import time
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image

GRID = 24
CELL = 22  # on-screen size of one grid cell in pixels
CANVAS = GRID * CELL


class DigitCollector:
    def __init__(self, root: tk.Tk, out_dir: Path, mode: str):
        self.root = root
        self.out_dir = out_dir
        self.mode = mode  # "balanced" or "random" or "sequential"
        self.pixels = np.zeros((GRID, GRID), dtype=np.float32)
        self.soft = True
        self.last_saved: list[Path] = []
        self.seq_digit = -1

        for d in range(10):
            (out_dir / str(d)).mkdir(parents=True, exist_ok=True)

        root.title("Digit Collector")
        root.configure(bg="#111")

        top = tk.Frame(root, bg="#111")
        top.pack(padx=12, pady=(12, 4), fill="x")
        tk.Label(top, text="Draw:", font=("Helvetica", 20), fg="#aaa", bg="#111").pack(side="left")
        self.target_label = tk.Label(top, text="", font=("Helvetica", 48, "bold"), fg="#4fc3f7", bg="#111")
        self.target_label.pack(side="left", padx=10)
        self.brush_label = tk.Label(top, text="", font=("Helvetica", 11), fg="#888", bg="#111")
        self.brush_label.pack(side="right")

        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(padx=12, pady=4)
        self.rects = [[self.canvas.create_rectangle(
            x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL,
            fill="#000", outline="#1a1a1a") for x in range(GRID)] for y in range(GRID)]

        self.count_label = tk.Label(root, text="", font=("Courier", 11), fg="#ccc", bg="#111", justify="left")
        self.count_label.pack(padx=12, pady=(4, 2))
        tk.Label(root, text="LMB draw · RMB erase · Enter/Space save · Backspace clear · "
                            "S skip · B brush · Ctrl+Z undo",
                 font=("Helvetica", 9), fg="#666", bg="#111").pack(padx=12, pady=(0, 10))

        self.canvas.bind("<B1-Motion>", lambda e: self.paint(e, 1.0))
        self.canvas.bind("<Button-1>", lambda e: self.paint(e, 1.0))
        self.canvas.bind("<B3-Motion>", lambda e: self.paint(e, 0.0))
        self.canvas.bind("<Button-3>", lambda e: self.paint(e, 0.0))
        root.bind("<Return>", lambda e: self.save())
        root.bind("<space>", lambda e: self.save())
        root.bind("<BackSpace>", lambda e: self.clear())
        root.bind("c", lambda e: self.clear())
        root.bind("s", lambda e: self.next_digit())
        root.bind("b", lambda e: self.toggle_brush())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Escape>", lambda e: root.destroy())

        self.update_brush_label()
        self.next_digit()

    # ---- dataset bookkeeping -------------------------------------------
    def counts(self) -> list[int]:
        return [len(list((self.out_dir / str(d)).glob("*.png"))) for d in range(10)]

    def pick_digit(self) -> int:
        if self.mode == "sequential":
            self.seq_digit = (self.seq_digit + 1) % 10
            return self.seq_digit
        if self.mode == "random":
            return random.randint(0, 9)
        c = self.counts()
        lowest = min(c)
        return random.choice([d for d in range(10) if c[d] == lowest])

    def refresh_counts(self):
        c = self.counts()
        row1 = "  ".join(f"{d}:{c[d]:>4}" for d in range(5))
        row2 = "  ".join(f"{d}:{c[d]:>4}" for d in range(5, 10))
        self.count_label.config(text=f"{row1}\n{row2}\ntotal: {sum(c)}")

    # ---- drawing -------------------------------------------------------
    def paint(self, event, value: float):
        gx, gy = event.x / CELL, event.y / CELL
        cx, cy = int(gx), int(gy)
        if not (0 <= cx < GRID and 0 <= cy < GRID):
            return
        if value == 0.0 or not self.soft:
            self.set_px(cx, cy, value)
            return
        # Soft brush: full ink at the cell under the cursor, falloff on neighbours.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < GRID and 0 <= y < GRID:
                    dist = np.hypot(x + 0.5 - gx, y + 0.5 - gy)
                    ink = max(0.0, 1.0 - max(0.0, dist - 0.5) / 1.0)
                    if ink > self.pixels[y, x]:
                        self.set_px(x, y, ink)

    def set_px(self, x: int, y: int, v: float):
        self.pixels[y, x] = v
        g = int(v * 255)
        self.canvas.itemconfig(self.rects[y][x], fill=f"#{g:02x}{g:02x}{g:02x}")

    def clear(self):
        self.pixels[:] = 0
        for y in range(GRID):
            for x in range(GRID):
                self.canvas.itemconfig(self.rects[y][x], fill="#000")

    def toggle_brush(self):
        self.soft = not self.soft
        self.update_brush_label()

    def update_brush_label(self):
        self.brush_label.config(text=f"brush: {'soft' if self.soft else 'hard'} (B)")

    # ---- save / navigation --------------------------------------------
    def next_digit(self):
        self.target = self.pick_digit()
        self.target_label.config(text=str(self.target))
        self.clear()
        self.refresh_counts()

    def save(self):
        if self.pixels.max() == 0:
            self.flash("#e57373")  # empty grid, refuse to save
            return
        img = Image.fromarray((self.pixels * 255).astype(np.uint8))
        name = f"{self.target}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
        path = self.out_dir / str(self.target) / name
        img.save(path)
        self.last_saved.append(path)
        self.flash("#81c784")
        self.next_digit()

    def undo(self):
        if self.last_saved:
            path = self.last_saved.pop()
            path.unlink(missing_ok=True)
            self.flash("#ffb74d")
            self.refresh_counts()

    def flash(self, color: str):
        self.canvas.config(highlightbackground=color, highlightthickness=3)
        self.root.after(200, lambda: self.canvas.config(highlightbackground="#444", highlightthickness=1))


def main():
    parser = argparse.ArgumentParser(description="Collect 24x24 handwritten digits.")
    parser.add_argument("-o", "--out", default=str(Path(__file__).parent / "dataset"),
                        help="output folder (default: ./dataset)")
    parser.add_argument("-m", "--mode", choices=["balanced", "random", "sequential"], default="balanced",
                        help="balanced = always ask for the digit with the fewest samples (default)")
    args = parser.parse_args()

    root = tk.Tk()
    root.resizable(False, False)
    DigitCollector(root, Path(args.out), args.mode)
    root.mainloop()


if __name__ == "__main__":
    main()
