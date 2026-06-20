"""
src/gui.py

Tkinter-based GUI for the Daraz ABSA project.
No third-party libraries required — uses only Python's built-in tkinter.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, font
import threading
from collections import Counter


class DarazABSAGui:

    # ── Colour palette ───────────────────────────────────────────────────────
    BG          = "#F5F4F0"
    CARD        = "#FFFFFF"
    BORDER      = "#DDDBD3"
    PRIMARY     = "#1A1A1A"
    MUTED       = "#6B6A64"
    ACCENT      = "#185FA5"
    BTN_BG      = "#1A1A1A"
    BTN_FG      = "#FFFFFF"
    BTN_HOV     = "#333333"
    POS_BG      = "#EAF3DE"
    POS_FG      = "#27500A"
    POS_BORDER  = "#639922"
    NEG_BG      = "#FCEBEB"
    NEG_FG      = "#791F1F"
    NEG_BORDER  = "#E24B4A"
    MIX_BG      = "#FAEEDA"
    MIX_FG      = "#633806"
    MIX_BORDER  = "#BA7517"
    TAG_BG      = "#E6F1FB"
    TAG_FG      = "#185FA5"

    def __init__(self, root, model, preprocessor, crf_tagger,
                 build_clause_contexts, contrast_conjunctions):
        self.root                  = root
        self.model                 = model
        self.preprocessor          = preprocessor
        self.crf_tagger            = crf_tagger
        self._build_clause_contexts = build_clause_contexts
        self.window_size           = 5

        self.root.title("Daraz ABSA — Aspect-Based Sentiment Analyzer")
        self.root.geometry("860x700")
        self.root.minsize(700, 560)
        self.root.configure(bg=self.BG)

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.CARD,
                       highlightthickness=1,
                       highlightbackground=self.BORDER)
        hdr.pack(fill="x")

        inner = tk.Frame(hdr, bg=self.CARD, pady=14, padx=20)
        inner.pack(fill="x")

        tk.Label(inner, text="Daraz ABSA",
                 font=("Helvetica", 17, "bold"),
                 bg=self.CARD, fg=self.PRIMARY).pack(side="left")

        tk.Label(inner,
                 text="  Aspect-Based Sentiment Analysis · Roman Urdu",
                 font=("Helvetica", 11),
                 bg=self.CARD, fg=self.MUTED).pack(side="left")

        # Model badge
        badge_frame = tk.Frame(inner, bg=self.TAG_BG,
                               padx=10, pady=4,
                               highlightthickness=1,
                               highlightbackground=self.ACCENT)
        badge_frame.pack(side="right")
        tk.Label(badge_frame,
                 text="Logistic Regression",
                 font=("Helvetica", 9, "bold"),
                 bg=self.TAG_BG, fg=self.TAG_FG).pack()

    def _build_body(self):
        body = tk.Frame(self.root, bg=self.BG, padx=18, pady=14)
        body.pack(fill="both", expand=True)

        self._build_input_card(body)
        self._build_verdict_area(body)
        self._build_results_area(body)

    def _build_input_card(self, parent):
        card = tk.Frame(parent, bg=self.CARD,
                        highlightthickness=1,
                        highlightbackground=self.BORDER,
                        padx=16, pady=14)
        card.pack(fill="x", pady=(0, 10))

        tk.Label(card, text="Enter a Daraz review",
                 font=("Helvetica", 12, "bold"),
                 bg=self.CARD, fg=self.PRIMARY).pack(anchor="w")
        tk.Label(card,
                 text="Roman Urdu, English, or code-mixed text supported",
                 font=("Helvetica", 10),
                 bg=self.CARD, fg=self.MUTED).pack(anchor="w", pady=(2, 10))

        # Input row
        row = tk.Frame(card, bg=self.CARD)
        row.pack(fill="x")

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(row,
                              textvariable=self.entry_var,
                              font=("Helvetica", 12),
                              relief="solid", bd=1,
                              highlightthickness=2,
                              highlightcolor=self.ACCENT,
                              highlightbackground=self.BORDER,
                              fg=self.PRIMARY, bg=self.CARD,
                              insertbackground=self.PRIMARY)
        self.entry.pack(side="left", fill="x", expand=True, ipady=9)
        self.entry.bind("<Return>", lambda e: self._analyze())
        self.entry.focus()

        self.btn = tk.Button(row,
                             text="Analyze  →",
                             font=("Helvetica", 11, "bold"),
                             bg=self.BTN_BG, fg=self.BTN_FG,
                             relief="flat",
                             padx=18, pady=9,
                             cursor="hand2",
                             activebackground=self.BTN_HOV,
                             activeforeground=self.BTN_FG,
                             command=self._analyze)
        self.btn.pack(side="left", padx=(8, 0))

        # Example chips
        chip_row = tk.Frame(card, bg=self.CARD)
        chip_row.pack(fill="x", pady=(10, 0))
        tk.Label(chip_row, text="Try:",
                 font=("Helvetica", 10),
                 bg=self.CARD, fg=self.MUTED).pack(side="left")

        examples = [
            "quality achi hai lekin delivery late thi",
            "screen zabardast hai magar battery bekar hai",
            "charger slow hai aur cable bhi weak hai",
        ]
        for ex in examples:
            b = tk.Button(chip_row, text=ex,
                          font=("Helvetica", 9),
                          bg="#EEF4FB", fg=self.ACCENT,
                          relief="flat", padx=8, pady=3,
                          cursor="hand2",
                          activebackground="#D6E8F7",
                          command=lambda t=ex: self._set_example(t))
            b.pack(side="left", padx=(6, 0))

    def _build_verdict_area(self, parent):
        self.verdict_outer = tk.Frame(parent, bg=self.BG)
        self.verdict_outer.pack(fill="x", pady=(0, 8))

    def _build_results_area(self, parent):
        tk.Label(parent, text="Analysis Results",
                 font=("Helvetica", 11, "bold"),
                 bg=self.BG, fg=self.PRIMARY).pack(anchor="w", pady=(0, 6))

        res_card = tk.Frame(parent, bg=self.CARD,
                            highlightthickness=1,
                            highlightbackground=self.BORDER)
        res_card.pack(fill="both", expand=True)

        self.results_text = scrolledtext.ScrolledText(
            res_card,
            font=("Courier", 11),
            bg=self.CARD, fg=self.PRIMARY,
            relief="flat", bd=0,
            padx=16, pady=14,
            state="disabled",
            wrap="word",
            cursor="arrow",
            spacing1=2, spacing3=2,
        )
        self.results_text.pack(fill="both", expand=True)

        # Text tags
        self.results_text.tag_config("pos_line",
            foreground=self.POS_FG, background=self.POS_BG,
            font=("Courier", 11, "bold"))
        self.results_text.tag_config("neg_line",
            foreground=self.NEG_FG, background=self.NEG_BG,
            font=("Courier", 11, "bold"))
        self.results_text.tag_config("aspect_name",
            foreground=self.ACCENT,
            font=("Courier", 12, "bold"))
        self.results_text.tag_config("context_line",
            foreground=self.MUTED,
            font=("Courier", 10))
        self.results_text.tag_config("header",
            foreground=self.PRIMARY,
            font=("Helvetica", 11, "bold"))
        self.results_text.tag_config("muted",
            foreground=self.MUTED,
            font=("Helvetica", 10))
        self.results_text.tag_config("divider",
            foreground=self.BORDER)

        self._show_placeholder()

    def _build_statusbar(self):
        tk.Frame(self.root, bg=self.BORDER, height=1).pack(fill="x")
        self.status_var = tk.StringVar(
            value="Ready  ·  CRF NER loaded  ·  Logistic Regression loaded")
        tk.Label(self.root,
                 textvariable=self.status_var,
                 font=("Helvetica", 9),
                 bg="#EEECEA", fg=self.MUTED,
                 pady=6, padx=16,
                 anchor="w").pack(fill="x")

    # ── State helpers ────────────────────────────────────────────────────────

    def _show_placeholder(self):
        self._unlock()
        self.results_text.insert("end",
            "Aspect-level predictions will appear here.\n\n", "muted")
        self.results_text.insert("end",
            "Each detected aspect shows its sentiment and the context "
            "window used for prediction.\n", "muted")
        self._lock()

    def _unlock(self):
        self.results_text.configure(state="normal")

    def _lock(self):
        self.results_text.configure(state="disabled")

    def _clear(self):
        self._unlock()
        self.results_text.delete("1.0", "end")

    def _set_example(self, text):
        self.entry_var.set(text)
        self.entry.focus()
        self.entry.icursor("end")

    def _clear_verdict(self):
        for w in self.verdict_outer.winfo_children():
            w.destroy()

    def _show_verdict_banner(self, verdict, pos, neg):
        self._clear_verdict()

        if verdict == "Overall Positive":
            bg, fg, bd, icon = (self.POS_BG, self.POS_FG,
                                self.POS_BORDER, "✓  Overall Positive")
        elif verdict == "Overall Negative":
            bg, fg, bd, icon = (self.NEG_BG, self.NEG_FG,
                                self.NEG_BORDER, "✗  Overall Negative")
        elif verdict == "Mixed":
            bg, fg, bd, icon = (self.MIX_BG, self.MIX_FG,
                                self.MIX_BORDER, "~  Mixed Sentiment")
        else:
            bg, fg, bd, icon = (self.BG, self.MUTED, self.BORDER,
                                "—  " + verdict)

        banner = tk.Frame(self.verdict_outer, bg=bg,
                          highlightthickness=1, highlightbackground=bd,
                          padx=14, pady=10)
        banner.pack(fill="x")

        tk.Label(banner, text=icon,
                 font=("Helvetica", 13, "bold"),
                 bg=bg, fg=fg).pack(side="left")
        tk.Label(banner,
                 text=f"   {pos} positive  ·  {neg} negative aspect(s) detected",
                 font=("Helvetica", 10),
                 bg=bg, fg=fg).pack(side="left")

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _analyze(self):
        text = self.entry_var.get().strip()
        if not text:
            return

        self.btn.configure(state="disabled", text="Analyzing…")
        self.status_var.set("Running CRF extraction and sentiment prediction…")
        self._clear_verdict()
        self._clear()
        self.results_text.insert("end", "Analyzing…\n", "muted")
        self._lock()

        threading.Thread(
            target=self._worker, args=(text,), daemon=True
        ).start()

    def _worker(self, raw_text):
        try:
            result = self._predict(raw_text)
            self.root.after(0, lambda: self._display(result))
        except Exception as exc:
            self.root.after(0, lambda: self._display_error(str(exc)))

    def _predict(self, raw_text: str) -> dict:
        cleaned = self.preprocessor.clean_text(raw_text)
        tokens  = self.preprocessor.tokenize(cleaned)

        if not tokens:
            return {"type": "empty"}

        aspects  = self.crf_tagger.extract_aspect_entities(tokens)
        contexts = self._build_clause_contexts(tokens, aspects, self.window_size)

        if not contexts:
            sentiment = self.model.predict(tokens)
            return {"type": "fallback", "sentiment": sentiment}

        results = []
        counts  = Counter()
        for ctx in contexts:
            sent = self.model.predict(ctx["context"])
            results.append({
                "aspect":    ctx["aspect"],
                "sentiment": sent,
                "context":   " ".join(ctx["context"]),
            })
            counts[sent] += 1

        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        if pos > 0 and neg > 0:
            verdict = "Mixed"
        elif pos > neg:
            verdict = "Overall Positive"
        elif neg > pos:
            verdict = "Overall Negative"
        else:
            verdict = "Mixed"

        return {
            "type":    "aspects",
            "results": results,
            "pos":     pos,
            "neg":     neg,
            "verdict": verdict,
        }

    # ── Display ──────────────────────────────────────────────────────────────

    def _display(self, result: dict):
        self._clear()
        t = self.results_text

        if result["type"] == "empty":
            t.insert("end", "No tokens found in input.\n", "muted")
            self.status_var.set("Empty input.")

        elif result["type"] == "fallback":
            sent = result["sentiment"]
            tag  = "pos_line" if sent == "Positive" else "neg_line"
            icon = "[+]" if sent == "Positive" else "[-]"
            t.insert("end", "No specific aspects detected in this review.\n\n",
                     "muted")
            t.insert("end", "Full-review fallback prediction\n", "header")
            t.insert("end", f"\n  {icon}  Overall Sentiment : ", "muted")
            t.insert("end", f"  {sent}  \n", tag)
            self._show_verdict_banner(sent, 0, 0)
            self.status_var.set(f"Fallback prediction: {sent}")

        else:
            n = len(result["results"])
            t.insert("end", f"Detected {n} aspect(s)\n\n", "header")

            for r in result["results"]:
                asp  = r["aspect"]
                sent = r["sentiment"]
                ctx  = r["context"]
                tag  = "pos_line" if sent == "Positive" else "neg_line"
                icon = "[+]" if sent == "Positive" else "[-]"

                t.insert("end", f"  {icon} ", tag)
                t.insert("end", asp, "aspect_name")
                t.insert("end", "  →  ")
                t.insert("end", f"  {sent}  \n", tag)
                t.insert("end", f"      context: {ctx}\n\n", "context_line")

            self._show_verdict_banner(
                result["verdict"], result["pos"], result["neg"]
            )
            self.status_var.set(
                f"Done  ·  {n} aspect(s) found  ·  {result['verdict']}"
            )

        self._lock()
        self.btn.configure(state="normal", text="Analyze  →")

    def _display_error(self, msg: str):
        self._clear()
        self.results_text.insert("end", f"Error: {msg}\n", "neg_line")
        self._lock()
        self.status_var.set("Error during analysis.")
        self.btn.configure(state="normal", text="Analyze  →")


# ── Public entry point ───────────────────────────────────────────────────────

def launch_gui(model, preprocessor, crf_tagger,
               build_clause_contexts, contrast_conjunctions):
    """
    Launches the Tkinter GUI window.
    Blocks until the user closes the window.
    """
    root = tk.Tk()

    # Make it look less default on Windows
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass

    DarazABSAGui(
        root, model, preprocessor, crf_tagger,
        build_clause_contexts, contrast_conjunctions
    )
    root.mainloop()