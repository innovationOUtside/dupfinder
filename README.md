# dup-finder

Finds duplicated / near-duplicated content in a PDF (e.g. exported from
TeX), where duplicates may differ only in emphasis/italics (invisible to
extracted text anyway) or occasional word substitutions.

Four detection levels:

- **`growing`** (default) — seed-and-extend: finds a small anchor match,
  then grows it outward sentence-by-sentence for as long as the content
  keeps matching. Reports each duplicate at its *natural* length (one
  paragraph, three sentences, a dozen paragraphs...) as a single clean
  match, with surrounding context and a word-level diff. Recommended
  starting point.
- `chunk` — fixed-size sliding window of paragraphs (default 12)
- `paragraph` — individual paragraphs
- `sentence` — fixed-size sliding window of sentences (default 4)

The fixed-window levels are useful for a quick scan or for comparing
against the growing-window results, but they report every overlapping
window position separately, so one real duplicate typically shows up as
several overlapping matches rather than one.

## Setup

Requires [uv](https://docs.astral.sh/uv/). No other setup needed — `uv
run` creates an isolated environment and installs dependencies (all pure
Python, no system packages like poppler or tesseract required)
automatically on first run.

### Installing as a global command (optional)

By default `uv run dupfind ...` only works from inside this folder. To
get a `dupfind` command usable from any directory:

```bash
cd dup-finder
uv tool install .
```

If `dupfind: command not found` afterwards, your shell's PATH probably
doesn't include uv's tool bin directory yet — run `uv tool update-shell`
and restart your terminal. After installing, this folder can be moved or
deleted; the tool keeps working. Reinstall after editing the code with
`uv tool install . --reinstall`; remove with `uv tool uninstall dup-finder`.

## Usage

From this directory (or from anywhere, if installed per above):

```bash
uv run dupfind /path/to/yourdoc.pdf
# or, if installed as a tool:
dupfind /path/to/yourdoc.pdf
```

This writes two reports next to the input PDF:

- `yourdoc.duplicates.md` — Markdown, good for reading in an editor or GitHub
- `yourdoc.duplicates.html` — a single self-contained HTML page:
  - Locations show the **actual PDF page number**, e.g. "paragraphs 3-12
    (pages 1-2)", not just an abstract paragraph count. This is the
    *physical* page position in the file (page 1 = the very first page),
    since that's what actually navigates correctly in a viewer. If the
    PDF defines its own printed page labels (common when there's front
    matter - title page, contents, roman-numeral preliminaries - before
    the "real" page 1) and that differs from the physical position, it's
    shown too, e.g. "page 12, printed \"8\"".
  - Click a row to expand its word-level diff (red = removed, green = added).
  - Click a column header (**% match**, **Size**, or **Words**) to
    re-sort, click again to reverse.
  - Click **View** on a row to open the PDF viewer panel at the top,
    jumped to both matched pages side by side, **with the first few
    words of the match highlighted** so you can see exactly where the
    two sides line up. This uses a bundled copy of Mozilla's PDF.js
    viewer (not your browser's own built-in PDF viewer) so the
    highlighting behaves the same way regardless of browser - `dupfind`
    copies a `dupfind_pdfjs/` folder next to the report automatically.
    This only works if **both** the PDF file and that folder stay in the
    same directory as the report; if the panel stays blank or the
    highlight doesn't show up, the page number given is still correct -
    open the PDF yourself and go to that page.
  - Tick the checkbox on a row to mark it reviewed (struck through,
    dimmed). This is remembered in your browser (`localStorage`) if you
    close and reopen the report later, in the same browser — it isn't
    saved into the report file itself, so it won't show up if you send
    the file to someone else or open it in a different browser.

Open the HTML report directly in a browser — no server needed.

Run on a whole folder of PDFs:

```bash
for f in ~/Desktop/papers/*.pdf; do uv run dupfind "$f"; done
```

### Options

```
uv run dupfind FILE.pdf [options]

--level {growing,chunk,paragraph,sentence,all}
                          default: growing
--chunk-size N            paragraphs per chunk window (chunk level, default 12)
--window N                 sentences per window (sentence level, default 4)
--min-pair-score N         min per-sentence-pair similarity to keep growing
                            a match, 0-100 (growing level, default 60)
--min-final-score N        min whole-span similarity to report a
                            multi-sentence match, 0-100 (growing level, default 60)
--min-single-sentence-score N
                            min similarity to report a single-sentence
                            match, 0-100 (growing level, default 78 - stricter,
                            since short coincidental overlaps are common)
--context-words N          words of context shown either side of each
                            growing-mode match (default 15)
--out PATH                 Markdown report file path (default: <pdf name>.duplicates.md)
--html-out PATH            HTML report file path (default: <pdf name>.duplicates.html)
--no-html                  skip writing the HTML report
--no-markdown              skip writing the Markdown report
```

Examples:

```bash
# Default: both reports, growing-window level only
uv run dupfind thesis.pdf

# Everything, for comparison
uv run dupfind thesis.pdf --level all

# HTML only
uv run dupfind thesis.pdf --no-markdown

# More sensitive to short duplicates (more false positives too)
uv run dupfind thesis.pdf --min-final-score 50 --min-single-sentence-score 65

# Custom output locations
uv run dupfind thesis.pdf --out ~/Desktop/report.md --html-out ~/Desktop/report.html
```

## How it works

1. **Extraction**: `pdfplumber` pulls text line-by-line with position
   data. Paragraph breaks are detected geometrically (a natural-break
   split in the distribution of vertical gaps between lines), since flat
   PDF text extraction has no blank-line paragraph markers. Text that
   runs across a page boundary mid-sentence is stitched back together
   rather than forced into a spurious paragraph break.
2. **Dehyphenation**: line-broken hyphenated words are rejoined.
3. **Segmentation**: into paragraphs, then sentences.
4. **Growing-window matching**: candidate sentence-pair "seeds" are found
   via `rapidfuzz.process.cdist` (fast full pairwise fuzzy comparison;
   falls back to a shingle/MinHash pre-filter only for very large
   documents), then each seed is grown outward in both directions while
   consecutive sentence pairs keep matching (tolerating an occasional
   weak pair, so a reworded sentence mid-passage doesn't break the
   match). Overlapping growths are merged, keeping the longest.
5. **Fixed-window levels** use word-shingle Jaccard similarity for
   candidate generation (MinHash/LSH for large documents) and
   `rapidfuzz.token_sort_ratio` to confirm and score.

## Notes / limitations

- Sentence splitting is a lightweight regex-based splitter; it handles
  typical prose well but can be thrown off by unusual abbreviation
  patterns. Swap in spaCy or nltk's `punkt` in `split_sentences()` for
  more robust splitting on heavily-abbreviated text.
- Short matches (a single sentence, or a short 2-3 sentence span) are
  inherently harder to distinguish from coincidental overlap than long
  ones, since `token_sort_ratio` naturally scores short spans lower even
  when they're genuine duplicates. The single-sentence threshold is set
  stricter by default for this reason — tune `--min-final-score` /
  `--min-single-sentence-score` if you're missing or over-reporting short
  matches.
- The **search-phrase highlight** in the PDF viewer panel is best-effort:
  it's built from our own extracted/normalized text, not the PDF's raw
  bytes, so it can occasionally fail to highlight (most likely if the
  phrase happens to start right on a line-wrap hyphenation point). Page
  navigation still works even when the highlight doesn't show up.
- The bundled PDF.js viewer is Mozilla's official prebuilt distribution,
  trimmed of source maps and non-English UI locales to keep the download
  smaller; full CJK font/cmap and JBIG2/JPEG2000 image support is kept
  intact.
