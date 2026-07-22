#!/usr/bin/env python3
"""
find_duplicates.py

Find duplicated / near-duplicated content in a PDF (e.g. exported from TeX)
at three granularities:
    a) chunk level   - windows of ~N paragraphs
    b) paragraph level - individual paragraphs
    c) sentence-span level - windows of ~3-4 sentences

Method:
    1. Extract text with pdfplumber, dehyphenate line-break hyphenation,
       strip likely running headers/footers.
    2. Segment into paragraphs and sentences.
    3. Build sliding-window "documents" at each granularity.
    4. Shingle each window into word n-grams, use MinHash + LSH to find
       *candidate* near-duplicate pairs cheaply (avoids O(n^2) on large docs).
    5. Confirm + score candidates with a real string similarity
       (rapidfuzz token_sort_ratio, which ignores word order changes and
       is insensitive to case/emphasis) and show a readable diff.

Install deps:
    pip install pdfplumber rapidfuzz datasketch --break-system-packages

Usage:
    python find_duplicates.py mydoc.pdf --level paragraph
    python find_duplicates.py mydoc.pdf --level chunk --chunk-size 12
    python find_duplicates.py mydoc.pdf --level sentence --window 4
    python find_duplicates.py mydoc.pdf --level all --out report.md
"""

import argparse
import difflib
import html
import json
import os
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations

try:
    import pdfplumber
except ImportError:
    sys.exit("Missing dependency: pip install pdfplumber --break-system-packages")

try:
    import pypdfium2 as pdfium
except ImportError:
    sys.exit("Missing dependency: pip install pypdfium2 --break-system-packages")

try:
    from rapidfuzz import fuzz, process
except ImportError:
    sys.exit("Missing dependency: pip install rapidfuzz --break-system-packages")

try:
    from datasketch import MinHash, MinHashLSH
    HAVE_DATASKETCH = True
except ImportError:
    HAVE_DATASKETCH = False  # falls back to brute-force pairwise comparison


# --------------------------------------------------------------------------
# 1. Extraction & normalization
# --------------------------------------------------------------------------

class PageInfo:
    """Wraps the parallel (paragraph_pages, page_labels) lists. Supports
    `page_info[i]` exactly like the plain paragraph_pages list did (the
    physical, 1-indexed page - what actually navigates correctly in a
    PDF viewer), plus `page_info.label(i)` for the document's own
    printed page label, when it differs from the physical position."""
    __slots__ = ("pages", "labels")

    def __init__(self, pages, labels):
        self.pages = pages
        self.labels = labels

    def __getitem__(self, idx):
        return self.pages[idx]

    def __len__(self):
        return len(self.pages)

    def label(self, para_idx):
        physical_page = self.pages[para_idx]
        return self.labels[physical_page - 1]


def extract_paragraphs(path: str):
    """
    Extract paragraphs with their PDF page numbers.

    Plain `page.extract_text()` collapses everything into single newlines
    with no reliable blank-line paragraph markers (PDFs have no paragraph
    concept, just positioned lines). So paragraph breaks are instead
    detected geometrically: we look at the vertical gap between
    consecutive lines and insert a paragraph break wherever the gap
    belongs to the "large gap" cluster rather than the "normal line
    spacing" cluster.

    The threshold is computed once, globally across all pages, using a
    natural-break split (the biggest jump in the sorted gap distribution)
    rather than "the median gap is normal spacing" - the median approach
    silently fails on documents where most paragraphs are only 1-2 lines
    long, because then most gaps *are* paragraph breaks, not line-wraps.

    Returns (paragraphs, paragraph_pages, page_labels): paragraph_pages[i]
    is the 1-indexed PDF page paragraph[i] starts on - the *physical*
    position in the file (first page = 1), which is what actually
    navigates correctly in a PDF viewer. page_labels[page_num-1] is the
    document's own printed label for that physical page (e.g. 'iv' or
    '12'), which can differ from the physical position whenever a PDF
    has front matter - a title page, a table of contents, roman-numeral
    preliminaries - before the "real" page 1. Both are surfaced in
    reports: the physical number for anything that needs to actually
    open the right page, the label (only shown when it differs) so the
    number in the report matches what a human sees printed on the page.
    """
    all_lines_by_page = []
    all_gaps = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text_lines()
            all_lines_by_page.append(lines)
            for i in range(len(lines) - 1):
                gap = lines[i + 1]["top"] - lines[i]["bottom"]
                if gap > 0:
                    all_gaps.append(gap)

    page_labels = _extract_page_labels(path, len(all_lines_by_page))

    threshold = _natural_break_threshold(all_gaps)

    page_texts = [_assemble_page_text(lines, threshold) for lines in all_lines_by_page]
    page_texts = _strip_running_headers_footers(page_texts)
    blocks = _join_pages(page_texts)  # list of (start_page, text), 1-indexed pages

    paragraphs, paragraph_pages = [], []
    for start_page, block_text in blocks:
        block_text = normalize_text(dehyphenate(block_text))
        for p in split_paragraphs(block_text):
            paragraphs.append(p)
            paragraph_pages.append(start_page)
    return paragraphs, paragraph_pages, page_labels


def _extract_page_labels(path, n_pages):
    """Real printed page labels via pypdfium2 (respects a PDF's own
    /PageLabels catalog, e.g. 'i, ii, iii, 1, 2, 3...'). Falls back to
    the physical page number as a string for any page where pypdfium2
    can't determine a label, or if pypdfium2 fails outright (corrupt/
    unusual PDF) - page numbering is a nice-to-have, not something that
    should ever crash the whole extraction."""
    labels = [str(i + 1) for i in range(n_pages)]
    try:
        pdf = pdfium.PdfDocument(path)
        for i in range(min(n_pages, len(pdf))):
            try:
                label = pdf.get_page_label(i)
                if label:
                    labels[i] = label
            except Exception:
                pass
    except Exception:
        pass
    return labels


def _join_pages(page_texts):
    """Join per-page text, tracking each resulting block's starting page.
    A page boundary is only a paragraph break if the previous page's last
    line actually ended a sentence - text can otherwise run mid-sentence
    or mid-word across a page break (a line wrap, just like any other),
    which the per-page vertical-gap heuristic can't see since each page's
    coordinates reset independently.

    Returns a list of (start_page, text) tuples, start_page 1-indexed.
    """
    if not page_texts:
        return []
    result = [[1, page_texts[0]]]
    for page_num, txt in enumerate(page_texts[1:], start=2):
        prev_text = result[-1][1]
        prev_stripped = prev_text.rstrip()
        if not prev_stripped or re.search(r"[.!?]$", prev_stripped):
            result.append([page_num, txt])  # genuine paragraph break between pages
        else:
            # sentence/word continues onto the next page: merge as a
            # continuation rather than forcing a break (start_page stays
            # the earlier page, where the paragraph visually begins)
            result[-1][1] = prev_stripped + " " + txt.lstrip()
    return [(p, t) for p, t in result]


def _natural_break_threshold(gaps, fallback_factor=1.4):
    """Find a split point between 'normal line spacing' and 'paragraph
    gap' by looking for the largest jump in the sorted gap distribution
    (a simple 1D natural-breaks / Jenks-style split). Falls back to
    median * factor if the distribution doesn't show a clear jump."""
    if not gaps:
        return float("inf")
    s = sorted(gaps)
    if len(s) < 4:
        return s[len(s) // 2] * fallback_factor

    diffs = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    max_jump_idx = max(range(len(diffs)), key=lambda i: diffs[i])
    max_jump = diffs[max_jump_idx]

    # only trust the jump if it's actually a meaningfully large separation,
    # not just noise between two adjacent gap values
    typical_spacing = s[len(s) // 2]
    if max_jump > 0.25 * typical_spacing:
        return (s[max_jump_idx] + s[max_jump_idx + 1]) / 2
    return typical_spacing * fallback_factor


def _assemble_page_text(lines, break_threshold):
    if not lines:
        return ""
    out = [lines[0]["text"]]
    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]
        out.append("\n\n" if gap > break_threshold else "\n")
        out.append(lines[i]["text"])
    return "".join(out)


def _strip_running_headers_footers(pages, n_lines=2):
    """
    Heuristic: look at the first/last n_lines of every page; if the same
    line (after normalization) recurs on most pages, treat it as a
    header/footer and remove it.
    """
    from collections import Counter

    top_counter, bot_counter = Counter(), Counter()
    page_lines = []
    for txt in pages:
        lines = txt.splitlines()
        page_lines.append(lines)
        for l in lines[:n_lines]:
            top_counter[_norm_line(l)] += 1
        for l in lines[-n_lines:]:
            bot_counter[_norm_line(l)] += 1

    threshold = max(2, int(0.6 * len(pages)))
    junk_top = {l for l, c in top_counter.items() if c >= threshold and l}
    junk_bot = {l for l, c in bot_counter.items() if c >= threshold and l}

    cleaned = []
    for lines in page_lines:
        keep = []
        for i, l in enumerate(lines):
            norm = _norm_line(l)
            if i < n_lines and norm in junk_top:
                continue
            if i >= len(lines) - n_lines and norm in junk_bot:
                continue
            keep.append(l)
        cleaned.append("\n".join(keep))
    return cleaned


def _norm_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).strip().lower()
    line = re.sub(r"\d+", "#", line)  # page numbers vary, so mask digits
    return line


def dehyphenate(text: str) -> str:
    """Rejoin words split across lines by a hyphen, e.g. 'exam-\nple'."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # also join lines that TeX/pdfplumber broke mid-sentence (no hyphen,
    # just a line wrap): merge single newlines into spaces, keep blank
    # lines (paragraph breaks) intact.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# 2. Segmentation
# --------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z])(?<!\bal)(?<=[.!?])\s+(?=[A-Z(\"'\u201c])"
)


def split_paragraphs(text: str):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def split_sentences(paragraph: str):
    """Cheap sentence splitter. Swap in spaCy/nltk punkt for better accuracy
    on heavily-abbreviated academic text."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sents = _SENT_SPLIT_RE.split(paragraph)
    return [s.strip() for s in sents if s.strip()]


# --------------------------------------------------------------------------
# 3. Windowing
# --------------------------------------------------------------------------

@dataclass
class Unit:
    id: str
    text: str
    location: str  # human-readable position, e.g. "paragraphs 12-23 (page 4)"
    page_start: int = 1
    page_end: int = 1


def _fmt_loc(start_para_idx, end_para_idx, page_info):
    """Human-readable location including the actual PDF page number(s),
    e.g. 'paragraph 17 (page 4)' or 'paragraphs 3-12 (pages 1-2)' - so a
    match can actually be found in the source PDF, not just counted
    among the document's other paragraphs.

    'Page N' is always the *physical* page (1 = the very first page of
    the file), since that's what actually navigates correctly in a PDF
    viewer. If the document defines its own printed page labels (common
    when there's front matter - a title page, contents, roman-numeral
    preliminaries - before the "real" page 1) and that label differs
    from the physical number, it's appended too, e.g.
    'paragraph 40 (page 12, printed "8")', so the number here still
    matches what's actually printed on the page.
    """
    start_page = page_info[start_para_idx]
    end_page = page_info[end_para_idx]
    para_str = (f"paragraph {start_para_idx+1}" if start_para_idx == end_para_idx
                else f"paragraphs {start_para_idx+1}-{end_para_idx+1}")
    page_str = (f"page {start_page}" if start_page == end_page
                else f"pages {start_page}-{end_page}")

    start_label = page_info.label(start_para_idx)
    end_label = page_info.label(end_para_idx)
    if start_label != str(start_page) or end_label != str(end_page):
        label_str = (f'printed "{start_label}"' if start_label == end_label
                     else f'printed "{start_label}"-"{end_label}"')
        page_str += f", {label_str}"

    return f"{para_str} ({page_str})"


def make_paragraph_units(paragraphs, paragraph_pages) -> list:
    return [
        Unit(id=f"p{i}", text=p, location=_fmt_loc(i, i, paragraph_pages),
             page_start=paragraph_pages[i], page_end=paragraph_pages[i])
        for i, p in enumerate(paragraphs)
    ]


def make_chunk_units(paragraphs, paragraph_pages, chunk_size=12, step=1) -> list:
    units = []
    n = len(paragraphs)
    for start in range(0, max(1, n - chunk_size + 1), step):
        window = paragraphs[start:start + chunk_size]
        if len(window) < max(2, chunk_size // 2):
            continue
        units.append(Unit(
            id=f"c{start}",
            text="\n\n".join(window),
            location=_fmt_loc(start, start + len(window) - 1, paragraph_pages),
            page_start=paragraph_pages[start],
            page_end=paragraph_pages[start + len(window) - 1],
        ))
    return units


def flatten_sentences(paragraphs):
    """Return a flat list of (paragraph_index, sentence_text) across the
    whole document. Shared by the fixed-window sentence mode and the
    seed-and-extend mode."""
    all_sents = []
    for pi, p in enumerate(paragraphs):
        for s in split_sentences(p):
            all_sents.append((pi, s))
    return all_sents


def make_sentence_units(paragraphs, paragraph_pages, window=4, step=1) -> list:
    all_sents = flatten_sentences(paragraphs)

    units = []
    n = len(all_sents)
    for start in range(0, max(1, n - window + 1), step):
        span = all_sents[start:start + window]
        if len(span) < 2:
            continue
        text = " ".join(s for _, s in span)
        p_first, p_last = span[0][0], span[-1][0]
        loc = _fmt_loc(p_first, p_last, paragraph_pages)
        units.append(Unit(id=f"s{start}", text=text, location=loc,
                           page_start=paragraph_pages[p_first], page_end=paragraph_pages[p_last]))
    return units


# --------------------------------------------------------------------------
# 3b. Seed-and-extend (growing-window) matching
#
# Rather than pre-choosing a window size, find small "seed" matches at the
# sentence level, then greedily grow each seed outward - sentence by
# sentence, in both directions - for as long as the newly-added sentences
# keep matching. This is the same idea as BLAST-style local alignment /
# the classic MOSS plagiarism-detection algorithm: it finds the *natural*
# extent of each duplicate (one paragraph, four sentences, a dozen
# paragraphs...) instead of you having to guess a window size up front.
# --------------------------------------------------------------------------

@dataclass
class GrownMatch:
    a_range: tuple   # (start_idx, end_idx) inclusive, into flat sentence list
    b_range: tuple
    a_loc: str
    b_loc: str
    similarity: float
    text_a: str
    text_b: str
    diff: str = ""
    context_a_before: str = ""
    context_a_after: str = ""
    context_b_before: str = ""
    a_page_start: int = 1
    a_page_end: int = 1
    b_page_start: int = 1
    b_page_end: int = 1
    context_b_after: str = ""


def find_seed_pairs(sentences, seed_score=55, min_words=4, cdist_ceiling=4000):
    """Cheap candidate sentence pairs to use as growth starting points.

    Uses rapidfuzz.process.cdist to get the full pairwise fuzzy-similarity
    matrix for all sentences at once (fast C implementation - a couple
    thousand sentences, i.e. a substantial paper, takes well under a
    second). This matters because the obvious alternative - filtering
    candidates by shingle/n-gram overlap first - can silently miss a seed
    when a single sentence has several word substitutions clustered
    together, since that can wipe out all shingle overlap even though the
    fuzzy similarity is still clearly high. A seed only needs to be a
    plausible anchor; the extension step does the real verification, so a
    looser threshold here than the final reporting threshold is fine.

    Falls back to a shingle/MinHash pre-filter only once the sentence
    count is large enough that the full pairwise matrix would be too
    slow/memory-heavy.
    """
    texts = [s for _, s in sentences]
    n = len(texts)
    long_enough = [len(t.split()) >= min_words for t in texts]

    if n <= cdist_ceiling:
        matrix = process.cdist(texts, texts, scorer=fuzz.token_sort_ratio, workers=-1)
        seeds = []
        for i in range(n):
            if not long_enough[i]:
                continue
            for j in range(i + 1, n):
                if long_enough[j] and matrix[i, j] >= seed_score:
                    seeds.append((i, j))
        return seeds

    # large-document fallback: shingle overlap as a coarse pre-filter
    shingle_sets = [shingles(t, n=3) for t in texts]
    lsh = MinHashLSH(threshold=0.3, num_perm=128)
    minhashes = []
    for idx, sset in enumerate(shingle_sets):
        mh = MinHash(num_perm=128)
        for sh in sset:
            mh.update(sh.encode("utf8"))
        minhashes.append(mh)
        lsh.insert(str(idx), mh)
    seen = set()
    seeds = []
    for idx, mh in enumerate(minhashes):
        if not long_enough[idx]:
            continue
        for match in lsh.query(mh):
            j = int(match)
            if j == idx or not long_enough[j]:
                continue
            key = tuple(sorted((idx, j)))
            if key in seen:
                continue
            seen.add(key)
            seeds.append(key)
    return seeds


def extend_seed(sentences, i, j, min_pair_score=60, max_gap=1):
    """Grow a seed (i, j) outward in both directions. Extension continues
    past an occasional weak/non-matching sentence pair (up to `max_gap`
    consecutive misses) to tolerate a sentence being inserted, deleted, or
    reworded beyond recognition in the middle of an otherwise-duplicated
    passage - then snaps back to the last position that still matched."""
    n = len(sentences)
    texts = [s for _, s in sentences]

    def pair_score(a, b):
        return fuzz.token_sort_ratio(texts[a], texts[b])

    lo_i, lo_j = i, j
    hi_i, hi_j = i, j

    # extend right
    cur_i, cur_j, misses = i, j, 0
    last_good_i, last_good_j = i, j
    while cur_i + 1 < n and cur_j + 1 < n:
        cur_i += 1
        cur_j += 1
        if pair_score(cur_i, cur_j) >= min_pair_score:
            misses = 0
            last_good_i, last_good_j = cur_i, cur_j
        else:
            misses += 1
            if misses > max_gap:
                break
    hi_i, hi_j = last_good_i, last_good_j

    # extend left
    cur_i, cur_j, misses = i, j, 0
    last_good_i, last_good_j = i, j
    while cur_i - 1 >= 0 and cur_j - 1 >= 0:
        cur_i -= 1
        cur_j -= 1
        if pair_score(cur_i, cur_j) >= min_pair_score:
            misses = 0
            last_good_i, last_good_j = cur_i, cur_j
        else:
            misses += 1
            if misses > max_gap:
                break
    lo_i, lo_j = last_good_i, last_good_j

    return (lo_i, hi_i), (lo_j, hi_j)


def _ranges_overlap(r1, r2):
    return r1[0] <= r2[1] and r2[0] <= r1[1]


def _context_window(texts, before_idx, after_idx, n_words=15):
    """Grab up to n_words of context immediately before `before_idx` and
    immediately after `after_idx` in the flat sentence-text list, walking
    outward sentence by sentence. Mirrors the "concordance" context lines
    in the linked blog posts, so a match can be read in situ rather than
    as an isolated fragment."""
    before_words = []
    k = before_idx
    while k >= 0 and len(before_words) < n_words:
        before_words = texts[k].split() + before_words
        k -= 1
    before = " ".join(before_words[-n_words:])

    after_words = []
    k = after_idx
    while k < len(texts) and len(after_words) < n_words:
        after_words = after_words + texts[k].split()
        k += 1
    after = " ".join(after_words[:n_words])

    return before, after


def merge_grown_matches(sentences, grown, paragraph_pages, min_final_score=60,
                         min_single_sentence_score=78, context_words=15):
    """Seeds inside the same true duplicate region all grow into nearly
    the same span, so collapse any matches whose A-range AND B-range both
    overlap, keeping the longest. Also drops matches that are just a
    single sentence pair (not a real 'span') and re-scores the final
    merged span as a whole.

    Uses a length-aware score threshold: a single-sentence match needs a
    stricter score (min_single_sentence_score) because two unrelated
    sentences coincidentally sharing a few common words land in roughly
    the same score range as a genuinely short paraphrased duplicate - a
    short span naturally scores lower than a long one on
    token_sort_ratio, since there's less shared text to dilute the effect
    of a few word changes, so one flat threshold can't tell them apart.
    Multi-sentence spans get the more permissive min_final_score.
    """
    texts = [s for _, s in sentences]

    def span_len(m):
        return (m[0][1] - m[0][0]) + (m[1][1] - m[1][0])

    grown = sorted(grown, key=span_len, reverse=True)
    kept = []
    for a_range, b_range in grown:
        if a_range == b_range:
            continue  # identical location, not a duplicate of something else
        dup = any(
            _ranges_overlap(a_range, ka) and _ranges_overlap(b_range, kb)
            for ka, kb in kept
        )
        if not dup:
            kept.append((a_range, b_range))

    results = []
    for a_range, b_range in kept:
        text_a = " ".join(texts[a_range[0]:a_range[1] + 1])
        text_b = " ".join(texts[b_range[0]:b_range[1] + 1])
        score = fuzz.token_sort_ratio(text_a, text_b)
        is_single_sentence = a_range[0] == a_range[1] and b_range[0] == b_range[1]
        threshold = min_single_sentence_score if is_single_sentence else min_final_score
        if score < threshold:
            continue
        p_a0, p_a1 = sentences[a_range[0]][0], sentences[a_range[1]][0]
        p_b0, p_b1 = sentences[b_range[0]][0], sentences[b_range[1]][0]
        a_loc = _fmt_loc(p_a0, p_a1, paragraph_pages)
        b_loc = _fmt_loc(p_b0, p_b1, paragraph_pages)
        ctx_a_before, ctx_a_after = _context_window(
            texts, a_range[0] - 1, a_range[1] + 1, context_words)
        ctx_b_before, ctx_b_after = _context_window(
            texts, b_range[0] - 1, b_range[1] + 1, context_words)
        results.append(GrownMatch(
            a_range=a_range, b_range=b_range, a_loc=a_loc, b_loc=b_loc,
            similarity=score, text_a=text_a, text_b=text_b,
            diff=make_diff(text_a, text_b),
            context_a_before=ctx_a_before, context_a_after=ctx_a_after,
            context_b_before=ctx_b_before, context_b_after=ctx_b_after,
            a_page_start=paragraph_pages[p_a0], a_page_end=paragraph_pages[p_a1],
            b_page_start=paragraph_pages[p_b0], b_page_end=paragraph_pages[p_b1],
        ))
    results.sort(key=lambda m: -m.similarity)
    return results


def find_growing_matches(paragraphs, paragraph_pages, min_pair_score=60, min_final_score=60,
                          min_single_sentence_score=78, max_gap=1, context_words=15):
    """Full seed-and-extend pipeline: seed -> grow -> merge. Finds
    duplicated spans of whatever natural length they have, rather than
    requiring a pre-chosen window size."""
    sentences = flatten_sentences(paragraphs)
    if len(sentences) < 2:
        return []
    seeds = find_seed_pairs(sentences)
    grown = [extend_seed(sentences, i, j, min_pair_score, max_gap) for i, j in seeds]
    return merge_grown_matches(sentences, grown, paragraph_pages, min_final_score,
                                min_single_sentence_score, context_words)


# --------------------------------------------------------------------------
# 4. Shingling + candidate generation (MinHash/LSH, or brute force)
# --------------------------------------------------------------------------

def shingles(text: str, n=6):
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_candidate_pairs(units, shingle_size, jaccard_threshold, num_perm=128,
                          brute_force_ceiling=200):
    """Returns list of (i, j) index pairs into `units` worth a closer look.

    Below `brute_force_ceiling` units, every pair is returned and left to
    the fuzzy scorer to judge - the shingle/Jaccard filter is a *speed*
    optimization for large corpora, and applying it on small documents can
    silently drop true near-duplicates when several substitutions cluster
    together and wipe out all n-gram overlap even though overall similarity
    is still high.
    """
    n = len(units)
    if n < 2:
        return []

    if n <= brute_force_ceiling:
        return list(combinations(range(n), 2))

    shingle_sets = [shingles(u.text, n=shingle_size) for u in units]

    if HAVE_DATASKETCH and n > 60:
        # MinHash + LSH scales to large documents without O(n^2) comparisons.
        lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        minhashes = []
        for idx, sset in enumerate(shingle_sets):
            mh = MinHash(num_perm=num_perm)
            for sh in sset:
                mh.update(sh.encode("utf8"))
            minhashes.append(mh)
            lsh.insert(str(idx), mh)

        seen = set()
        pairs = []
        for idx, mh in enumerate(minhashes):
            for match in lsh.query(mh):
                j = int(match)
                if j == idx:
                    continue
                key = tuple(sorted((idx, j)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        return pairs
    else:
        # Brute force is fine for small unit counts (a single paper's
        # paragraph/chunk list is usually well under a few hundred).
        pairs = []
        for i, j in combinations(range(n), 2):
            # skip windows that trivially overlap each other (adjacent
            # sliding-window steps sharing most of their content)
            if abs(i - j) <= 1 and units[i].id[0] != 'p':
                continue
            if jaccard(shingle_sets[i], shingle_sets[j]) >= jaccard_threshold:
                pairs.append((i, j))
        return pairs


# --------------------------------------------------------------------------
# 5. Confirmation + diff
# --------------------------------------------------------------------------

@dataclass
class Match:
    unit_a: Unit
    unit_b: Unit
    similarity: float
    diff: str = ""


def _window_start(unit_id: str):
    """Parse the sliding-window start index encoded in chunk/sentence unit
    ids (e.g. 'c24' -> 24). Returns None for paragraph units ('p3'), which
    don't overlap with each other and need no such filtering."""
    if unit_id[0] not in ("c", "s"):
        return None
    return int(unit_id[1:])


def confirm_and_score(units, pairs, min_score=70, window_size=1) -> list:
    results = []
    for i, j in pairs:
        a, b = units[i], units[j]

        # skip trivially-overlapping adjacent windows (same sliding window,
        # shifted by less than a full window) - these share most of their
        # text by construction and aren't real duplicates
        sa, sb = _window_start(a.id), _window_start(b.id)
        if sa is not None and sb is not None and abs(sa - sb) < window_size:
            continue

        score = fuzz.token_sort_ratio(a.text, b.text)  # 0-100, word-order robust
        if score >= min_score:
            diff = make_diff(a.text, b.text)
            results.append(Match(a, b, score, diff))
    results.sort(key=lambda m: -m.similarity)
    return results


def make_diff(a: str, b: str) -> str:
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split())
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a_span = " ".join(a.split()[i1:i2])
        b_span = " ".join(b.split()[j1:j2])
        out.append(f"  [{tag}] A: '{a_span}'  ->  B: '{b_span}'")
    return "\n".join(out) if out else "  (identical after normalization)"


def make_html_diff(a: str, b: str):
    """Inline word-level diff as HTML: unchanged words are plain,
    A-only words are struck-through/red, B-only words are highlighted
    green - like a typical 'track changes' view. Returns (html_a, html_b)
    - the whole of A with deletions marked, and the whole of B with
    insertions marked - so both sides can be shown in full context."""
    a_words, b_words = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=a_words, b=b_words)
    out_a, out_b = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_span = html.escape(" ".join(a_words[i1:i2]))
        b_span = html.escape(" ".join(b_words[j1:j2]))
        if tag == "equal":
            out_a.append(a_span)
            out_b.append(b_span)
        elif tag == "replace":
            if a_span:
                out_a.append(f'<del>{a_span}</del>')
            if b_span:
                out_b.append(f'<ins>{b_span}</ins>')
        elif tag == "delete":
            out_a.append(f'<del>{a_span}</del>')
        elif tag == "insert":
            out_b.append(f'<ins>{b_span}</ins>')
    return " ".join(out_a), " ".join(out_b)


# --------------------------------------------------------------------------
# 6. Driver
# --------------------------------------------------------------------------

LEVEL_CONFIG = {
    # level: (shingle_size, jaccard_threshold, fuzzy_min_score)
    "chunk":     (8, 0.5, 75),
    "paragraph": (5, 0.5, 70),
    "sentence":  (4, 0.6, 75),
}


def merge_consecutive_paragraph_matches(matches, paragraphs, paragraph_pages):
    """Individual paragraph-level matches are found one paragraph at a
    time, so three consecutive paragraphs that each independently match
    (paragraph 5<->30, 6<->31, 7<->32) show up as three separate rows even
    though they're really one 3-paragraph match. This chains together any
    run of matches where paragraph i<->j is followed by (i+1)<->(j+1),
    (i+2)<->(j+2), etc., and re-scores the merged span as a whole -
    mirroring what growing mode does at the sentence level, but applied
    directly to the confirmed paragraph-level matches."""
    pair_map = {}
    for m in matches:
        i = int(m.unit_a.id[1:])
        j = int(m.unit_b.id[1:])
        pair_map[(i, j)] = m

    visited = set()
    merged = []
    for (i, j) in sorted(pair_map.keys()):
        if (i, j) in visited:
            continue
        start_i, start_j = i, j
        while (start_i - 1, start_j - 1) in pair_map and (start_i - 1, start_j - 1) not in visited:
            start_i -= 1
            start_j -= 1
        cur_i, cur_j = start_i, start_j
        chain = [(cur_i, cur_j)]
        while (cur_i + 1, cur_j + 1) in pair_map:
            cur_i += 1
            cur_j += 1
            chain.append((cur_i, cur_j))
        visited.update(chain)
        end_i, end_j = chain[-1]

        text_a = "\n\n".join(paragraphs[start_i:end_i + 1])
        text_b = "\n\n".join(paragraphs[start_j:end_j + 1])
        score = fuzz.token_sort_ratio(text_a, text_b)
        a_loc = _fmt_loc(start_i, end_i, paragraph_pages)
        b_loc = _fmt_loc(start_j, end_j, paragraph_pages)
        unit_a = Unit(id=f"p{start_i}", text=text_a, location=a_loc,
                      page_start=paragraph_pages[start_i], page_end=paragraph_pages[end_i])
        unit_b = Unit(id=f"p{start_j}", text=text_b, location=b_loc,
                      page_start=paragraph_pages[start_j], page_end=paragraph_pages[end_j])
        merged.append(Match(unit_a=unit_a, unit_b=unit_b, similarity=score,
                             diff=make_diff(text_a, text_b)))

    merged.sort(key=lambda m: -m.similarity)
    return merged


def run_level(level, paragraphs, paragraph_pages, args):
    shingle_size, jac_thresh, fuzzy_min = LEVEL_CONFIG[level]

    if level == "chunk":
        units = make_chunk_units(paragraphs, paragraph_pages, chunk_size=args.chunk_size)
        window_size = args.chunk_size
    elif level == "paragraph":
        units = make_paragraph_units(paragraphs, paragraph_pages)
        window_size = 1
    else:
        units = make_sentence_units(paragraphs, paragraph_pages, window=args.window)
        window_size = args.window

    if len(units) < 2:
        return []

    pairs = find_candidate_pairs(units, shingle_size, jac_thresh)
    matches = confirm_and_score(units, pairs, min_score=fuzzy_min, window_size=window_size)

    if level == "paragraph":
        matches = merge_consecutive_paragraph_matches(matches, paragraphs, paragraph_pages)

    return matches


# --------------------------------------------------------------------------
# 6. Markdown report
# --------------------------------------------------------------------------

def _fmt_context(before, quoted, after):
    """Concordance-style single line: dim context either side of the
    matched text, similar to the 'show text just before/after' style in
    the linked blog posts."""
    parts = []
    if before:
        parts.append(f"*...{before}*")
    parts.append(f"**{quoted.strip()}**")
    if after:
        parts.append(f"*{after}...*")
    return " ".join(parts)


def build_growing_report_md(matches, source_name):
    lines = [f"## Growing (seed-and-extend) matches — {len(matches)} found\n"]
    if not matches:
        lines.append("_No matches found at this level._\n")
        return "\n".join(lines)

    lines.append(
        "Each match below was seeded from a small anchor and grown outward "
        "sentence-by-sentence in both directions for as long as the content "
        "kept matching, so the span length is whatever it naturally is — "
        "not a fixed window.\n"
    )

    for n, m in enumerate(matches, 1):
        n_sents_a = m.a_range[1] - m.a_range[0] + 1
        n_sents_b = m.b_range[1] - m.b_range[0] + 1
        lines.append(f"### Match {n} — {m.similarity:.0f}% similar\n")
        lines.append(f"- **A:** {m.a_loc} ({n_sents_a} sentence{'s' if n_sents_a != 1 else ''})")
        lines.append(f"- **B:** {m.b_loc} ({n_sents_b} sentence{'s' if n_sents_b != 1 else ''})\n")
        lines.append("**In context:**\n")
        lines.append(f"> A: {_fmt_context(m.context_a_before, m.text_a, m.context_a_after)}")
        lines.append(">")
        lines.append(f"> B: {_fmt_context(m.context_b_before, m.text_b, m.context_b_after)}\n")
        lines.append("<details><summary>Word-level diff</summary>\n")
        lines.append("```")
        lines.append(m.diff)
        lines.append("```")
        lines.append("</details>\n")
        lines.append("---\n")
    return "\n".join(lines)


def build_fixed_report_md(level, matches):
    lines = [f"## {level.capitalize()}-level matches — {len(matches)} found\n"]
    if not matches:
        lines.append("_No matches found at this level._\n")
        return "\n".join(lines)

    for n, m in enumerate(matches, 1):
        lines.append(f"### Match {n} — {m.similarity:.0f}% similar\n")
        lines.append(f"- **A:** {m.unit_a.location}")
        lines.append(f"- **B:** {m.unit_b.location}\n")
        lines.append(f"> A: {m.unit_a.text[:300]}{'...' if len(m.unit_a.text) > 300 else ''}")
        lines.append(">")
        lines.append(f"> B: {m.unit_b.text[:300]}{'...' if len(m.unit_b.text) > 300 else ''}\n")
        lines.append("<details><summary>Word-level diff</summary>\n")
        lines.append("```")
        lines.append(m.diff)
        lines.append("```")
        lines.append("</details>\n")
        lines.append("---\n")
    return "\n".join(lines)


def build_summary(source_name, n_paragraphs, level_counts):
    lines = [f"# Duplicate-content report: `{source_name}`\n"]
    lines.append(f"Extracted **{n_paragraphs}** paragraphs.\n")
    lines.append("| Level | Matches found |")
    lines.append("|---|---|")
    for level, count in level_counts.items():
        lines.append(f"| {level} | {count} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 7. HTML report (sortable table + inline diff)
# --------------------------------------------------------------------------

_LOC_SPAN_RE = re.compile(r"paragraphs?\s+(\d+)(?:-(\d+))?")


def _loc_span(loc: str) -> int:
    """Number of paragraphs covered by a location string like
    'paragraph 5' or 'paragraphs 5-16'. Used as the sortable 'size' of a
    match across every level, since growing-mode locations are expressed
    in the same paragraph-range format as the fixed-window levels."""
    m = _LOC_SPAN_RE.search(loc)
    if not m:
        return 1
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    return end - start + 1


def _search_phrase(text, n_words=6):
    """Short phrase from the start of a matched span, used to make the
    bundled PDF.js viewer jump to and highlight the exact spot, not just
    the page. Picked from our own extracted/normalized text rather than
    anything reconstructed from the raw PDF bytes, so it can occasionally
    fail to highlight (e.g. if it happens to fall right on a line-wrap
    hyphenation point) - if that happens the page navigation still
    works, just without the highlight."""
    return " ".join(text.split()[:n_words])


def _growing_rows(matches):
    rows = []
    for m in matches:
        diff_a, diff_b = make_html_diff(m.text_a, m.text_b)
        size = max(_loc_span(m.a_loc), _loc_span(m.b_loc))
        words = max(len(m.text_a.split()), len(m.text_b.split()))
        n_sents_a = m.a_range[1] - m.a_range[0] + 1
        n_sents_b = m.b_range[1] - m.b_range[0] + 1
        rows.append({
            "similarity": m.similarity,
            "size": size,
            "size_label": f"{size} paragraph{'s' if size != 1 else ''}"
                          f" ({n_sents_a} vs {n_sents_b} sentences)",
            "words": words,
            "words_label": f"{len(m.text_a.split())} vs {len(m.text_b.split())} words",
            "a_loc": m.a_loc, "b_loc": m.b_loc,
            "page_a": m.a_page_start, "page_b": m.b_page_start,
            "phrase_a": _search_phrase(m.text_a), "phrase_b": _search_phrase(m.text_b),
            "diff_a": diff_a, "diff_b": diff_b,
            "context_a": (m.context_a_before, m.context_a_after),
            "context_b": (m.context_b_before, m.context_b_after),
        })
    return rows


def _fixed_rows(matches):
    rows = []
    for m in matches:
        diff_a, diff_b = make_html_diff(m.unit_a.text, m.unit_b.text)
        size = max(_loc_span(m.unit_a.location), _loc_span(m.unit_b.location))
        words_a, words_b = len(m.unit_a.text.split()), len(m.unit_b.text.split())
        rows.append({
            "similarity": m.similarity,
            "size": size,
            "size_label": f"{size} paragraph{'s' if size != 1 else ''}",
            "words": max(words_a, words_b),
            "words_label": f"{words_a} vs {words_b} words",
            "a_loc": m.unit_a.location, "b_loc": m.unit_b.location,
            "page_a": m.unit_a.page_start, "page_b": m.unit_b.page_start,
            "phrase_a": _search_phrase(m.unit_a.text), "phrase_b": _search_phrase(m.unit_b.text),
            "diff_a": diff_a, "diff_b": diff_b,
            "context_a": None, "context_b": None,
        })
    return rows


def _html_table(level_key, level_name, rows):
    if not rows:
        return f'<h2>{html.escape(level_name)} — 0 matches</h2><p class="empty">No matches found at this level.</p>'

    rows = sorted(rows, key=lambda r: r["size"], reverse=True)  # default: most lines first

    out = [f'<h2>{html.escape(level_name)} — {len(rows)} match{"es" if len(rows) != 1 else ""}</h2>']
    out.append('<table class="dup-table"><thead><tr>')
    out.append('<th title="Mark reviewed">&#x2713;</th>')
    out.append('<th data-key="rank">#</th>')
    out.append('<th data-key="similarity" class="sortable">% match &#x2195;</th>')
    out.append('<th data-key="size" class="sortable">Size &#x2195;</th>')
    out.append('<th data-key="words" class="sortable">Words &#x2195;</th>')
    out.append('<th>A</th><th>B</th><th>PDF</th>')
    out.append('</tr></thead><tbody>')

    for i, r in enumerate(rows, 1):
        # stable key independent of sort order / row index, so "reviewed"
        # state survives re-sorting and reopening the report
        row_key = html.escape(f"{level_key}::{r['a_loc']}::{r['b_loc']}", quote=True)
        pdf_call = (f"showInPdf({r['page_a']}, {json.dumps(r['phrase_a'])}, "
                    f"{r['page_b']}, {json.dumps(r['phrase_b'])})")
        out.append(
            f'<tr class="match-row" data-similarity="{r["similarity"]:.1f}" '
            f'data-size="{r["size"]}" data-words="{r["words"]}" data-key="{row_key}">'
            f'<td class="check-cell"><input type="checkbox" class="reviewed-check" '
            f'data-row-key="{row_key}" onclick="event.stopPropagation(); toggleReviewed(this)"></td>'
            f'<td onclick="toggleDiff(this.parentNode)">{i}</td>'
            f'<td onclick="toggleDiff(this.parentNode)">{r["similarity"]:.0f}%</td>'
            f'<td onclick="toggleDiff(this.parentNode)">{html.escape(r["size_label"])}</td>'
            f'<td onclick="toggleDiff(this.parentNode)">{html.escape(r["words_label"])}</td>'
            f'<td onclick="toggleDiff(this.parentNode)">{html.escape(r["a_loc"])}</td>'
            f'<td onclick="toggleDiff(this.parentNode)">{html.escape(r["b_loc"])}</td>'
            f'<td><button type="button" class="pdf-btn" '
            f'onclick="event.stopPropagation(); {html.escape(pdf_call, quote=True)}">View</button></td>'
            f'</tr>'
        )
        out.append('<tr class="diff-row" style="display:none"><td colspan="8">')
        if r["context_a"]:
            ctx_a_before, ctx_a_after = r["context_a"]
            ctx_b_before, ctx_b_after = r["context_b"]
            out.append('<div class="context">')
            if ctx_a_before or ctx_a_after:
                out.append(f'<p><em>...{html.escape(ctx_a_before)}</em> <strong>[A]</strong> '
                            f'<em>{html.escape(ctx_a_after)}...</em></p>')
            if ctx_b_before or ctx_b_after:
                out.append(f'<p><em>...{html.escape(ctx_b_before)}</em> <strong>[B]</strong> '
                            f'<em>{html.escape(ctx_b_after)}...</em></p>')
            out.append('</div>')
        out.append('<div class="diff-pane"><div class="diff-col"><h4>A</h4><p>')
        out.append(r["diff_a"])
        out.append('</p></div><div class="diff-col"><h4>B</h4><p>')
        out.append(r["diff_b"])
        out.append('</p></div></div>')
        out.append('</td></tr>')

    out.append('</tbody></table>')
    return "\n".join(out)


_HTML_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1400px;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
.summary-table { border-collapse: collapse; margin-bottom: 1.5rem; }
.summary-table td, .summary-table th { border: 1px solid #ddd; padding: .3rem .8rem; text-align: left; }
table.dup-table { border-collapse: collapse; width: 100%; font-size: .92rem; }
.dup-table th, .dup-table td { border: 1px solid #e0e0e0; padding: .4rem .6rem; text-align: left; }
.dup-table th { background: #f5f5f5; position: sticky; top: 0; }
.dup-table th.sortable { cursor: pointer; user-select: none; }
.dup-table th.sortable:hover { background: #eaeaea; }
.check-cell { text-align: center; }
.match-row td:not(.check-cell) { cursor: pointer; }
.match-row:hover { background: #f9f9f9; }
.match-row.reviewed { color: #999; }
.match-row.reviewed td:not(.check-cell) { text-decoration: line-through; text-decoration-color: #ccc; }
.pdf-btn { font-size: .8rem; padding: .2rem .5rem; cursor: pointer; border: 1px solid #ccc;
           border-radius: 4px; background: #fff; }
.pdf-btn:hover { background: #eef4ff; border-color: #99b8e8; }
.diff-row td { background: #fafafa; padding: 1rem; }
.diff-pane { display: flex; gap: 1.5rem; }
.diff-col { flex: 1; min-width: 0; }
.diff-col h4 { margin: 0 0 .3rem 0; color: #555; }
.diff-col p { white-space: pre-wrap; word-wrap: break-word; }
del { background: #ffd7d7; color: #7a1f1f; text-decoration: line-through; }
ins { background: #d3f5d3; color: #1f5c1f; text-decoration: none; }
.context { background: #f0f4f8; padding: .6rem .8rem; border-radius: 4px; margin-bottom: .8rem; font-size: .88rem; }
.context em { color: #666; font-style: italic; }
.empty { color: #888; font-style: italic; }

#pdf-panel { position: sticky; top: 0; z-index: 10; background: #fff; border: 1px solid #ddd;
             border-radius: 6px; padding: .6rem; margin: 1rem 0; box-shadow: 0 2px 6px rgba(0,0,0,.08); }
#pdf-panel.collapsed .pdf-frames { display: none; }
#pdf-panel-header { display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
#pdf-panel-header h3 { margin: 0; font-size: .95rem; }
.pdf-frames { display: flex; gap: .6rem; margin-top: .6rem; }
.pdf-frame-col { flex: 1; min-width: 0; }
.pdf-frame-col h4 { margin: 0 0 .3rem 0; font-size: .85rem; color: #555; }
.pdf-frame-col iframe { width: 100%; height: 480px; border: 1px solid #ddd; border-radius: 4px; }
.pdf-note { font-size: .78rem; color: #888; margin-top: .4rem; }
"""

_HTML_JS = """
const PDF_FILE = %(pdf_file_json)s;
const PDFJS_VIEWER = %(pdfjs_viewer_json)s;
const STORAGE_PREFIX = %(storage_prefix_json)s;

function toggleDiff(rowEl) {
  const next = rowEl.nextElementSibling;
  next.style.display = next.style.display === 'none' ? '' : 'none';
}

function toggleReviewed(checkbox) {
  const row = checkbox.closest('tr');
  const key = checkbox.dataset.rowKey;
  const storeKey = STORAGE_PREFIX + key;
  if (checkbox.checked) {
    localStorage.setItem(storeKey, '1');
    row.classList.add('reviewed');
  } else {
    localStorage.removeItem(storeKey);
    row.classList.remove('reviewed');
  }
}

function restoreReviewed() {
  document.querySelectorAll('.reviewed-check').forEach(function(cb) {
    const storeKey = STORAGE_PREFIX + cb.dataset.rowKey;
    if (localStorage.getItem(storeKey) === '1') {
      cb.checked = true;
      cb.closest('tr').classList.add('reviewed');
    }
  });
}

function showInPdf(pageA, phraseA, pageB, phraseB) {
  const panel = document.getElementById('pdf-panel');
  panel.classList.remove('collapsed');
  document.getElementById('pdf-frame-a').src = pdfjsViewerUrl(pageA, phraseA);
  document.getElementById('pdf-frame-b').src = pdfjsViewerUrl(pageB, phraseB);
  panel.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function pdfjsViewerUrl(page, phrase) {
  // PDFJS_VIEWER lives at <output_dir>/PDFJS_DIR/web/viewer.html, so the
  // PDF (in <output_dir>, alongside this report) is two levels up.
  const fileParam = '../../' + encodeURIComponent(PDF_FILE);
  const search = encodeURIComponent(phrase);
  return PDFJS_VIEWER + '?file=' + fileParam + '#page=' + page + '&search=' + search + '&phrase=true';
}

document.getElementById('pdf-panel-header').addEventListener('click', function() {
  document.getElementById('pdf-panel').classList.toggle('collapsed');
});

document.querySelectorAll('table.dup-table').forEach(function(table) {
  table.querySelectorAll('th.sortable').forEach(function(th) {
    let asc = false;
    th.addEventListener('click', function() {
      const key = th.dataset.key;
      const tbody = table.querySelector('tbody');
      const pairs = [];
      const rows = Array.from(tbody.querySelectorAll('tr.match-row'));
      rows.forEach(function(r) { pairs.push([r, r.nextElementSibling]); });
      asc = !asc;
      pairs.sort(function(x, y) {
        const a = parseFloat(x[0].dataset[key]);
        const b = parseFloat(y[0].dataset[key]);
        return asc ? a - b : b - a;
      });
      pairs.forEach(function(p) { tbody.appendChild(p[0]); tbody.appendChild(p[1]); });
    });
  });
});

restoreReviewed();
"""


PDFJS_DIR_NAME = "dupfind_pdfjs"
_BUNDLED_PDFJS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfjs")


def _copy_pdfjs_viewer(output_dir):
    """Copy the bundled PDF.js viewer next to the HTML report so the
    'View' button's iframes can load it, without needing a network
    connection or any external CDN. Skipped (with a warning, not a
    crash) if the bundle isn't found - the rest of the report still
    works fine, just without the in-page PDF viewer."""
    if not os.path.isdir(_BUNDLED_PDFJS_DIR):
        print(f"Note: bundled PDF.js viewer not found at {_BUNDLED_PDFJS_DIR}; "
              f"the report's PDF viewer panel won't work.", file=sys.stderr)
        return
    target = os.path.join(output_dir, PDFJS_DIR_NAME)
    shutil.copytree(_BUNDLED_PDFJS_DIR, target, dirs_exist_ok=True)


def build_html_report(source_name, pdf_filename, n_paragraphs, level_counts, level_tables):
    js = _HTML_JS % {
        "pdf_file_json": json.dumps(pdf_filename),
        "pdfjs_viewer_json": json.dumps(f"{PDFJS_DIR_NAME}/web/viewer.html"),
        "storage_prefix_json": json.dumps(f"dupfind::{pdf_filename}::"),
    }
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Duplicate report — {html.escape(source_name)}</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        f"<h1>Duplicate-content report: {html.escape(source_name)}</h1>",
        f"<p>Extracted {n_paragraphs} paragraphs. Click a row to expand its diff; "
        f"click a column header (% match / Size / Words) to sort — click again to reverse. "
        f"Tick the checkbox once you've reviewed a match (remembered next time you open this "
        f"report in this browser). Click <strong>View</strong> to jump the PDF panel below to "
        f"both matched pages, with the matching text highlighted.</p>",
        "<table class='summary-table'><tr><th>Level</th><th>Matches found</th></tr>",
    ]
    for level, count in level_counts.items():
        parts.append(f"<tr><td>{html.escape(level)}</td><td>{count}</td></tr>")
    parts.append("</table>")

    parts.append(
        "<div id='pdf-panel' class='collapsed'>"
        "<div id='pdf-panel-header'><h3>PDF viewer</h3><span>click to expand/collapse</span></div>"
        "<div class='pdf-frames'>"
        "<div class='pdf-frame-col'><h4>A</h4><iframe id='pdf-frame-a'></iframe></div>"
        "<div class='pdf-frame-col'><h4>B</h4><iframe id='pdf-frame-b'></iframe></div>"
        "</div>"
        f"<p class='pdf-note'>Shows <code>{html.escape(pdf_filename)}</code> jumped to each "
        "matched page, with the first few words of the match highlighted so you can see where "
        "the two sides line up. Needs both that PDF file <em>and</em> the "
        f"<code>{PDFJS_DIR_NAME}/</code> folder to be kept in the same folder as this report "
        "(dupfind copies that folder there automatically). If the panel stays blank, or the "
        "highlight doesn't appear, the page number shown is still correct — open the PDF "
        "yourself and go to that page.</p>"
        "</div>"
    )

    parts.extend(level_tables)
    parts.append(f"<script>{js}</script></body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf_path")
    ap.add_argument("--level",
                     choices=["chunk", "paragraph", "sentence", "growing", "all"],
                     default="growing",
                     help="'growing' (default) uses seed-and-extend to find "
                          "matches of whatever natural length they have, "
                          "instead of a fixed window size")
    ap.add_argument("--chunk-size", type=int, default=12,
                     help="paragraphs per chunk window (default 12, only for --level chunk)")
    ap.add_argument("--window", type=int, default=4,
                     help="sentences per window (default 4, only for --level sentence)")
    ap.add_argument("--min-pair-score", type=int, default=60,
                     help="min per-sentence-pair similarity to keep growing a match (growing mode)")
    ap.add_argument("--min-final-score", type=int, default=60,
                     help="min whole-span similarity to report a multi-sentence match (growing mode)")
    ap.add_argument("--min-single-sentence-score", type=int, default=78,
                     help="min similarity to report a single-sentence match (growing mode); "
                          "stricter than --min-final-score since single-sentence coincidences "
                          "are common")
    ap.add_argument("--context-words", type=int, default=15,
                     help="words of context shown either side of each growing-mode match (default 15)")
    ap.add_argument("--out", default=None,
                     help="markdown report file path (default: <pdf name>.duplicates.md next to the input)")
    ap.add_argument("--html-out", default=None,
                     help="HTML report file path (default: <pdf name>.duplicates.html next to the input)")
    ap.add_argument("--no-html", action="store_true", help="skip writing the HTML report")
    ap.add_argument("--no-markdown", action="store_true", help="skip writing the Markdown report")
    args = ap.parse_args()

    raw_paragraphs, paragraph_pages_raw, page_labels = extract_paragraphs(args.pdf_path)
    paragraphs = raw_paragraphs
    paragraph_pages = PageInfo(paragraph_pages_raw, page_labels)
    print(f"Extracted {len(paragraphs)} paragraphs from {args.pdf_path}", file=sys.stderr)

    if args.level == "all":
        levels = ["growing", "chunk", "paragraph", "sentence"]
    else:
        levels = [args.level]

    md_sections, html_tables = [], []
    level_counts = {}
    for level in levels:
        if level == "growing":
            matches = find_growing_matches(
                paragraphs, paragraph_pages,
                min_pair_score=args.min_pair_score,
                min_final_score=args.min_final_score,
                min_single_sentence_score=args.min_single_sentence_score,
                context_words=args.context_words,
            )
            level_counts["growing"] = len(matches)
            md_sections.append(build_growing_report_md(matches, args.pdf_path))
            html_tables.append(_html_table("growing", "Growing (seed-and-extend) matches", _growing_rows(matches)))
        else:
            matches = run_level(level, paragraphs, paragraph_pages, args)
            level_counts[level] = len(matches)
            md_sections.append(build_fixed_report_md(level, matches))
            html_tables.append(_html_table(level, f"{level.capitalize()}-level matches", _fixed_rows(matches)))

    base = re.sub(r"\.pdf$", "", args.pdf_path, flags=re.I)

    if not args.no_markdown:
        report = build_summary(args.pdf_path, len(paragraphs), level_counts)
        report += "\n" + "\n".join(md_sections)
        md_path = args.out or (base + ".duplicates.md")
        with open(md_path, "w") as f:
            f.write(report)
        print(f"Markdown report written to {md_path}", file=sys.stderr)

    if not args.no_html:
        pdf_filename = os.path.basename(args.pdf_path)
        html_report = build_html_report(args.pdf_path, pdf_filename, len(paragraphs),
                                         level_counts, html_tables)
        html_path = args.html_out or (base + ".duplicates.html")
        with open(html_path, "w") as f:
            f.write(html_report)
        _copy_pdfjs_viewer(os.path.dirname(os.path.abspath(html_path)))
        print(f"HTML report written to {html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
