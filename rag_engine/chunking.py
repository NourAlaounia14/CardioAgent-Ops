"""rag_engine/chunking.py

Extraction PDF -> nettoyage -> chunking par sections, pour le RAG.

Principe : on respecte la structure du document (titres, sous-titres) au lieu de
couper tous les N caractères. Chaque chunk garde sa section et sa page, ce qui
permet de citer la source dans la réponse du LLM et de filtrer la recherche.

Usage (aperçu SANS embedding ni Qdrant, pour vérifier la qualité des chunks) :
    python -m rag_engine.chunking --pdf-dir data/docs --preview 5

Fichier optionnel  <pdf-dir>/docs_meta.json  pour fixer titre/type de chaque PDF :
    {"mon_fichier.pdf": {"title": "...", "source_type": "cours"}}
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .config import CHUNKS_PATH

CHUNK_SIZE = 900      # caractères (~300 tokens en français) : tient dans les 512 tokens du modèle
OVERLAP = 150         # chevauchement entre chunks voisins d'une même section
MIN_CHARS = 40        # chunks plus courts : ignorés

# ---- règles de nettoyage (adaptables) ----------------------------------------
COVER_RE = re.compile(r"HAL is a multi-disciplinary|archive ouverte pluridisciplinaire HAL|To cite this version", re.I)
START_RE = re.compile(r"^(points clefs|résumé|abstract|introduction|i\s*/\s*introduction)\b", re.I)
STOP_RE = re.compile(r"^(references|références|bibliographie)\s*:?\s*$", re.I)
DROP_RE = re.compile(r"^(plan du cours|objectifs du cours|pré-requis|figure\s*\d+|fig\.\s*\d+)\b", re.I)
PAGE_NO_RE = re.compile(r"^(page\s*)?\d{1,3}(\s*/\s*\d{1,3})?$", re.I)
BULLET_RE = re.compile(r"^(?:[-–—•*▪●◦]|\d{1,2}\s*[-.)/]\s*(?=[A-ZÀ-Ý])|[a-z]\)\s)")
ROMAN_LETTER_RE = re.compile(r"^[IVX]{1,4}\s*/\s*[A-Z]\b")          # "III/A : ..."   -> sous-titre
ROMAN_RE = re.compile(r"^[IVX]{1,4}\s*/")                              # "II/ Technique" -> titre
LETTER_RE = re.compile(r"^(?![IVX]\s*/)[A-Z]\s*/")                    # "A/ Conditions" -> sous-titre
ESC_RE = re.compile(r"soci[ée]t[ée] europ[ée]enne de cardiologie|European Society of Cardiology|\bESC\b", re.I)


# ============================================================ 1. extraction
def extract_pages(pdf_path) -> list[str]:
    try:
        import pymupdf
    except ImportError:  # anciennes versions
        import fitz as pymupdf
    with pymupdf.open(str(pdf_path)) as doc:
        return [p.get_text("text", sort=True) for p in doc]


# ============================================================ 2. nettoyage
def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("\xad", "").replace("\u200b", "")
    return re.sub(r"[ \t\u00a0]+", " ", t)


def clean_pages(raw_pages: list[str]) -> list[tuple[int, str]]:
    """-> liste de (n° de page, ligne). '' = ligne vide. Retire couvertures,
    en-têtes/pieds de page répétés et numéros de page."""
    pages = []
    for i, t in enumerate(raw_pages, 1):
        if COVER_RE.search(t):
            continue
        pages.append((i, [l.strip() for l in _norm(t).splitlines()]))

    repeated: set[str] = set()
    if len(pages) >= 4:
        cnt = Counter(l for _, ls in pages for l in set(ls) if 6 <= len(l) <= 80)
        thr = max(3, int(0.4 * len(pages)))
        repeated = {l for l, c in cnt.items() if c >= thr}

    lines = []
    for i, ls in pages:
        for l in ls:
            if l in repeated or PAGE_NO_RE.match(l):
                continue
            lines.append((i, l))
        lines.append((i, ""))
    return lines


# ====================================================== 3. structure (titres)
def heading_level(s: str, prev_terminal: bool, nxt: str) -> int:
    """0 = texte, 1 = titre, 2 = sous-titre."""
    if len(s) > 110:
        return 0
    if ROMAN_LETTER_RE.match(s):
        return 2
    if ROMAN_RE.match(s):
        return 1
    if LETTER_RE.match(s):
        return 2
    if (prev_terminal and len(s) <= 75 and len(s.split()) <= 12 and s[0].isupper()
            and not s.endswith((".", ",", ";")) and not BULLET_RE.match(s)
            and not (nxt and nxt[0].islower())):
        return 1
    return 0


def clean_title(s: str) -> str:
    return re.sub(r"\s*[:\-]\s*$", "", s).strip()[:100]


def parse(lines: list[tuple[int, str]]) -> list[tuple]:
    """Événements : ('h', niveau, titre) ou ('p', texte, page)."""
    events: list[tuple] = []
    texts = [s for _, s in lines]
    started = not any(START_RE.match(s) for s in texts)   # saute page de titre / affiliations
    buf, buf_page, prev_terminal = "", None, True

    def flush():
        nonlocal buf, buf_page
        if buf.strip():
            events.append(("p", buf.strip(), buf_page))
        buf, buf_page = "", None

    for i, (page, s) in enumerate(lines):
        if not s:
            flush()
            prev_terminal = True
            continue
        if not started:
            if START_RE.match(s):
                started = True
            else:
                continue
        if STOP_RE.match(s):
            break
        if DROP_RE.match(s):
            flush()
            prev_terminal = True
            continue
        nxt = next((t for t in texts[i + 1 : i + 5] if t), "")
        lvl = heading_level(s, prev_terminal, nxt)
        if lvl:
            flush()
            events.append(("h", lvl, clean_title(s)))
            prev_terminal = True
            continue
        if buf and (BULLET_RE.match(s) or (buf[-1] in ".!?" and s[0].isupper())):
            flush()
        if not buf:
            buf, buf_page = s, page
        elif buf.endswith("-") and s[:1].islower():
            buf += s
        else:
            buf += " " + s
        prev_terminal = s[-1] in ".!?:;"
    flush()
    return events


# ============================================================== 4. découpage
def _split_long(para: str, size: int) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+", para):
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def _tail(text: str, n: int) -> str:
    if len(text) <= n:
        return ""
    t = text[-n:]
    i = t.find(" ")
    return t[i + 1 :] if i != -1 else t


def pack(units: list[tuple[str, int]], size: int, overlap: int) -> list[tuple[str, int]]:
    """Regroupe des paragraphes en chunks <= ~size caractères, avec chevauchement."""
    expanded = []
    for t, p in units:
        expanded.extend([(t, p)] if len(t) <= size else [(s, p) for s in _split_long(t, size)])
    chunks, cur, cur_page = [], "", None
    for t, p in expanded:
        if cur and len(cur) + 1 + len(t) > size:
            chunks.append((cur, cur_page))
            cur, cur_page = (_tail(cur, overlap) + " " + t).strip(), p
        else:
            if not cur:
                cur_page = p
            cur = f"{cur}\n{t}" if cur else t
    if cur:
        chunks.append((cur, cur_page))
    return chunks


def chunk_document(lines, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    stack: list[str] = []
    units: list[tuple[str, int]] = []
    out: list[dict] = []

    def flush():
        nonlocal units
        if units:
            section = " › ".join(stack) or "Général"
            for text, page in pack(units, size, overlap):
                if len(text) >= MIN_CHARS:
                    out.append(dict(section=section, page=page, text=text))
        units = []

    for ev in parse(lines):
        if ev[0] == "h":
            flush()
            _, lvl, title = ev
            stack[:] = ([stack[0]] if (lvl == 2 and stack) else []) + [title]
        else:
            units.append((ev[1], ev[2]))
    flush()
    return out


def chunk_raw_pages(raw_pages: list[str], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    return chunk_document(clean_pages(raw_pages), size, overlap)


def contextual_text(c: dict) -> str:
    """Texte réellement embarqué : le titre/section aide la recherche sémantique."""
    return f"{c['title']} — {c['section']}\n{c['text']}"


# ================================================================ 5. corpus
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).strip("-")


def load_corpus(pdf_dir: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    d = Path(pdf_dir)
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Aucun .pdf dans {d}")
    meta_path = d / "docs_meta.json"
    overrides = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    corpus = []
    for pdf in pdfs:
        parts = chunk_raw_pages(extract_pages(pdf), size, overlap)
        ov = overrides.get(pdf.name, {})
        full = " ".join(p["text"] for p in parts)
        base = dict(
            doc_id=ov.get("doc_id") or _slug(pdf.stem),
            title=ov.get("title") or re.sub(r"[_\-]+", " ", pdf.stem).strip(),
            source_file=pdf.name,
            source_type=ov.get("source_type") or ("recommandation_esc" if ESC_RE.search(full) else "cours"),
            lang=ov.get("lang", "fr"),
        )
        for i, p in enumerate(parts):
            corpus.append({
                **base,
                **p,
                "chunk_id": f"{base['doc_id']}:{pdf.name}:{i}",
                "source": base["source_file"],
                "page_start": p["page"],
                "heading": p["section"],
                "chunk_index": i,
                "n_chars": len(p["text"]),
            })
        print(f"{pdf.name}: {len(parts)} chunks (type={base['source_type']})")
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/docs")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--out", default=str(CHUNKS_PATH))
    ap.add_argument("--preview", type=int, default=0, help="affiche N chunks par document")
    a = ap.parse_args()

    corpus = load_corpus(a.pdf_dir, a.chunk_size, a.overlap)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lens = [c["n_chars"] for c in corpus]
    print(f"\nTotal : {len(corpus)} chunks | longueur min/moy/max = "
          f"{min(lens)}/{sum(lens) // len(lens)}/{max(lens)} caractères -> {out}")
    if a.preview:
        shown = Counter()
        for c in corpus:
            if shown[c["doc_id"]] < a.preview:
                shown[c["doc_id"]] += 1
                print(f"\n--- [{c['doc_id']} #{c['chunk_index']}] p.{c['page']} | {c['section']}\n{c['text']}")
if __name__ == "__main__":
    main()