"""
Image Extraction and Visual Grounding Utilities
================================================
Extracts embedded figures, charts, and diagrams from PDFs, Word documents,
and standalone image files, linking them to their captions, page numbers,
and surrounding contextual text for associative Hopfield memory indexing.
"""

import os
import re
import io
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image


class ExtractedImageInfo:
    """Represents an extracted visual figure and its spatial/textual context."""

    def __init__(
        self,
        image_url: str,
        file_path: str,
        caption: str,
        context_text: str,
        page_number: int,
        source: str,
        width: int,
        height: int,
    ):
        self.image_url = image_url
        self.file_path = file_path
        self.caption = caption
        self.context_text = context_text
        self.page_number = page_number
        self.source = source
        self.width = width
        self.height = height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_url": self.image_url,
            "caption": self.caption,
            "context_text": self.context_text,
            "page_number": self.page_number,
            "source": self.source,
            "width": self.width,
            "height": self.height,
        }


def _clean_stem(filename: str) -> str:
    """Create a URL-safe, alphanumeric stem from a filename."""
    stem = Path(filename).stem
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)
    return clean[:40]


def _extract_caption_candidates(text: str) -> List[str]:
    """Find figure/chart/table captions in text."""
    patterns = [
        r'(?:Figure|Fig\.?|Chart|Diagram|Illustration|Table)\s*(\d+)?[:.\-]?\s*([^\n\r.]+)',
        r'(?:Architecture|Workflow|Overview|System)\s+(?:Diagram|Chart|Pipeline|Model)[:.\-]?\s*([^\n\r.]+)',
    ]
    captions = []
    for pat in patterns:
        matches = re.finditer(pat, text, re.IGNORECASE)
        for m in matches:
            full_match = m.group(0).strip()
            if len(full_match) >= 6:
                captions.append(full_match)
    return captions


def extract_images_from_pdf(
    file_bytes: bytes,
    filename: str,
    output_dir: str = "static/extracted_images",
    min_width: int = 80,
    min_height: int = 80,
    min_bytes: int = 1500,
) -> List[ExtractedImageInfo]:
    """
    Extract figures and diagrams from PDF pages using pypdf.
    Links each image with page text and any nearby figure captions.
    """
    os.makedirs(output_dir, exist_ok=True)
    images: List[ExtractedImageInfo] = []
    clean_name = _clean_stem(filename)

    try:
        import pypdf
        bio = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(bio)

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = ""
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            captions_found = _extract_caption_candidates(page_text)

            # Extract images from page
            try:
                page_images = page.images
            except Exception:
                continue

            for img_idx, img_file in enumerate(page_images, start=1):
                try:
                    data = img_file.data
                    if len(data) < min_bytes:
                        continue

                    # Verify image validity and dimensions via PIL
                    pil_img = Image.open(io.BytesIO(data))
                    w, h = pil_img.size
                    if w < min_width or h < min_height:
                        continue

                    # Determine format and save
                    img_hash = hashlib.md5(data).hexdigest()[:8]
                    fmt = (pil_img.format or "PNG").lower()
                    if fmt not in ("png", "jpeg", "jpg", "webp"):
                        fmt = "png"
                    out_filename = f"{clean_name}_p{page_idx}_fig{img_idx}_{img_hash}.{fmt}"
                    out_path = os.path.join(output_dir, out_filename)

                    pil_img.save(out_path)

                    # Determine best caption
                    if img_idx <= len(captions_found):
                        caption = captions_found[img_idx - 1]
                    elif captions_found:
                        caption = captions_found[0]
                    else:
                        caption = f"Figure from {filename} (Page {page_idx})"

                    # Context snippet from surrounding page text
                    clean_page_text = re.sub(r'\s+', ' ', page_text).strip()
                    context_snippet = clean_page_text[:400] if clean_page_text else f"Illustration on page {page_idx}"

                    images.append(
                        ExtractedImageInfo(
                            image_url=f"/extracted_images/{out_filename}",
                            file_path=out_path,
                            caption=caption,
                            context_text=context_snippet,
                            page_number=page_idx,
                            source=filename,
                            width=w,
                            height=h,
                        )
                    )
                except Exception as ie:
                    continue

    except Exception as e:
        print(f"[ImageExtractor] PDF extraction notice: {e}")

    return images


def extract_images_from_docx(
    file_bytes: bytes,
    filename: str,
    output_dir: str = "static/extracted_images",
    min_width: int = 80,
    min_height: int = 80,
) -> List[ExtractedImageInfo]:
    """Extract embedded images from DOCX documents."""
    os.makedirs(output_dir, exist_ok=True)
    images: List[ExtractedImageInfo] = []
    clean_name = _clean_stem(filename)

    try:
        from docx import Document
        bio = io.BytesIO(file_bytes)
        doc = Document(bio)

        doc_text = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
        captions_found = _extract_caption_candidates(doc_text)

        img_idx = 1
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    data = rel.target_part.blob
                    pil_img = Image.open(io.BytesIO(data))
                    w, h = pil_img.size
                    if w < min_width or h < min_height:
                        continue

                    img_hash = hashlib.md5(data).hexdigest()[:8]
                    fmt = (pil_img.format or "PNG").lower()
                    if fmt not in ("png", "jpeg", "jpg", "webp"):
                        fmt = "png"
                    out_filename = f"{clean_name}_img{img_idx}_{img_hash}.{fmt}"
                    out_path = os.path.join(output_dir, out_filename)
                    pil_img.save(out_path)

                    caption = captions_found[img_idx - 1] if (img_idx <= len(captions_found)) else f"Figure {img_idx} from {filename}"

                    images.append(
                        ExtractedImageInfo(
                            image_url=f"/extracted_images/{out_filename}",
                            file_path=out_path,
                            caption=caption,
                            context_text=doc_text[:350],
                            page_number=1,
                            source=filename,
                            width=w,
                            height=h,
                        )
                    )
                    img_idx += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"[ImageExtractor] DOCX extraction notice: {e}")

    return images


def save_standalone_image(
    file_bytes: bytes,
    filename: str,
    output_dir: str = "static/extracted_images",
) -> Optional[ExtractedImageInfo]:
    """Process a directly uploaded image file (PNG, JPG, WEBP)."""
    os.makedirs(output_dir, exist_ok=True)
    clean_name = _clean_stem(filename)

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        w, h = pil_img.size
        img_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        fmt = (pil_img.format or "PNG").lower()
        if fmt not in ("png", "jpeg", "jpg", "webp"):
            fmt = "png"
        out_filename = f"{clean_name}_{img_hash}.{fmt}"
        out_path = os.path.join(output_dir, out_filename)
        pil_img.save(out_path)

        # Generate human-friendly title from filename
        title = re.sub(r'[_\\-]+', ' ', Path(filename).stem).title()
        caption = f"Uploaded Visual: {title}"
        context_text = f"Graphic and visual document containing diagram or figure titled '{title}'."

        return ExtractedImageInfo(
            image_url=f"/extracted_images/{out_filename}",
            file_path=out_path,
            caption=caption,
            context_text=context_text,
            page_number=1,
            source=filename,
            width=w,
            height=h,
        )
    except Exception as e:
        print(f"[ImageExtractor] Direct image save notice: {e}")
        return None


def extract_images_from_file_bytes(
    file_bytes: bytes,
    filename: str,
    output_dir: str = "static/extracted_images",
) -> List[ExtractedImageInfo]:
    """
    Unified entry point: Extracts images from any supported file format.
    Supports PDF, DOCX, and direct image files (PNG, JPG, WEBP).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_images_from_pdf(file_bytes, filename, output_dir=output_dir)
    elif ext == ".docx":
        return extract_images_from_docx(file_bytes, filename, output_dir=output_dir)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        img = save_standalone_image(file_bytes, filename, output_dir=output_dir)
        return [img] if img else []
    return []
