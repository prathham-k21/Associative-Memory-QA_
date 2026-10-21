"""
Tests for Visual Figures Extraction, Hopfield Indexing, and Retrieval
"""

import os
import io
import pytest
from PIL import Image, ImageDraw
from utils.image_extractor import (
    extract_images_from_file_bytes,
    save_standalone_image,
    ExtractedImageInfo
)
from core.pipeline import HopfieldQAPipeline, PipelineConfig


def test_standalone_image_saving(tmp_path):
    output_dir = str(tmp_path / "extracted_images")
    # Create test image
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Architecture Flowchart", fill=(255, 255, 255))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    content = bio.getvalue()

    result = save_standalone_image(content, "network_architecture.png", output_dir=output_dir)
    assert result is not None
    assert isinstance(result, ExtractedImageInfo)
    assert "network_architecture" in result.image_url
    assert os.path.exists(result.file_path)
    assert result.width == 200
    assert result.height == 200


def test_pdf_image_extraction_mock(tmp_path):
    output_dir = str(tmp_path / "extracted_images")
    import pypdf

    # Create a small valid PDF with an image
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    img = Image.new("RGB", (120, 120), color=(50, 100, 150))
    img_bio = io.BytesIO()
    img.save(img_bio, format="PNG")

    # Add image to page
    try:
        page.add_image(pypdf.generic.DecodedStreamObject(), "img1")
    except Exception:
        pass

    pdf_bio = io.BytesIO()
    writer.write(pdf_bio)
    pdf_bytes = pdf_bio.getvalue()

    # Should not raise exception
    res = extract_images_from_file_bytes(pdf_bytes, "test_doc.pdf", output_dir=output_dir)
    assert isinstance(res, list)


def test_pipeline_visual_indexing_and_retrieval(tmp_path):
    config = PipelineConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=256,
        hopfield_beta=8.0,
        top_k=3,
        persistence_dir=str(tmp_path / "data")
    )
    pipeline = HopfieldQAPipeline(config)

    # Mock extracted visual figure
    mock_image = ExtractedImageInfo(
        image_url="/extracted_images/transformer_hopfield_diagram.png",
        file_path="/tmp/transformer_hopfield_diagram.png",
        caption="Figure 3: Modern Hopfield Transformer Equivalence Architecture",
        context_text="Detailed schematic showing the equivalence between continuous Hopfield energy minimization and single-step softmax self-attention.",
        page_number=4,
        source="hopfield_paper.pdf",
        width=800,
        height=600
    )

    doc_text = "Hopfield networks provide an associative memory framework. Ramsauer et al. introduced continuous state modern Hopfield networks."
    pipeline.add_document(doc_text, source="hopfield_paper.pdf", images=[mock_image])

    assert pipeline.n_documents == 1
    assert pipeline.n_chunks >= 2  # Text chunk + Image chunk

    # Query specifically seeking the visual diagram
    res = pipeline.query("Show me the Modern Hopfield Transformer architecture diagram")
    assert res is not None

    # Check that retrieved evidence chunks contain the visual figure
    image_chunks = [c for c in res.evidence_collection.chunks if getattr(c, "metadata", {}).get("has_image")]
    assert len(image_chunks) > 0
    top_img_chunk = image_chunks[0]
    assert top_img_chunk.metadata.get("image_url") == "/extracted_images/transformer_hopfield_diagram.png"
    assert "Figure 3" in top_img_chunk.metadata.get("image_caption")
