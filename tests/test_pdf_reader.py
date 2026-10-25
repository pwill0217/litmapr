from litmapr.ingest.pdf_reader import PdfReader


def test_reads_every_page(sample_pdf):
    pages = list(PdfReader(sample_pdf).pages())
    assert [page.number for page in pages] == [1, 2]


def test_text_page_has_text(sample_pdf):
    first = next(iter(PdfReader(sample_pdf).pages()))
    assert "Reed Richards" in first.text


def test_blank_page_text_is_none(sample_pdf):
    pages = list(PdfReader(sample_pdf).pages())
    assert pages[1].text is None


def test_every_page_has_png_image(sample_pdf):
    pages = list(PdfReader(sample_pdf).pages())
    png_signature = b"\x89PNG"
    assert all(page.image.startswith(png_signature) for page in pages)
