import pymupdf

from litmapr.ingest.base import Page, Reader

# Resolution for page images. 150 DPI is sharp enough for OCR later
# without making every page image huge.
RENDER_DPI = 150


class PdfReader(Reader):
    def pages(self):
        # "with" closes the file automatically when the loop finishes
        with pymupdf.open(self.path) as document:
            for number, pdf_page in enumerate(document, start=1):
                # Prose PDFs have a text layer. Scanned/comic pages don't,
                # so get_text() comes back empty -> None means "needs OCR later".
                text = pdf_page.get_text().strip() or None
                # Render the page to a PNG so every page has an image,
                # whether or not it had text.
                image = pdf_page.get_pixmap(dpi=RENDER_DPI).tobytes("png")
                yield Page(number=number, image=image, text=text, path=None)
