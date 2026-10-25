from cbz import ComicInfo

from litmapr.ingest.base import Page, Reader


class CbzReader(Reader):
    def pages(self):
        # 1. load the comic from self.path
        comic = ComicInfo.from_cbz(self.path)
        # 2. loop over its pages, keeping track of the page number
        for i, page in enumerate(comic.pages):
            # 2a. read the image bytes
            # 2b. yield a Page object with the number, image bytes, and no text or path
            yield Page(number=i + 1, image=page.content, text=None, path=None)
