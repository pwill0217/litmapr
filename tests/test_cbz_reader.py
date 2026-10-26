from litmapr.ingest.cbz_reader import CbzReader


def test_reads_every_page(sample_cbz):
    pages = list(CbzReader(sample_cbz).pages())
    assert len(pages) == 3


def test_pages_numbered_from_one(sample_cbz):
    pages = list(CbzReader(sample_cbz).pages())
    assert [page.number for page in pages] == [1, 2, 3]


def test_pages_have_image_bytes(sample_cbz):
    pages = list(CbzReader(sample_cbz).pages())
    assert all(len(page.image) > 0 for page in pages)
    assert all(page.text is None for page in pages)


def test_metadata_from_comicinfo(tagged_cbz):
    assert CbzReader(tagged_cbz).metadata() == {
        "series": "FF",
        "issue": "5",
        "year": 2012,
        "month": 3,
    }


def test_metadata_missing_is_empty(sample_cbz):
    assert CbzReader(sample_cbz).metadata() == {}
