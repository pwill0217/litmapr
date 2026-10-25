from litmapr.cache import PageCache, file_hash


def test_file_hash_is_stable(sample_cbz):
    assert file_hash(sample_cbz) == file_hash(sample_cbz)


def test_file_hash_follows_contents_not_name(sample_cbz, tmp_path):
    renamed = tmp_path / "renamed.cbz"
    renamed.write_bytes(sample_cbz.read_bytes())
    assert file_hash(renamed) == file_hash(sample_cbz)


def test_file_hash_changes_when_contents_change(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert file_hash(a) != file_hash(b)


def test_cache_miss_returns_none(tmp_path):
    cache = PageCache("abc", "qwen2.5vl:3b", cache_dir=tmp_path)
    assert cache.get(1) is None


def test_cache_remembers_across_runs(tmp_path):
    PageCache("abc", "qwen2.5vl:3b", cache_dir=tmp_path).put(1, "BOOM!", "A scene.")
    reopened = PageCache("abc", "qwen2.5vl:3b", cache_dir=tmp_path)
    assert reopened.get(1) == {"text": "BOOM!", "scene": "A scene."}


def test_cache_is_separate_per_model(tmp_path):
    PageCache("abc", "qwen2.5vl:3b", cache_dir=tmp_path).put(1, "BOOM!", "A scene.")
    other_model = PageCache("abc", "qwen3.5:2b", cache_dir=tmp_path)
    assert other_model.get(1) is None


def test_cache_filename_has_no_colon(tmp_path):
    cache = PageCache("abc", "qwen2.5vl:3b", cache_dir=tmp_path)
    assert ":" not in cache.path.name  # colons aren't allowed in Windows filenames
