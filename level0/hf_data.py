"""Download the project data from the public Hugging Face dataset into data/.

Screenshots come packed in tar shards; they are unpacked into img/ folders. Features (*.npy) are not included:
rebuild them from the screenshots with the encode scripts.

  python level0/hf_data.py fetch lamshiu/gui-world-model-data [SETS]   SETS = comma list, e.g. level0,snake
"""
import sys, tarfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def fetch(repo, sets=None):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type="dataset", local_dir=DATA, max_workers=16, allow_patterns=[f"{s}/**" for s in sets] if sets else None)
    for t in sorted(DATA.rglob("img-*.tar")):
        (t.parent / "img").mkdir(exist_ok=True)
        with tarfile.open(t) as tar:
            tar.extractall(t.parent / "img", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        t.unlink()
        print(f"unpacked {t.relative_to(DATA)}", flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "fetch", __doc__
    fetch(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
