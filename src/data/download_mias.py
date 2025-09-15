from pathlib import Path
import shutil
import kagglehub

def main():
    # 1) Download Kaggle MIAS (kmader/mias-mammography)
    path = kagglehub.dataset_download("kmader/mias-mammography")
    print("KaggleHub path:", path)

    # 2) Mirror to our repo data/raw
    src = Path(path)
    dst = Path(__file__).resolve().parents[2] / "data" / "raw" / "mias"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("Removing existing:", dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("Copied to:", dst)

if __name__ == "__main__":
    main()
