import shutil
from pathlib import Path
import kagglehub

def download_csic_2010(dest):
    path = Path(kagglehub.dataset_download("ispangler/csic-2010-web-application-attacks"))

    dest.mkdir(parents=True, exist_ok=True)

    for item in path.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"CSIC-2010 dataset downloaded to {dest}")