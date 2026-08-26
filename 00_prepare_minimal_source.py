
from pathlib import Path
import shutil

OLD = Path(r"D:\47\472\New-Papers\GIS\Codes")
NEW = OLD/"New_Branch"

for name in ["data_loader.py","models_sacu.py","physics_metrics.py","utils.py"]:
    src = OLD/"src"/name
    dst = NEW/"src"/name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src,dst)
        print("[COPY]", src, "->", dst)
    else:
        print("[SKIP]", dst)

for d in [NEW/"src", NEW/"src"/"training", NEW/"src"/"evaluation"]:
    d.mkdir(parents=True, exist_ok=True)
    (d/"__init__.py").touch(exist_ok=True)
