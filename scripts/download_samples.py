"""Download the sample cabin video used in the README.

driver-action-recognition.mp4 is from https://github.com/intel-iot-devkit/sample-videos
(CC BY 4.0): three drivers, about 7 minutes, recorded from a camera on the dashboard.
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/"
VIDEOS = ["driver-action-recognition.mp4"]
DEST = Path(__file__).resolve().parent.parent / "samples"


def main():
    DEST.mkdir(exist_ok=True)
    for name in VIDEOS:
        target = DEST / name
        if target.exists():
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(BASE + name, target)
        except Exception as e:
            print(f"failed ({e})")
            target.unlink(missing_ok=True)
            return 1
        print(f"{target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
