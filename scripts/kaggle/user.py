"""Print the Kaggle username from the API key file (never the key itself)."""

import json
import os
import subprocess
import sys
from pathlib import Path

if os.environ.get("KAGGLE_USERNAME"):
    print(os.environ["KAGGLE_USERNAME"])
    sys.exit()
for path in (Path.home() / ".kaggle/kaggle.json", Path.home() / ".config/kaggle/kaggle.json"):
    if path.is_file():
        print(json.loads(path.read_text())["username"])
        sys.exit()
# newer access tokens (~/.kaggle/access_token) carry no username: ask the CLI
out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
for line in out.splitlines():
    key, _, value = line.strip().lstrip("- ").partition(":")
    if key.strip() == "username" and value.strip() and value.strip() != "None":
        print(value.strip())
        sys.exit()
sys.exit("no Kaggle API key found: see https://www.kaggle.com/settings (API → Create New Token)")
