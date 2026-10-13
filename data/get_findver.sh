#!/usr/bin/env bash
# Clone FinDVer (MIT) at the commit used for the paper into data/findver/.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d findver ]; then
  git clone https://github.com/yilunzhao/FinDVer findver
fi
git -C findver checkout -q e8bb237
echo "FinDVer ready in data/findver ($(ls findver/financial_reports | wc -l) report folders)"
