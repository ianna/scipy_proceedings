#!/usr/bin/env bash
# Rebuilds the render toolchain (ephemeral across sessions). Run once per session.
set -e
sudo apt-get update -q
sudo apt-get install -y -q nodejs npm poppler-utils \
  libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
  libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
  libnss3 libnspr4 libatspi2.0-0 fonts-liberation
sudo npm install -g @marp-team/marp-cli
pip install -q matplotlib
mkdir -p ~/.local/chrome && cd ~/.local/chrome
URL=$(python3 -c "import urllib.request,json; d=json.load(urllib.request.urlopen('https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json')); s=d['channels']['Stable']; print([x['url'] for x in s['downloads']['chrome'] if x['platform']=='linux64'][0])")
curl -sSL "$URL" -o c.zip && unzip -oq c.zip && rm c.zip
printf '#!/usr/bin/env bash\nexec ~/.local/chrome/chrome-linux64/chrome --no-sandbox --disable-gpu --disable-dev-shm-usage "$@"\n' > chrome-nosandbox.sh
chmod +x chrome-nosandbox.sh chrome-linux64/chrome
echo "toolchain ready. Render: CHROME_PATH=~/.local/chrome/chrome-nosandbox.sh marp --no-stdin --pdf --allow-local-files --browser chrome slides.md -o slides.pdf"
