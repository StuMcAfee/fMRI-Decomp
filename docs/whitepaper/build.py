"""Build the white paper: embed figures (or a marked placeholder) and render a PDF."""
import base64, re, sys
from pathlib import Path
here = Path(__file__).parent
src = (here / "whitepaper_src.html").read_text()
figs = here / "figs"
missing = []
def sub(m):
    attrs, inner = m.group(1), m.group(2)
    name = re.search(r'data-src="([^"]+)"', attrs).group(1)
    pend = re.search(r'data-pending="([^"]+)"', attrs)
    f = figs / name
    if f.exists():
        b64 = base64.b64encode(f.read_bytes()).decode()
        alt = re.sub("<[^>]+>", "", inner).strip()[:300].replace('"', "'")
        body = f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
    else:
        missing.append(name)
        body = f'<div class="pending-fig"><b>Figure from the cluster run (pending):</b> {pend.group(1) if pend else ""}<br><code>{name}</code></div>'
    return f"<figure>{body}{inner}</figure>"
out = re.sub(r"<figure( data-src=[^>]*)>(.*?)</figure>", sub, src, flags=re.S)
(here / "whitepaper.html").write_text(out)
print("pending:", missing)
import subprocess
subprocess.run(["node", str(here / "pdf.js")], check=True)
print("built")
