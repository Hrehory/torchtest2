#!/usr/bin/env bash
# The diagrams of torchtest2 have one source each (*.mmd). This script renders them to PDF for
# the tutorial slides, and checks that docs/architecture.md embeds exactly the same
# sources (GitHub and GitLab render mermaid in Markdown directly).
# Needs node (npx) and Chrome or Chromium; CHROME=/path/to/chrome overrides the detection.
set -euo pipefail
cd "$(dirname "$0")"
CHROME=${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)}
config=$(mktemp)
trap 'rm -f "$config"' EXIT
printf '{"executablePath": "%s", "args": ["--no-sandbox"]}\n' "$CHROME" > "$config"
for source in *.mmd; do
  npx -y @mermaid-js/mermaid-cli@11 -q -p "$config" -c mermaid.json \
    -i "$source" -o "${source%.mmd}.pdf" --pdfFit
  echo "rendered ${source%.mmd}.pdf"
done
python3 - <<'PYEOF'
import pathlib, sys
doc = pathlib.Path("../architecture.md").read_text()
stale = [p.name for p in sorted(pathlib.Path(".").glob("*.mmd")) if p.read_text().strip() not in doc]
if stale:
    sys.exit(f"docs/architecture.md does not embed the current source of: {', '.join(stale)}")
print("docs/architecture.md embeds every diagram source")
PYEOF
