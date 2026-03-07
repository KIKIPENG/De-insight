#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/backend/.venv"
REQUIREMENTS="$ROOT_DIR/backend/requirements.txt"
ENV_FILE="$ROOT_DIR/.env"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/deinsight"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ first."
  exit 1
fi

echo "[1/4] Creating backend virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

echo "[2/4] Installing backend dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo "[3/4] Preparing .env..."
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
LLM_MODEL=anthropic/claude-sonnet-4-20250514
ANTHROPIC_API_KEY=
EOF
  echo "Created .env. Please fill in your API key."
fi

echo "[4/4] Installing launcher command..."
mkdir -p "$BIN_DIR"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "$ROOT_DIR/scripts/deinsight" "\$@"
EOF
chmod +x "$LAUNCHER"

chmod +x "$ROOT_DIR/scripts/deinsight" "$ROOT_DIR/De-insight.command"

echo
echo "Install complete."
echo "Next:"
echo "  1) Add key to: $ENV_FILE"
echo "  2) Start app: deinsight"
echo
echo "If 'deinsight' is not found, add this to your shell config:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
