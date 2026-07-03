#!/usr/bin/env bash
# stormVoice Lite — teammate onboarding script
# Usage: bash setup.sh
set -e

PYTHON=${PYTHON:-python3}

echo "=== stormVoice Lite setup ==="
echo ""

# 1. Virtual environment
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment…"
  $PYTHON -m venv venv
else
  echo "→ Virtual environment already exists"
fi

source venv/bin/activate
echo "→ Activated venv"

# 2. Install dependencies
echo "→ Installing requirements…"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "→ Dependencies installed"

# 3. Placeholder data
echo ""
echo "→ Generating placeholder audio data (60 clips × 4 speakers)…"
python scripts/generate_placeholder_data.py

# 4. Models directory
mkdir -p models

echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " NEXT STEPS:"
echo ""
echo " 1. Train the models (run these notebooks in Jupyter):"
echo "      jupyter notebook notebooks/01_classical_pipeline.ipynb"
echo "      jupyter notebook notebooks/02_neural_pipeline.ipynb"
echo ""
echo " 2. Replace placeholder data with real recordings:"
echo "      Record WAV files and place them in:"
echo "      data/recordings/<your_name>/*.wav"
echo "      Then re-run both notebooks."
echo ""
echo " 3. Start the server:"
echo "      uvicorn scripts.serve:app --reload"
echo "      open http://127.0.0.1:8000"
echo ""
echo " Optional — use a different port (e.g. if 8000 is taken):"
echo "      uvicorn scripts.serve:app --reload --port 8001"
echo ""
echo " Optional — PostgreSQL instead of SQLite:"
echo "      export DATABASE_URL=postgresql://user:pass@host/dbname"
echo "      uvicorn scripts.serve:app --reload"
echo "============================================================"
