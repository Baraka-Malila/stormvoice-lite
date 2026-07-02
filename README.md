# speaker_id

CA studies group assignment — **speaker identification** as a pattern recognition task.

Given a short audio clip, name who spoke. 4 classes = 4 group members.

See [CLAUDE.md](CLAUDE.md) for the full project brief, phased scope, and structure.

## Quickstart (once venv exists)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Each member records themselves
python scripts/record.py --speaker alice --minutes 10

# Train the models on collected data
python scripts/train.py

# Live demo
python scripts/serve.py
# open http://localhost:8000
```
