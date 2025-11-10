# Arvane

Arvane
    |__ config/
    |__ source/
    |__ .env
    |__ README.md
    |__ requirements.txt

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dotenv --file .env run -- python -m source.main

