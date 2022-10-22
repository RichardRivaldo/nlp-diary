#!/bin/sh
pip install -q -r requirements.txt

waitress-serve --port=8000 app.app:app