#!/bin/sh
pip install -r requirements.txt

waitress-serve --port=8000 app.app:app