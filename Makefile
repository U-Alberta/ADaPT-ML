init:
	python3 -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt
extra:
	python -m nltk.downloader popular
demo:
	python label/matrix.py
	python label/model.py
clean:
	rm -rf .venv
