init:
	python3 -m venv .venv
	. .venv/bin/activate; \
	pip install -r requirements.txt;
update:
	. .venv/bin/activate; \
	pip install -r requirements.txt --upgrade;
extra:
	python -m nltk.downloader popular
demo:
	. .venv/bin/activate; \
	python label/matrix.py; \
	python label/model.py
clean:
	rm -rf .venv
