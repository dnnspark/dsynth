install:
	pip install -e .

ci:
	pytest

clean_install: install clean

flake8:
	flake8 --ignore=E501,F401,E128,E402,E731,F821 .

clean:
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
	rm -rf junit-py*.xml

.PHONY: install ci clean_install test clean
