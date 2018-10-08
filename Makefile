venv:
	virtualenv venv -p `which python3.6` 

install:
	source ./venv/bin/activate; \
	pip install -e .; \

install_test:
	pip install -e . && pip install pytest && pip install flake8

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
	rm -rf venv/

.PHONY: venv install install_test ci clean_install clean
