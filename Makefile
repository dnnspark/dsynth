error:
	@echo "Please choose one of the following target: venv, install_core, install_test, install_tools, dev, ci, flake8, clean"
	@exit 2

venv:
	virtualenv venv -p `which python3.6` 
	ln -s venv/bin/activate activate

install_core:
	source ./venv/bin/activate; \
	pip install -e .

install_test:
	source ./venv/bin/activate; \
	pip install pytest flake8

install_tools:
	source ./venv/bin/activate; \
	pip install seaborn scikit-image imageio

dev: venv install_core install_test install_tools

ci:
	pytest

flake8:
	flake8 --ignore=E501,F401,E128,E402,E731,F821 pybullet
	flake8 --ignore=E501,F401,E128,E402,E731,F821 tests


clean:
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
	rm -rf junit-py*.xml
	rm -rf venv/
	rm -rf `find . -name '*.pyc'`
	rm -rf `find . -name __pycache__`
	rm -rf activate

.PHONY: venv install install_test ci clean_install clean
