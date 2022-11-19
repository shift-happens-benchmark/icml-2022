install:
	python3 -m pip install --upgrade twine

build:
	python3 -m pip install --upgrade build
	python3 -m build

upload_test:
	python3 -m twine upload --repository testpypi dist/*

upload:
	python3 -m twine upload dist/*

test:
	python -m pytest -vvv

mypy:
	mypy --install-types --non-interactive shifthappens/
	mypy -p shifthappens

format:
	black shifthappens/
	isort shifthappens/