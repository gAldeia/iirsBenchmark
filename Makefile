SHELL=/bin/bash

PYTHON = python

# will use num_processors - 1 to perform tests in parallel
# (requires pytest-xdist installed)
NUM = $($(shell grep -c ^processor /proc/cpuinfo) \- 1)

all: test build-dist clean

test:
	# Create detailed html file. Ignore warnings
	${PYTHON} -m pytest tests/*.py \
		-ra --html=tests/tests-report.html --self-contained-html -p no:warnings

	# run a specific test file:
	# python -m pytest tests/<test_filename>.py -n NUM

build-dist:
	if [ -d "./dist/*" ]; then \
        rm -r ./dist/*; \
    fi \
	
	${PYTHON} setup.py develop
	${PYTHON} setup.py sdist
	
	${PYTHON} -m pip install ./dist/*.tar.gz

clean:
	rm -r .pytest_cache
	rm `find ./ -name '__pycache__'` -rf

# Upload on test.py: (to upload on pypi remove '--repository testpypi')
# python -m twine upload --repository testpypi dist/*