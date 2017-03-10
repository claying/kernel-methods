PYTHON ?= python

# compilation
in: inplace

inplace:
	$(PYTHON) setup.py build_ext -i


