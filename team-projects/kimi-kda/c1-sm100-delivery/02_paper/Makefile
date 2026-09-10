TECTONIC ?= tectonic

.PHONY: all rebuild

all: main.pdf

main.pdf: main.tex references.bib
	$(TECTONIC) main.tex

rebuild:
	$(TECTONIC) main.tex
