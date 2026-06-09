.PHONY: all clean

all: out/main.pdf out/prelim.pdf

out/main.pdf: main.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out main.tex
	BIBINPUTS=out:.: bibtex out/main
	pdflatex -output-directory=out main.tex
	pdflatex -output-directory=out main.tex

out/prelim.pdf: prelim.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out prelim.tex
	BIBINPUTS=out:.: bibtex out/prelim
	pdflatex -output-directory=out prelim.tex
	pdflatex -output-directory=out prelim.tex

clean:
	rm -rf out
