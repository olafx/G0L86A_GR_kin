.PHONY: all clean

all: prelim main main_2

prelim: out/prelim.pdf
out/prelim.pdf: prelim.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out prelim.tex
	BIBINPUTS=out:.: bibtex out/prelim
	pdflatex -output-directory=out prelim.tex
	pdflatex -output-directory=out prelim.tex

main: out/main.pdf
out/main.pdf: main.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out main.tex
	BIBINPUTS=out:.: bibtex out/main
	pdflatex -output-directory=out main.tex
	pdflatex -output-directory=out main.tex

main_2: out/main_2.pdf
out/main_2.pdf: main_2.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out main_2.tex
	BIBINPUTS=out:.: bibtex out/main_2
	pdflatex -output-directory=out main_2.tex
	pdflatex -output-directory=out main_2.tex

clean:
	rm -rf out
