.PHONY: all clean

out/main.pdf: main.tex main.bib
	mkdir -p out
	pdflatex -output-directory=out main.tex
	BIBINPUTS=out:.: bibtex out/main
	pdflatex -output-directory=out main.tex
	pdflatex -output-directory=out main.tex

clean:
	rm -rf out
