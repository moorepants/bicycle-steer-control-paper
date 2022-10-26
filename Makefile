pdf: paper.pdf
paper.html: paper.tex figures/lqr-eig.png figures/pd-eigenvalues.png figures/pole-place-v06.png figures/uncontrolled-eigenvalues-with-rider.png
	pandoc --mathjax --standalone -o paper.html paper.tex
paper.pdf: paper.tex figures/lqr-eig.png figures/pd-eigenvalues.png figures/pole-place-v06.png figures/uncontrolled-eigenvalues-with-rider.png
	pdflatex paper.tex
figures/lqr-eig.png: src/lqr.py
	python src/lqr.py
figures/pd-eigenvalues.png: src/pd_control.py
	python src/pd_control.py
figures/pole-place-v06.png: src/pole_placement.py
	python src/pole_placement.py
figures/uncontrolled-eigenvalues-with-rider.png: src/uncontrolled.py
	python src/uncontrolled.py
clearpdf:
	rm paper.pdf
clean:
	(rm -rf *.ps *.log *.dvi *.aux *.*% *.lof *.lop *.lot *.toc *.idx *.ilg *.ind *.bbl *.blg *.cpt *.out)
	rm figures/*.png
