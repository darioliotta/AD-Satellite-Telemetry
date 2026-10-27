# Run `latexmk -pdf` from project root; `latexmk -c` removes auxiliaries.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -halt-on-error %O %S';
$bibtex_use = 2;
$quiet = 1;
