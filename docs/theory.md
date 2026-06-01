# Board Generator Theory

The canonical theory documentation is now the PDF report
[`docs/board_generator_theory.pdf`](./board_generator_theory.pdf), generated from the LaTeX source
[`docs/board_generator_theory.tex`](./board_generator_theory.tex).

This file is intentionally kept as a short redirect so older links to `docs/theory.md` do not break.

To rebuild the PDF locally:

```bash
/mnt/c/cygwin64/bin/pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp/board_generator_theory_build docs/board_generator_theory.tex
```
