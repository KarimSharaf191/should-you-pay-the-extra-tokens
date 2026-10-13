# Paper

`main.tex` is the paper source (ACL template, review mode) and `custom.bib` its
bibliography. Figures are drawn in the source with PGFPlots/TikZ.

To compile, add `acl.sty` and `acl_natbib.bst` from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), then run

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
