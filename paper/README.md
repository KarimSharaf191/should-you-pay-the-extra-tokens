# Paper

`main.tex` is the paper source, written against the ACL template in review mode.

To compile, add next to it:

- `acl.sty` and `acl_natbib.bst` from [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)
- `custom.bib`, the bibliography (not yet in the repository)

Then run `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
