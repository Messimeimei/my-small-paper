PROJECT_ROOT := $(abspath ..)
POPPLER_BIN := $(PROJECT_ROOT)/.tools/poppler/usr/bin
POPPLER_LIB := $(PROJECT_ROOT)/.tools/poppler/usr/lib/x86_64-linux-gnu

.PHONY: paper paper-clean paper-check paper-env-check

paper:
	$(MAKE) -C manuscript pdf

paper-clean:
	$(MAKE) -C manuscript clean

paper-check:
	$(MAKE) -C manuscript check

paper-env-check:
	@test -x $(PROJECT_ROOT)/.tools/texlive/2026/bin/x86_64-linux/latexmk
	@test -x $(PROJECT_ROOT)/.tools/texlive/2026/bin/x86_64-linux/pdflatex
	@test -x $(POPPLER_BIN)/pdfinfo
	@test -x $(POPPLER_BIN)/pdffonts
	@LD_LIBRARY_PATH=$(POPPLER_LIB) $(POPPLER_BIN)/pdfinfo -v >/dev/null 2>&1
	@LD_LIBRARY_PATH=$(POPPLER_LIB) $(POPPLER_BIN)/pdffonts -v >/dev/null 2>&1
	@MPLCONFIGDIR=$(PROJECT_ROOT)/.cache/matplotlib $(PROJECT_ROOT)/.venv-paper/bin/python -c "import matplotlib, numpy, pandas, pypdf, scipy, seaborn, statsmodels"
	@echo "Paper toolchain is available."
