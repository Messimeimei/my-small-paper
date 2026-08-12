TEXLIVE_BIN := $(abspath ../../.tools/texlive/2026/bin/x86_64-linux)
LATEXMK := $(TEXLIVE_BIN)/latexmk
BUILD_DIR := build
SOURCE := main.tex

.PHONY: pdf clean check

pdf:
	@mkdir -p $(BUILD_DIR)
	@PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -pdf \
		-interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir=$(BUILD_DIR) $(SOURCE)
	@cp -f $(BUILD_DIR)/main.pdf main.pdf

clean:
	@PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -C -outdir=$(BUILD_DIR) $(SOURCE)
	@rm -f $(BUILD_DIR)/main.bbl

check: pdf
	@test -s $(BUILD_DIR)/main.pdf
	@../../.venv-paper/bin/python ../analysis/scripts/check_pdf.py $(BUILD_DIR)/main.pdf
	@echo "Compiled and checked manuscript: $(BUILD_DIR)/main.pdf"
