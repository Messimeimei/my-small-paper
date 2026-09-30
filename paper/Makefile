TEXLIVE_BIN := $(abspath ../.tools/texlive/2026/bin/x86_64-linux)
LATEXMK := $(TEXLIVE_BIN)/latexmk
FIGURE_DIR := manuscript/figures

.PHONY: paper paper-clean paper-check figure-training-targets \
	figure-rationale-errors

paper:
	$(MAKE) -C manuscript pdf

paper-clean:
	$(MAKE) -C manuscript clean

paper-check:
	$(MAKE) -C manuscript check

figure-training-targets:
	@PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -cd -g -pdf \
		-interaction=nonstopmode -halt-on-error -file-line-error \
		$(FIGURE_DIR)/training_targets_by_task_type.tex

figure-rationale-errors:
	@PATH=$(TEXLIVE_BIN):$$PATH $(LATEXMK) -cd -g -pdf \
		-interaction=nonstopmode -halt-on-error -file-line-error \
		$(FIGURE_DIR)/rationale_error_bars.tex
