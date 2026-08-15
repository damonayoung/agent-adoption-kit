# Makefile — agent-adoption-kit.
#
# Thin wrappers around the `aak` CLI so the paper's stated commands (`make results`,
# `make figures`) work for a reader who clones THIS repo. Generation lives entirely in
# the CLI; these targets only call it. Requires `pip install -e .` (or override AAK=,
# e.g. `make results AAK=.venv/bin/aak`).

AAK  ?= aak
SEED := 42
OUT  := out
PATHOLOGIES := healthy shallow_plateau awareness_gap ability_gap reinforcement_decay champion_dependency

.DEFAULT_GOAL := help
.PHONY: help results figures test

help:
	@echo "agent-adoption-kit — targets:"
	@echo "  make results  Simulate the six seed-$(SEED) pathologies and print each cohort's"
	@echo "                diagnosis — stage distribution, score, stall point, flags (the data behind Table 1)."
	@echo "  make figures  Export the paper-theme healthy-vs-shallow-plateau comparison as $(OUT)/comparison.pdf (Figure 2)."
	@echo "  make test     Run the pytest suite."

# results: reproduce the per-pathology results behind Table 1. Fresh dbs every run
# (aak simulate appends to an existing file), so the classifications stay deterministic.
results:
	@mkdir -p $(OUT)
	@for p in $(PATHOLOGIES); do \
		rm -f $(OUT)/$$p.db; \
		$(AAK) simulate --pathology $$p --seed $(SEED) --out $(OUT)/$$p.db; \
	done
	@for p in $(PATHOLOGIES); do \
		echo "== $$p =="; \
		$(AAK) analyze $(OUT)/$$p.db --cohort cohort-1; \
	done

# figures: emit Figure 2 — the paper-theme comparison of the headline pair. Deterministic;
# no API key, no network.
figures:
	@mkdir -p $(OUT)
	rm -f $(OUT)/healthy.db $(OUT)/shallow_plateau.db
	$(AAK) simulate --pathology healthy         --seed $(SEED) --out $(OUT)/healthy.db
	$(AAK) simulate --pathology shallow_plateau  --seed $(SEED) --out $(OUT)/shallow_plateau.db
	$(AAK) report --compare $(OUT)/healthy.db $(OUT)/shallow_plateau.db \
		--ref-cohort cohort-1 --observed-cohort cohort-1 \
		--theme paper --out $(OUT)/comparison.html --export-figure $(OUT)/comparison.pdf
	@echo "Figure 2 -> $(OUT)/comparison.pdf"

test:
	pytest
