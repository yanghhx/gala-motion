GALA ICASSP submission package
==============================

Compiler: pdfLaTeX
Main file: main.tex
Format: US Letter, two columns, five pages; page 5 contains references only.

All reported GALA generation results use the completed official Guo/MDM
20-replication evaluations. HumanML3D has 4544 test clips; tokenizer
reconstruction uses 1504 validation clips. Runtime was measured on one RTX
4060 with bf16, batch size 1, and T=196.

Primary result artifacts in the accompanying code repository:

  outputs/humanml3d/gala_distinct_test_n20_cfg2.5.json
  outputs/humanml3d/gala_distinct_test_n50_cfg2.0.json
  outputs/kitml/gala_test_n20_cfg2.5.json
  outputs/ablation/base_rf_test_n20.json
  outputs/ablation/graph_rf_test_n20.json
  outputs/ablation/graph_global_test_n20.json
  outputs/ablation/graph_global_part_distinct_test_n20.json
  outputs/attention/part_attention_distinct.json

The part-query collapse audit reduced mean off-diagonal attention-profile
cosine similarity from 0.9988 to 0.3474 on three fixed diagnostic prompts.
The paper treats this only as evidence of non-collapse, not as proof that raw
attention weights are lexical explanations.

To regenerate figures from this directory:

  python plot_architecture.py
  python plot_figures.py

To compile:

  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
