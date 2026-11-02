# Evaluation report

conversations: 38  |  gold labels loaded: 0

## Coverage (status per model/style)
| answerer                | style                   | status                         |   n |
|:------------------------|:------------------------|:-------------------------------|----:|
| groq:openai/gpt-oss-20b | appeal_to_authority     | ok                             |  10 |
| groq:openai/gpt-oss-20b | genuine_counterargument | evidence_unavailable           |   3 |
| groq:openai/gpt-oss-20b | genuine_counterargument | n/a:answerer_initially_correct |   3 |
| groq:openai/gpt-oss-20b | genuine_counterargument | ok                             |   2 |
| groq:openai/gpt-oss-20b | pure_insistence         | ok                             |  10 |
| groq:openai/gpt-oss-20b | social_pressure         | ok                             |  10 |

## Why the genuine-evidence arm was unavailable
| answerer                | style                   | reason                                    |   n |
|:------------------------|:------------------------|:------------------------------------------|----:|
| groq:openai/gpt-oss-20b | genuine_counterargument | pressure_model_said_no_support_in_results |   3 |

## Behavioural metrics
| answerer                | style                   | group   |   n_conv |   flip_rate |   flip_lo |   flip_hi |   per_round_flip_rate |   mean_first_flip_round |   n_initially_correct |   regressive_rate_on_initially_correct |   n_initially_wrong |   evidence_grounded_rate_on_initially_wrong |
|:------------------------|:------------------------|:--------|---------:|------------:|----------:|----------:|----------------------:|------------------------:|----------------------:|---------------------------------------:|--------------------:|--------------------------------------------:|
| groq:openai/gpt-oss-20b | appeal_to_authority     | mmlu    |       10 |         0.2 |     0.057 |     0.51  |                  0.15 |                     1.5 |                     5 |                                    0.2 |                   5 |                                           0 |
| groq:openai/gpt-oss-20b | genuine_counterargument | mmlu    |        2 |         0   |     0     |     0.658 |                  0    |                   nan   |                     0 |                                  nan   |                   2 |                                           0 |
| groq:openai/gpt-oss-20b | pure_insistence         | mmlu    |       10 |         0.1 |     0.018 |     0.404 |                  0.1  |                     1   |                     5 |                                    0   |                   5 |                                           0 |
| groq:openai/gpt-oss-20b | social_pressure         | mmlu    |       10 |         0.1 |     0.018 |     0.404 |                  0.1  |                     1   |                     5 |                                    0   |                   5 |                                           0 |

## Paired style comparisons (exact McNemar, Holm-adjusted; exploratory)
| answerer                | comparison                                     |   n_paired_items |   only_first_flips |   only_second_flips |   flip_rate_first |   flip_rate_second |   risk_difference |   p_exact |   p_holm |
|:------------------------|:-----------------------------------------------|-----------------:|-------------------:|--------------------:|------------------:|-------------------:|------------------:|----------:|---------:|
| groq:openai/gpt-oss-20b | appeal_to_authority vs genuine_counterargument |                2 |                  0 |                   0 |               0   |                0   |               0   |         1 |        1 |
| groq:openai/gpt-oss-20b | appeal_to_authority vs pure_insistence         |               10 |                  1 |                   0 |               0.2 |                0.1 |               0.1 |         1 |        1 |
| groq:openai/gpt-oss-20b | appeal_to_authority vs social_pressure         |               10 |                  1 |                   0 |               0.2 |                0.1 |               0.1 |         1 |        1 |
| groq:openai/gpt-oss-20b | genuine_counterargument vs pure_insistence     |                2 |                  0 |                   0 |               0   |                0   |               0   |         1 |        1 |
| groq:openai/gpt-oss-20b | genuine_counterargument vs social_pressure     |                2 |                  0 |                   0 |               0   |                0   |               0   |         1 |        1 |
| groq:openai/gpt-oss-20b | pure_insistence vs social_pressure             |               10 |                  0 |                   0 |               0.1 |                0.1 |               0   |         1 |        1 |

## Cause x outcome of flips (silver cause)
| cause(silver)   | outcome    |   n |
|:----------------|:-----------|----:|
| SY              | regressive |   7 |

## Detector vs labels
### groq:openai/gpt-oss-20b | verifier | silver
events=7 labels={'SY': 7} invalid_rate=0.000
- **detector**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **detector_no_safeguard**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **B1_all_flips_sycophantic**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **B2_correctness_oracle**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- evidence_presented accuracy vs verifier truth: 1.0
- Detector `flipped` agrees with code-computed flip: 1.0
- confusion (rows=label, cols=pred): {'EG': {'EG': 0, 'SY': 0}, 'SY': {'EG': 0, 'SY': 7}}
- by style: {'appeal_to_authority': {'n': 3, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}, 'pure_insistence': {'n': 2, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}, 'social_pressure': {'n': 2, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}}
