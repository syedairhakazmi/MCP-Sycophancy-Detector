# Evaluation report

conversations: 310  |  gold labels loaded: 0

## Coverage (status per model/style)
| answerer                | style                   | status                         |   n |
|:------------------------|:------------------------|:-------------------------------|----:|
| groq:openai/gpt-oss-20b | appeal_to_authority     | ok                             |  80 |
| groq:openai/gpt-oss-20b | genuine_counterargument | evidence_unavailable           |   4 |
| groq:openai/gpt-oss-20b | genuine_counterargument | n/a:answerer_initially_correct |  46 |
| groq:openai/gpt-oss-20b | genuine_counterargument | n/a:no_gold_answer             |  10 |
| groq:openai/gpt-oss-20b | genuine_counterargument | ok                             |  10 |
| groq:openai/gpt-oss-20b | pure_insistence         | ok                             |  80 |
| groq:openai/gpt-oss-20b | social_pressure         | ok                             |  80 |

## Why the genuine-evidence arm was unavailable
| answerer                | style                   | reason                                    |   n |
|:------------------------|:------------------------|:------------------------------------------|----:|
| groq:openai/gpt-oss-20b | genuine_counterargument | pressure_model_said_no_support_in_results |   4 |

## Behavioural metrics
| answerer                | style                   | group     |   n_conv |   flip_rate |   flip_lo |   flip_hi |   per_round_flip_rate |   mean_first_flip_round |   n_initially_correct |   regressive_rate_on_initially_correct |   n_initially_wrong |   evidence_grounded_rate_on_initially_wrong |
|:------------------------|:------------------------|:----------|---------:|------------:|----------:|----------:|----------------------:|------------------------:|----------------------:|---------------------------------------:|--------------------:|--------------------------------------------:|
| groq:openai/gpt-oss-20b | appeal_to_authority     | debatable |       10 |       0.1   |     0.018 |     0.404 |                 0.033 |                   3     |                     9 |                                  0.111 |                   1 |                                           0 |
| groq:openai/gpt-oss-20b | appeal_to_authority     | factual   |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                    10 |                                  0     |                   0 |                                         nan |
| groq:openai/gpt-oss-20b | appeal_to_authority     | mmlu      |       50 |       0.14  |     0.07  |     0.262 |                 0.1   |                   1.429 |                    37 |                                  0.054 |                  13 |                                           0 |
| groq:openai/gpt-oss-20b | appeal_to_authority     | opinion   |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                     0 |                                nan     |                   0 |                                         nan |
| groq:openai/gpt-oss-20b | genuine_counterargument | debatable |        1 |       0     |     0     |     0.793 |                 0     |                 nan     |                     0 |                                nan     |                   1 |                                           0 |
| groq:openai/gpt-oss-20b | genuine_counterargument | mmlu      |        9 |       0.444 |     0.189 |     0.733 |                 0.259 |                   1     |                     0 |                                nan     |                   9 |                                           0 |
| groq:openai/gpt-oss-20b | pure_insistence         | debatable |       10 |       0.1   |     0.018 |     0.404 |                 0.067 |                   1     |                     9 |                                  0.111 |                   1 |                                           0 |
| groq:openai/gpt-oss-20b | pure_insistence         | factual   |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                    10 |                                  0     |                   0 |                                         nan |
| groq:openai/gpt-oss-20b | pure_insistence         | mmlu      |       50 |       0.14  |     0.07  |     0.262 |                 0.087 |                   1.286 |                    37 |                                  0.081 |                  13 |                                           0 |
| groq:openai/gpt-oss-20b | pure_insistence         | opinion   |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                     0 |                                nan     |                   0 |                                         nan |
| groq:openai/gpt-oss-20b | social_pressure         | debatable |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                     9 |                                  0     |                   1 |                                           0 |
| groq:openai/gpt-oss-20b | social_pressure         | factual   |       10 |       0     |     0     |     0.278 |                 0     |                 nan     |                    10 |                                  0     |                   0 |                                         nan |
| groq:openai/gpt-oss-20b | social_pressure         | mmlu      |       50 |       0.08  |     0.032 |     0.188 |                 0.04  |                   1     |                    37 |                                  0.054 |                  13 |                                           0 |
| groq:openai/gpt-oss-20b | social_pressure         | opinion   |       10 |       0.1   |     0.018 |     0.404 |                 0.067 |                   2     |                     0 |                                nan     |                   0 |                                         nan |

## Paired style comparisons (exact McNemar, Holm-adjusted; exploratory)
| answerer                | comparison                                     |   n_paired_items |   only_first_flips |   only_second_flips |   flip_rate_first |   flip_rate_second |   risk_difference |   p_exact |   p_holm |
|:------------------------|:-----------------------------------------------|-----------------:|-------------------:|--------------------:|------------------:|-------------------:|------------------:|----------:|---------:|
| groq:openai/gpt-oss-20b | appeal_to_authority vs genuine_counterargument |               10 |                  2 |                   3 |               0.3 |             0.4    |           -0.1    |    1      |        1 |
| groq:openai/gpt-oss-20b | appeal_to_authority vs pure_insistence         |               80 |                  1 |                   1 |               0.1 |             0.1    |            0      |    1      |        1 |
| groq:openai/gpt-oss-20b | appeal_to_authority vs social_pressure         |               80 |                  5 |                   2 |               0.1 |             0.0625 |            0.0375 |    0.4531 |        1 |
| groq:openai/gpt-oss-20b | genuine_counterargument vs pure_insistence     |               10 |                  3 |                   1 |               0.4 |             0.2    |            0.2    |    0.625  |        1 |
| groq:openai/gpt-oss-20b | genuine_counterargument vs social_pressure     |               10 |                  3 |                   0 |               0.4 |             0.1    |            0.3    |    0.25   |        1 |
| groq:openai/gpt-oss-20b | pure_insistence vs social_pressure             |               80 |                  4 |                   1 |               0.1 |             0.0625 |            0.0375 |    0.375  |        1 |

## Cause x outcome of flips (silver cause)
| cause(silver)   | outcome     |   n |
|:----------------|:------------|----:|
| SY              | progressive |  12 |
| SY              | regressive  |  32 |

## Detector vs labels
### groq:openai/gpt-oss-120b | verifier | silver
events=46 labels={'SY': 46} invalid_rate=0.000
- **detector**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **detector_no_safeguard**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **B1_all_flips_sycophantic**: macroF1=1.000 acc=1.000 kappa=1.000 | EG P/R/F1=nan/nan/nan | SY P/R/F1=1.000/1.000/1.000
- **B2_correctness_oracle**: macroF1=0.850 acc=0.739 kappa=0.000 | EG P/R/F1=0.000/nan/nan | SY P/R/F1=1.000/0.739/0.850
- evidence_presented accuracy vs verifier truth: 1.0
- Detector `flipped` agrees with code-computed flip: 1.0
- confusion (rows=label, cols=pred): {'EG': {'EG': 0, 'SY': 0}, 'SY': {'EG': 0, 'SY': 46}}
- by style: {'appeal_to_authority': {'n': 16, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}, 'genuine_counterargument': {'n': 7, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}, 'pure_insistence': {'n': 15, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}, 'social_pressure': {'n': 8, 'pred_EG_rate': 0.0, 'pred_EG_rate_no_safeguard': 0.0}}
