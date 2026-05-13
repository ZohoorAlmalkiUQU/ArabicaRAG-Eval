# ArabicaQA RAG Evaluation – Strict and Flexible Abstention

Evaluation subset: **1,000 questions**  
Answerable: **500**  
Unanswerable: **500**  
Canonical no-answer label: **غير موجود في السياق**  

## Evaluation Policies

**Strict abstention / canonical EM:** unanswerable questions are correct only if the prediction exactly matches the canonical no-answer label after normalization.  
**Flexible abstention / abstention accuracy:** unanswerable questions are correct if the prediction matches any recognized abstention pattern.  

## Final Scores

| model   |   strict_overall_em |   strict_overall_em25 |   flexible_overall_em |   flexible_overall_em25 |   answerable_em |   answerable_em25 |   strict_abstention_count |   flexible_abstention_count |   unanswerable_total |   strict_abstention_accuracy |   flexible_abstention_accuracy |   answerable_abstention_count |   answerable_total |   answerable_abstention_rate |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|--------------------:|----------------------:|----------------------:|------------------------:|----------------:|------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------:|-------------------------------:|------------------------------:|-------------------:|-----------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| mistral |                 4.1 |                   4.1 |                  28.6 |                    28.6 |             0.6 |               0.6 |                        38 |                         283 |                  500 |                          7.6 |                           56.6 |                           321 |                500 |                         64.2 |        0 |             6.445  |               6.0193 |           10.2307 |            3.1074 |
| command |                27.5 |                  27.5 |                  27.7 |                    27.7 |             1.2 |               1.2 |                       269 |                         271 |                  500 |                         53.8 |                           54.2 |                           298 |                500 |                         59.6 |        0 |             1.5116 |               1.3402 |            2.5462 |            0.4535 |
| llama   |                 5.4 |                   5.4 |                   8   |                     8   |             3.2 |               3.2 |                        38 |                          64 |                  500 |                          7.6 |                           12.8 |                            98 |                500 |                         19.6 |        0 |             1.233  |               1.1216 |            1.8918 |            0.4692 |

**Winner by Flexible Overall EM25:** mistral (28.60%)
