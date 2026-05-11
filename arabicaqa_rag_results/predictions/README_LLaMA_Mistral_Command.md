# ArabicaQA RAG Evaluation – LLaMA vs Mistral vs Command

Evaluation subset: **1,000 questions**  
Answerable: **500**  
Unanswerable: **500**  
No-answer label: **غير موجود في السياق**  

## Final QA Scores

| model   |   overall_em |   overall_em25 |   answerable_em |   answerable_em25 |   unanswerable_no_answer_count |   unanswerable_total |   unanswerable_abstention_accuracy |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|-------------:|---------------:|----------------:|------------------:|-------------------------------:|---------------------:|-----------------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| command |         27.5 |           27.5 |             1.2 |               1.2 |                            271 |                  500 |                               54.2 |        0 |             1.5116 |               1.3402 |            2.5462 |            0.4535 |
| llama   |          5.4 |            5.4 |             3.2 |               3.2 |                             61 |                  500 |                               12.2 |        0 |             1.233  |               1.1216 |            1.8918 |            0.4692 |
| mistral |          4.1 |            4.1 |             0.6 |               0.6 |                            271 |                  500 |                               54.2 |        0 |             6.445  |               6.0193 |           10.2307 |            3.1074 |

**Winner by Overall EM25:** command (27.50%)
