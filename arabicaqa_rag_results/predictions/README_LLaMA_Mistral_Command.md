# ArabicaQA RAG Evaluation – LLaMA vs Mistral vs Command

Evaluation subset: **2 questions**  
Answerable: **1**  
Unanswerable: **1**  
No-answer label: **غير موجود في السياق**  

## Final QA Scores

| model   |   overall_em |   overall_em25 |   answerable_em |   answerable_em25 |   unanswerable_no_answer_count |   unanswerable_total |   unanswerable_abstention_accuracy |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|-------------:|---------------:|----------------:|------------------:|-------------------------------:|---------------------:|-----------------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| command |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            110.427 |              110.427 |           139.411 |           45.5438 |
| llama   |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            130.12  |              130.12  |           164.385 |           53.842  |
| mistral |            0 |              0 |               0 |                 0 |                              0 |                    1 |                                  0 |        0 |            319.17  |              319.17  |           426.486 |          168.632  |

**Winner by Overall EM25:** command (50.00%)
