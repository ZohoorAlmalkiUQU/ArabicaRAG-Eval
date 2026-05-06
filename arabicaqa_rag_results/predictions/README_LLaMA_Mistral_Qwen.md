# ArabicaQA RAG Evaluation – LLaMA vs Mistral vs Qwen

Evaluation subset: **2 questions**  
Answerable: **1**  
Unanswerable: **1**  
No-answer label: **غير موجود في السياق**  

## Final QA Scores

| model   |   overall_em |   overall_em25 |   answerable_em |   answerable_em25 |   unanswerable_no_answer_count |   unanswerable_total |   unanswerable_abstention_accuracy |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|-------------:|---------------:|----------------:|------------------:|-------------------------------:|---------------------:|-----------------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| qwen    |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            125.91  |              125.91  |           162.778 |           57.9323 |
| llama   |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            138.965 |              138.965 |           180.422 |           65.1431 |
| mistral |            0 |              0 |               0 |                 0 |                              0 |                    1 |                                  0 |        0 |            366.345 |              366.345 |           477.619 |          174.851  |

**Winner by Overall EM25:** qwen (50.00%)
