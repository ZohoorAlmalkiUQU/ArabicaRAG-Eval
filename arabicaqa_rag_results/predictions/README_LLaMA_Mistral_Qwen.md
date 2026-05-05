# ArabicaQA RAG Evaluation – LLaMA vs Mistral vs Qwen

Evaluation subset: **2 questions**  
Answerable: **1**  
Unanswerable: **1**  
No-answer label: **غير موجود في السياق**  

## Final QA Scores

| model   |   overall_em |   overall_em25 |   answerable_em |   answerable_em25 |   unanswerable_no_answer_count |   unanswerable_total |   unanswerable_abstention_accuracy |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|-------------:|---------------:|----------------:|------------------:|-------------------------------:|---------------------:|-----------------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| llama   |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            180.042 |              180.042 |           189.769 |           15.2848 |
| qwen    |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |            197.064 |              197.064 |           215.082 |           28.313  |
| mistral |            0 |              0 |               0 |                 0 |                              0 |                    1 |                                  0 |        0 |            484.229 |              484.229 |           491.111 |           10.813  |

**Winner by Overall EM25:** llama (50.00%)
