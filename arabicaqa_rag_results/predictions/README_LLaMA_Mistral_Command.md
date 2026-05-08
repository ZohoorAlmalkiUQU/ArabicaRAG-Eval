# ArabicaQA RAG Evaluation – LLaMA vs Mistral vs Command

Evaluation subset: **2 questions**  
Answerable: **1**  
Unanswerable: **1**  
No-answer label: **غير موجود في السياق**  

## Final QA Scores

| model   |   overall_em |   overall_em25 |   answerable_em |   answerable_em25 |   unanswerable_no_answer_count |   unanswerable_total |   unanswerable_abstention_accuracy |   errors |   latency_mean_sec |   latency_median_sec |   latency_p95_sec |   latency_std_sec |
|:--------|-------------:|---------------:|----------------:|------------------:|-------------------------------:|---------------------:|-----------------------------------:|---------:|-------------------:|---------------------:|------------------:|------------------:|
| command |           50 |             50 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |             2.6829 |               2.6829 |            3.3171 |            0.9965 |
| llama   |            0 |              0 |               0 |                 0 |                              1 |                    1 |                                100 |        0 |             1.9605 |               1.9605 |            2.2038 |            0.3823 |
| mistral |            0 |              0 |               0 |                 0 |                              0 |                    1 |                                  0 |        0 |             5.9581 |               5.9581 |            6.4924 |            0.8395 |

**Winner by Overall EM25:** command (50.00%)
