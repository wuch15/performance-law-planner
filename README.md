# LLM Training Planner
A training plan generator based on the performance law of large language models.

## Usage
Given limited training resource budgets, it is critical to design model architectures to achieve optimal performance. By searching architectures with different configurations, our program can generate top training plans for reference.
Just run the code on Windows using the command:
```
python plan.py
```

<img src=https://github.com/wuch15/performance-law-planner/blob/main/plan.png width=500>

You can enter the key parameters and choose the preferred mode, then click the `compute` button to search for top recommended training plans. However, these plans are just based on the predictions of the performance law rather than real experiments. For LLM researchers, it is highly recommended to calibrate the predictions based on experiments on their in-house data. 

Note: We set the number of KV heads to 8 and the size of vocabulary to 150K.

## Reference
If you find this plan generator helpful for your LLM development, please feel free to cite this paper:
```
@article{wu2024performance,
  title={Performance Law of Large Language Models},
  author={Wu, Chuhan and Tang, Ruiming},
  journal={arXiv preprint arXiv:2408.09895},
  year={2024}
}
```
