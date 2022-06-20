



## Overview

This is the code for the paper [“Propose and Review”: Interactive Bias Mitigation of
Machine Classifiers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4139244).

By Tianyi Li, Zhoufei Tang, Tao Lu and Xiaoquan (Michael) Zhang 



## Demo

The  [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) is used in this demo code.

Run the code with the following command in python 3 with the package listed in the :

```python
python main.py
```

the result is stored in the "Result" folder. With a threshold $\epsilon=0.005$, the attributes’ distance to origin along iterations of bias mitigation is shown in the figure below.

<img src="readme.assets\Figure5.png" alt="Figure5" style="zoom: 33%;" />

---

## Reference

```bibtex
@article{Li2022,
	author = {Tianyi Li, Zhoufei Tang, Tao Lu and Xiaoquan (Michael) Zhang},
	title = {"Propose and Review'': Interactive Bias Mitigation for Machine Classifiers},
	journal = {SSRN preprint https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4139244},
	year = {2022}
}
```

