# Evaluating Domain Adversarial Neural Networks on Multi-Genre Natural Language Inference

The recently published MULTINLI dataset introduces the domain adaptation challenge for the Natural Language Inference (NLI) task.
In this work, we address this challenge by using a domain adversarial approach, which has been previously applied for multiple Natural Language Processing (NLP) tasks.
We extend the setup used in prior work, to support training with multiple source domains.




Our implementation is based on the BiLSTM baseline code, which was published during the RepEval 2017 challenge - https://github.com/woollysocks/multiNLI

We also used the gradient flip operation, as implemented by @pumpikano - https://github.com/pumpikano/tf-dann
