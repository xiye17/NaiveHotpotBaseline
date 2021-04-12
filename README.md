# NaiveHotpotBaseline
A very naive bert/roberta baseline for Hotpot QA (ditractor settings).
The model follows select-and-anser style. First, we use a Roberta classifier to select 2 gold pragraphs. Next, we use a Roberta QA model to answer the question.
It serves as the base Hotpot model for [Evaluating Explanations for Reading Comprehension with Realistic Counterfactuals](https://github.com/xiye17/EvalQAExpl).
