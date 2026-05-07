# In-Context Multiple Instance Learning (ICMIL)

Official code for the paper **"In-Context Multiple Instance Learning"**.

ICMIL is a Prior-data Fitted Network (PFN) for Multiple Instance Learning. A Perceiver-style transformer is pretrained on synthetic bag-structured data and, at inference time, classifies new MIL tasks **in a single forward pass** — no gradient updates, no hyperparameter tuning, no task-specific finetuning.

## Highlights

- **In-context MIL**: solve a new bag classification task by feeding labeled context bags directly to the model.
- **Perceiver-style architecture** for hierarchical set inputs that handles scalability, task-dependent instance compression, and within-bag permutation invariance.
- **Synthetic priors** for bag-structured data (factorized and joint), with a mixture prior that combines their complementary inductive biases.

## License

MIT — see [LICENSE](LICENSE).
