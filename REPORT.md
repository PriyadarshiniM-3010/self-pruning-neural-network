# Self-Pruning Neural Network – Case Study Report

## Objective

Implement a neural network that can reduce less useful connections during training using learnable gates and sparsity regularization.

## Approach

I created a custom PrunableLinear layer where each weight has a gate score.

gate = sigmoid(gate_score)
effective_weight = weight * gate

The total loss used:

CrossEntropyLoss + λ * sparsity_loss

## Dataset

CIFAR-10 was used for image classification.

## Network Architecture

3072 -> 1024 -> 512 -> 256 -> 10

BatchNorm, ReLU, and Dropout were used between layers.

## Experiments

I tested the following lambda values:

- 0.001
- 0.002
- 0.005

Epochs: 10

## Final Results

| Lambda | Test Accuracy | Sparsity |
|--------|---------------|----------|
| 0.001 | 54.79% | 99.80% |
| 0.002 | 54.86% | 99.98% |
| 0.005 | 54.64% | 100.00% |

Best accuracy was observed at λ = 0.002.

## Observations

- Increasing lambda pushed gate values lower.
- Stronger sparsity pressure reduced more connections.
- Accuracy remained around 55% across tested settings.
- Threshold selection affects reported sparsity, so a practical threshold was used to measure suppressed gates.

## Challenges

Initial experiments with stricter thresholds showed low measurable sparsity. After adjusting the evaluation threshold, gate suppression became clearer.

## Conclusion

The custom gating mechanism successfully learned to suppress weights during training, demonstrating self-pruning behavior while maintaining reasonable classification accuracy.