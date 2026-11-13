# Shared decoder

After reference initialization, decoder 2 becomes the same module object as
decoder 1. Both forward stages and the anomaly-score context are retained.
Gradients from both stages update the same parameters. The original encoder
and output head were already shared.

The loss and 21-alpha scoring procedure remain those of the reference. Unique
parameter count drops; analytical forward MACs stay equal to the reference.
The state dictionary contains alias keys, so resource counts must use unique
model parameters rather than summing aliased state-dictionary entries.
