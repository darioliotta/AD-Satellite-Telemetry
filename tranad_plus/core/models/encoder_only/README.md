# Encoder only

Retain one zero-context encoder pass. Apply the original linear-sigmoid output
head directly to the final encoder token. Remove both Transformer decoder
modules and the second encoder pass. Keep the 2F input embedding and the
existing positional encoding rather than adding a new learned bottleneck.

There is one reconstruction and a single-output MSE objective; alpha is not
applicable. This is the more aggressive exploratory intervention. Compare it
with the matched reference setting and seed from the main/compact campaigns.
