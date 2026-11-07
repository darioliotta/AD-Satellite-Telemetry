# Single pass

Retain only the original first encoder-decoder pass, including its zero context,
2F embedding, positional encoding and reconstruction head. Remove decoder 2,
the second encoder pass and self-conditioning. Retained initial weights match
the corresponding reference modules for the same seed.

There is one tensor output and a single-reconstruction MSE objective. Alpha is
not applicable. This tests the combined removal of the refinement stage and its
training objective; it must not be described as an identical-loss intervention.
Unique parameter count equals shared-decoder, but only one stage is computed.
