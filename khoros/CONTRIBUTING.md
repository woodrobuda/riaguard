# Contributing

## Basic rule

Make the code easier to understand, test, and maintain. Prefer small changes that can be checked with a test.

## Before changing behavior

1. Read the relevant programmer-guide section.
2. Find the historical source or recovered evidence that defines the behavior.
3. Add or update a test.
4. Run the complete test suite.
5. Update the documentation when the public behavior changes.

## Do not do this

Do not replace a historical behavior with a modern interpretation without documenting the difference. Do not use Python `eval()` to replace the recovered Khoros expression language. Do not add a hidden dependency on an original Khoros executable.

## Documentation level

Write explanations so that a technically interested high-school student can follow them. Explain unusual terms the first time they appear. Prefer examples over unexplained jargon.
