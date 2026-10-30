# Developer Checklist

Before adding or changing an operator:

1. Find the historical C, Fortran, pane, CMS, and test material.
2. Write down what the old code actually does.
3. Keep file/data meaning separate from graph meaning.
4. Do not use `eval` on old command strings.
5. Do not pass old command strings to a shell.
6. Add a small test before calling the behavior complete.
7. Compare with historical results when they exist.
8. Record whether a historical executable oracle was available.
9. Keep the code comments simple enough for a new programmer to follow.
10. Update the operator closure matrix when the evidence level changes.
