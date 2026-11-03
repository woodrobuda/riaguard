# Contributing to riaguard

Thanks for helping. riaguard is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).

## Ground rules
- **Analysis and alerts only.** No order entry, broker order APIs or automated execution. PRs that add them will be closed.
- **No secrets.** Never commit API keys or credentials. Read keys from environment variables (for example `POLYGON_API_KEY`).
- **Descriptive alert wording.** Follow [docs/ALERT_WORDING.md](docs/ALERT_WORDING.md).
- Keep PRs small, with tests. Run `pytest -q` before pushing.

## Developer Certificate of Origin (DCO)
We use the [Developer Certificate of Origin 1.1](https://developercertificate.org/) instead of a CLA.
By signing off a commit you certify that you wrote the change, or otherwise have the right to submit it
under the project's open-source license.

Sign off every commit with `-s`:

```bash
git commit -s -m "Describe the change"
```

This adds a trailer that must match the commit author:

```
Signed-off-by: Your Name <you@example.com>
```

Forgot? Fix the last commit with `git commit --amend -s --no-edit`, or a whole branch with
`git rebase --signoff main`, then `git push --force-with-lease`.

A CI check (`.github/workflows/dco.yml`) fails any pull request that contains a commit without a
`Signed-off-by` trailer matching its author's email.
