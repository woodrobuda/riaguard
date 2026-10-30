# GitHub deployment

This document explains how to publish the project on GitHub.

## 1. Create the repository

Create a new public repository named `khoros-cantata-python`. Do not initialize
it with a README, license, or `.gitignore`; those files are already in this
project.

## 2. Check the package locally

From that folder run:

```text
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest
```

The release test suite runs without the private historical corpus.

## 3. Create the first Git commit and publish

```text
git init
git add .
git commit -m "Publish Python-native Khoros Cantata reconstruction"
git branch -M main
git remote add origin https://github.com/<your-account>/khoros-cantata-python.git
git push -u origin main
```

The repository is licensed under Apache-2.0. The recovered historical source
corpus and private validation datasets are not included.
