#!/usr/bin/env bash
# Sets the repo description, website and topics. Requires: pkg install gh && gh auth login
set -euo pipefail
REPO="Nikhil-creat/synthetic-data-pipeline"
gh repo edit "$REPO" \
  --description "Equilibra: agentic synthetic-data pipeline for severely imbalanced datasets. SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, ResNet features, Qdrant RAG, a Groq + Gemini AI assistant, FastAPI + Docker, and honest PR-AUC evaluation." \
  --homepage "https://nikhil-creat.github.io/synthetic-data-pipeline/"
for t in machine-learning imbalanced-data smote imbalanced-learn synthetic-data pytorch resnet fastapi qdrant rag agentic-ai llm ai-assistant groq gemini docker mlops data-science precision-recall github-pages; do
  gh repo edit "$REPO" --add-topic "$t"
done
echo "Done."
