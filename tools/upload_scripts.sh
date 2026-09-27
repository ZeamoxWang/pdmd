#!/usr/bin/env bash
# Publish worker/ as the h3-scripts ConfigMap, mounted at /opt/h3 by the setup Job and the GPU worker.
# A running worker keeps its old code until it restarts (tools/switch_model.sh restarts it).
set -euo pipefail
cd "$(dirname "$0")/.."
kubectl create configmap h3-scripts --from-file=worker/ --dry-run=client -o yaml | kubectl apply -f -
