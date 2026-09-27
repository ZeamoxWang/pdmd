#!/usr/bin/env bash
# Queue a jobs JSON for the worker: tools/submit.sh jobs/giant_cat_harbor_768p_4nfe.json
set -euo pipefail
DEPLOY=${DEPLOY:-zimo-deployment-h3-a10}
JOB=$1
NAME=$(basename "$JOB")
# Write under a hidden name first so the worker never reads a partial file
kubectl exec -i "deploy/$DEPLOY" -- bash -c \
  "cat > /pv/h3/queue/.$NAME.tmp && mv /pv/h3/queue/.$NAME.tmp /pv/h3/queue/$NAME && ls /pv/h3/queue"
