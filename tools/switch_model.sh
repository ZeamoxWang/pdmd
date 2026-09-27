#!/usr/bin/env bash
# Choose the served checkpoint: tools/switch_model.sh 4nfe|2nfe
# If the GPU worker is running, it is restarted, which reloads the model (~25-30 min).
# If it is not running yet, only /pv/h3/serve_args is written (through the CPU pod), so the
# worker loads the chosen checkpoint when it starts.
set -euo pipefail
GPU_DEPLOY=${GPU_DEPLOY:-zimo-deployment-h3-a10}
CPU_DEPLOY=${CPU_DEPLOY:-zimo-deployment-h3-cpu}
case "${1:-}" in
  4nfe) ARGS="--transformer-path /pv/h3/ckpt/pdmd_4NFE_full --inference-steps 4" ;;
  2nfe) ARGS="--transformer-path /pv/h3/ckpt/pdmd_2NFE_fused --inference-steps 2" ;;
  *) echo "usage: $0 4nfe|2nfe" >&2; exit 1 ;;
esac
if ! kubectl get deploy "$GPU_DEPLOY" >/dev/null 2>&1; then
  kubectl exec "deploy/$CPU_DEPLOY" -- bash -c "echo '$ARGS' > /pv/h3/serve_args && cat /pv/h3/serve_args"
  echo "GPU worker not running; it will serve this checkpoint when started"
  exit 0
fi
kubectl exec "deploy/$GPU_DEPLOY" -- bash -c "echo '$ARGS' > /pv/h3/serve_args && cat /pv/h3/serve_args"
# The pattern must not match this exec's own command line, hence the anchored match
kubectl exec "deploy/$GPU_DEPLOY" -- bash -c 'pkill -f "^python /opt/h3/run_a10.py" || true'
echo "worker restarting; follow with: kubectl logs -f deploy/$GPU_DEPLOY"
