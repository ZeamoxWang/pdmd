#!/usr/bin/env bash
# Download a video from /pv/h3/outputs: tools/fetch.sh <file.mp4> [dest_dir]
# kubectl streams tend to drop after ~1.5MB, so the file is copied in 512KB chunks, each
# verified by md5 and retried, then the whole file is verified again.
set -euo pipefail
DEPLOY=${DEPLOY:-zimo-deployment-h3-cpu}
NAME=$1
DEST=${2:-.}
REMOTE=/tmp/fetch_$$
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; kubectl exec "deploy/$DEPLOY" -- rm -rf "$REMOTE" >/dev/null 2>&1 || true' EXIT

md5() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d" " -f1; else md5 -q "$1"; fi; }

kubectl exec "deploy/$DEPLOY" -- bash -c \
  "mkdir -p $REMOTE && split -b 512K -d -a 4 /pv/h3/outputs/$NAME $REMOTE/p && cd $REMOTE && md5sum p* > sums && md5sum /pv/h3/outputs/$NAME | cut -d' ' -f1 > total"
kubectl exec "deploy/$DEPLOY" -- cat "$REMOTE/sums" > "$TMP/sums"
while read -r want part; do
  for try in 1 2 3 4 5 6; do
    kubectl exec "deploy/$DEPLOY" -- cat "$REMOTE/$part" > "$TMP/$part" 2>/dev/null || true
    [ "$(md5 "$TMP/$part")" = "$want" ] && break
    [ "$try" = 6 ] && { echo "failed to copy chunk $part" >&2; exit 1; }
  done
done < "$TMP/sums"
mkdir -p "$DEST"
awk -v d="$TMP" '{print d"/"$2}' "$TMP/sums" | xargs cat > "$DEST/$NAME"
want=$(kubectl exec "deploy/$DEPLOY" -- cat "$REMOTE/total")
[ "$(md5 "$DEST/$NAME")" = "$want" ] || { echo "md5 mismatch for $NAME" >&2; exit 1; }
echo "saved $DEST/$NAME (md5 $want)"
