n=1
for f in *.jpg *.jpeg *.png *.webp; do
    [ -e "$f" ] || continue
    ext="${f##*.}"
    mv -- "$f" "$n.$ext"
    n=$((n+1))
done
