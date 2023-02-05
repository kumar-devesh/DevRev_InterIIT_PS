wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NIYPVTI-gFmfkgZGxVYh_W8AAFQtNtBD' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=1NIYPVTI-gFmfkgZGxVYh_W8AAFQtNtBD" \
-O ./data/synthetic/generated_T5small_squadv2.json && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OqzV5ANnDvLchX59i4LbxyClwKQHxArf' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=1OqzV5ANnDvLchX59i4LbxyClwKQHxArf" \
-O ./data/synthetic/generated_T5base_squadv2_.json && rm -rf /tmp/cookies.txt
