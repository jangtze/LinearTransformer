# %%
# get header to server

! for file in \
'linear_transformer.py'\
; do\
    echo "downloading ${file} ... ";\
  curl \
  -o "${file}"\
  -L "https://raw.githubusercontent.com/jangtze/LinearTransformer/refs/heads/padding/${file}";\
done
#   --create-dirs\

# %%
import linear_transformer
# %%
