rm -rf _srcdocs
rm -rf _build

find . -name "*.ipynb" -exec reset_notebook {} \;

python build_source_docs.py;
jupyter-book build -W --keep-going .