# How to Contribute Docs

Doc pages can be added as `.ipynb` or `.md` files within the `aviary/docs` folder.
We're using [jupyter-book](https://jupyterbook.org/) for the docs, which is a well-documented and full-featured platform.
To build the docs, you'll need to install jupyter-book following [these instructions](https://jupyterbook.org/en/stable/start/overview.html).
Jupyter-book allows for arbitrary Jupyter notebook usage to intersperse code and documentation and it uses [MyST markdown](https://jupyterbook.org/en/stable/content/myst.html).

To modify the docs, simply add a file to the repo within the docs folder.
You can then add it to the `docs/_toc.yml` file following the structure for the skeletal outline.

You can then run the `build_book.sh` bash script using the `sh build_book.sh` command to build the docs.
Currently, they are not hosted publicly online.
To view the docs you must build them locally.
The built docs live at `..Aviary/aviary/docs/_build/html/intro.html`.
Navigate to this file in your file manager once you have built the docs, and you can open it from there to your favorite internet browser.

One of the powerful features that we use to write docs is automatic formatting of docstrings into documentation.
[This file](../theory_guide/merging_syntax.ipynb) contains an example of turning docstrings into documentation.
It is important to note when writing docstrings that the docstrings must be in numpy format.
[Here](https://numpydoc.readthedocs.io/en/latest/format.html) is a link to instructions on how to write numpy format docstrings.

## Adding doc pages without putting them in the Table of Contents

Sometimes when building the docs, you may see a warning like this:

>```checking consistency... /home/user/Work/OpenMDAO/plugins/aviary/aviary/docs/user_guide/external_aero.ipynb: WARNING: document isn't included in any toctree```

This is because the notebook is not in a table of contents.
It isn't always ideal to put everything in the toc, so you can choose to omit a notebook by adding the `"orphan": true` to the metadata at the bottom of the file.

```json
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  },
  "orphan": true
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

Note, this will require manual editing of the ipynb file.
Don't forget to add the comma at the end of the previous brace.

## Doc formatting and style

````{margin}
```{note}
[The active voice](https://www.grammarly.com/blog/active-vs-passive-voice/) is when the subject of the sentence is doing the action.
For example, "The dog ate the bone" is active, whereas "The bone was eaten by the dog" is passive.
```
````

When writing docs, please

- use simple, clear, and concise language
- write each sentence on a new line (this helps make diffs more clear)
- use the active voice
- consider the audience of the particular section you're writing
