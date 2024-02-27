
## Installation
```bash
pip install sphinx
pip install sphinx-rtd-theme
```

## Config project
```bash
cd doc
sphinx-quickstart
cd ..
```

## Generate docs
```bash
# sphinx-apidoc -o [DOC PATH] [CODE SRC PATH]
sphinx-apidoc -o doc .
```

## Config file
Add the root path of source code to the system paths at the beginning of conf.py, otherwise sphinx will not find them:
```bash
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
```
Change the following variables in conf.py:
- ```html_theme = 'sphinx_rtd_theme'```
- ```extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]```

## Generate html files
```bash
cd doc
.make.bat html
```


