
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

## Generate API docs
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

Change the following variables in conf.py to configure themes and docstring parsers:
- ```html_theme = 'sphinx_rtd_theme'```
- ```extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc"]```

If Markdown files will be used, then install 
```pip install recommonmark```

and add the following extenstions and suffix in ```conf.py```:
```python 
extensions = ["recommonmark"]
source_suffix = {
'.rst': 'restructuredtext',
'.md': 'markdown',
}
```
## Generate html files
```bash
cd doc
# For Windows run
.\make.bat html
# For Linux run
make html
```


