Description
----
This directory can contain Jupyter notebooks for interactive manipulation of data and figures. To directly interface with `swiftzoom` and the data science pipeline, 

```python
%matplotlib inline
```

```python
import boilerplate
from loader import GroupZoom

group_object = GroupZoom('your/simulation/directory', redshift=your_redshift)
```

The boilerplate file
----
This directory is external to the package and should host your custom notebooks. You can use `boilerplate.py` to bind your notebooks to the SWIFTzoom package by specifying functions and static paths that automate access to the SWIFTzoom features. If you displace the notebooks directory, you can refractor the boilerplate to maintain access to the packages.
