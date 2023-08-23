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
