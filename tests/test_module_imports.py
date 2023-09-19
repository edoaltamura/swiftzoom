"""
In this test script:

+ We create a list of module names (``modules_to_test``) that you want to check for import.
+ We use ``@pytest.mark.parametrize`` to parametrize the test with the module names from the list.
+ In the test function ``test_import_modules``, we try to import each module using ``importlib.import_module``. If the
    import fails, the test fails with a custom message.
+ Finally, we use ``pytest.main()`` to run the tests.

To run the tests, execute the script (``test_module_imports.py``):

.. code-block:: bash

    pytest test_module_imports.py
"""

import pytest
import os
import sys

# Add the path to the parent directory (one directory up) to sys.path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)

import swiftzoom

# List of modules to test
modules_to_test = [
    "loading",
    "loading.register",
    "loading.register.GroupZoom",
    "loading.dict2hdf",
    "loading.dict2hdf.Dict2HDF",
    "loading.dict2hdf.CustomPickler",
    "loading.output_list",
    "loading.constants",
    "properties",
    "visualisation",
]


@pytest.mark.parametrize("module_name", modules_to_test)
def test_import_top_modules(module_name):
    try:

        if any(s.isupper() for s in module_name) and len(module_name.split('.')) > 1:

            class_name = module_name.split('.')[-1]
            submodule_name = module_name.rsplit('.', 1)[0]
            submodule_fullname = '.'.join([swiftzoom.name, submodule_name])

            module = __import__(submodule_fullname, fromlist=[class_name])
            getattr(module, class_name)

        else:
            __import__(f"{swiftzoom.name:s}.{module_name:s}")

    except (ModuleNotFoundError, AttributeError):
        pytest.fail(f"Module '{module_name:s}' not found in the {swiftzoom.name:s} package.")


if __name__ == '__main__':
    pytest.main()
