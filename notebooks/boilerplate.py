try:
    from matplotlib import pyplot as plt
  
    # Use the MNRAS style for the plot
    plt.style.use("mnras.mplstyle")
except:
    print(('Matplotlib stylesheet `mnras.mplstyle` not found. You can download it from ' 
           'https://github.com/edoaltamura/matplotlib-stylesheets - Reverting to default.'))
    
import h5py as h5
h5.get_config().default_file_mode = 'r'

import sys
sys.path.append('..')
sys.path.append('../swiftzoom')
