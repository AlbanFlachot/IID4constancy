import subprocess
import shlex
import numpy as np


ref_reflectances = list(np.round(np.array([0.2, 0.4, 0.6]), 1))
test_reflectances = list(np.array([0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

command = 'blender --background --python render_testset.py --python-use-system-env ' \
          f'-- --filename testsets/images_eevee --renderer EEVEE --samples 128'
args = shlex.split(command)
subprocess.call(args)

