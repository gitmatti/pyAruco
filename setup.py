# This setup code has been slightly modified from its original version by
# NorthernStars (https://github.com/NorthernStars/python-aruco/blob/master/Aruco/src/setup.py)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ["*.pyx", "arucofidmarkers.cpp",
               "board.cpp", "boarddetector.cpp",
               "cameraparameters.cpp", "cvdrawingutils.cpp",
               "marker.cpp", "markerdetector.cpp"]

extension=[Extension("*",
        sourcefiles,
        language="c++",
        include_dirs = ["/usr/include/opencv"],
        libraries = ["opencv_core", "opencv_imgproc", "opencv_calib3d"],
        library_dirs = ["/usr/local/lib"])]

setup(name = "PyAruco",
      ext_modules = cythonize(extension),
)
