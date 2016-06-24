# Copyright 2015 Mathias Schmerling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#ctypedef void* int_parameter
#ctypedef int_parameter two "2"
#ctypedef Point_[float, two] Point2f


import numpy as np
cimport numpy as np # for np.ndarray
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython.ref cimport PyObject


# Declares OpenCV's cv::Mat, cv:Size and cv::Vec2f class
cdef extern from "opencv2/core/core.hpp":
    cdef cppclass Mat:
        Mat() except +
    cdef cppclass Vec2f:# "cv::Vec<float, 2>":
        Vec2f() except +
        Vec2f(float v0, float v1)
        float operator[](int)
    cdef cppclass Size:
        Size() except +
        Size(float width, float height)

# Declares the official wrapper conversion functions 
# and NumPy's import_array() function
cdef extern from "conversion.cpp":
    void import_array()
    PyObject* pyopencv_from(const Mat&)
    int pyopencv_to(PyObject*, Mat&)


# Function to be called at initialization
cdef void conversion_init():
    import_array()

# Python to C++ conversion
cdef Mat nparrayToMat(object array):
    cdef Mat mat
    cdef PyObject* pyobject = <PyObject*> array
    pyopencv_to(pyobject, mat)
    return <Mat> mat

# C++ to Python conversion
cdef object matToNparray(Mat mat):
    return <object> pyopencv_from(mat)
cdef object vec2fToNparray(Vec2f vec2f):
    cdef float x = vec2f[0]
    cdef float y = vec2f[1]
    return np.array([x, y])



cdef extern from "marker.h" namespace "aruco":
    cdef cppclass Marker:
        Marker() except +
        int id
        float ssize
        Mat Rvec
        Mat Tvec
        float getArea()
        Vec2f getCenter()


cdef extern from "cameraparameters.h" namespace "aruco":
    cdef cppclass CameraParameters:
        CameraParameters() except +
        CameraParameters(Mat, Mat, Size)
        Mat CameraMatrix
        Mat Distorsion
        Size camSize
        void readFromXMLFile(string)


cdef extern from "markerdetector.h" namespace "aruco":
    cdef cppclass MarkerDetector:
        MarkerDetector() except +
        #~MarkerDetector()
        void detect(Mat&, vector[Marker]& , CameraParameters, float)


cdef class PyDetector:
    cdef vector[Marker] Markers
    cdef Mat InImage
    cdef CameraParameters CamParam
    cdef MarkerDetector MDetector
    def __cinit__(self):
        conversion_init()
        self.MDetector = MarkerDetector()
    def readCamParamFromXMLFile(self, xmlfname):
        self.CamParam.readFromXMLFile(xmlfname)
    def setCamParam(self, matrix, distorsion, size):
        cdef Size CamSize = Size(size[0],size[1])
        cdef Mat CamMatrix = nparrayToMat(matrix)
        cdef Mat CamDistorsion = nparrayToMat(distorsion)
        self.CamParam = CameraParameters(CamMatrix, CamDistorsion, CamSize)
    def __dealloc__(self):
        pass
    def detect(self, InImage, markersize=0.04):
        self.InImage = nparrayToMat(InImage)
        self.MDetector.detect(self.InImage,
                              self.Markers,
                              self.CamParam,
                              markersize)
    def getInImage(self):
        return matToNparray(self.InImage)
    def getID(self):
        if self.Markers.size()==0:
            return []
        else:
            out = []
            for i in range(self.Markers.size()):
                out.append(self.Markers[i].id)
            return out        
    def getTvec(self):
        if self.Markers.size()==0:
            return []
        else:
            out = []
            for i in range(self.Markers.size()):
                out.append(matToNparray(self.Markers[i].Tvec))
            return out
    def getRvec(self):
        if self.Markers.size()==0:
            return []
        else:
            out = []
            for i in range(self.Markers.size()):
                out.append(matToNparray(self.Markers[i].Rvec))
            return out
    def getCenter(self):
        if self.Markers.size()==0:
            return []
        else:
            out = []
            for i in range(self.Markers.size()):
                out.append(vec2fToNparray(self.Markers[i].getCenter()))
            return out


