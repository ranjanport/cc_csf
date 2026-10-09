#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CSF.h"
#ifdef CSF_USE_OPENMP
#include <omp.h>
#endif
#include <cstring>

namespace py = pybind11;

class CSFWrapper {
public:
    CSF csf;
    
    CSFWrapper() {}
    
    void set_params(bool bSloopSmooth, double cloth_resolution, int iterations, double class_threshold, int rigidness) {
        csf.params.bSloopSmooth = bSloopSmooth;
        csf.params.cloth_resolution = cloth_resolution;
        csf.params.interations = iterations;
        csf.params.class_threshold = class_threshold;
        csf.params.rigidness = rigidness;
    }

    void set_num_threads(int threads) {
#ifdef CSF_USE_OPENMP
        if (threads > 0) {
            omp_set_num_threads(threads);
        }
#endif
    }
    
    void set_point_cloud(py::array_t<double> xyz) {
        py::buffer_info buf = xyz.request();
        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("Input array must be Nx3");
        }
        
        int rows = static_cast<int>(buf.shape[0]);
        double *ptr = static_cast<double *>(buf.ptr);
        
        csf.getPointCloud().resize(rows);
        
        #ifdef CSF_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < rows; i++) {
            double px = ptr[i * 3 + 0];
            double py = ptr[i * 3 + 1];
            double pz = ptr[i * 3 + 2];
            
            csf::Point p;
            p.x = px;
            p.y = -pz;
            p.z = py;
            
            csf.getPointCloud()[i] = p;
        }
    }
    
    py::array_t<int> do_filtering() {
        std::vector<int> ground_indexes, off_ground_indexes;
        {
            py::gil_scoped_release release;
            csf.do_filtering(ground_indexes, off_ground_indexes);
        }
        
        py::array_t<int> result(static_cast<py::ssize_t>(ground_indexes.size()));
        if (!ground_indexes.empty()) {
            std::memcpy(result.mutable_data(), ground_indexes.data(), ground_indexes.size() * sizeof(int));
        }
        return result;
    }
};

PYBIND11_MODULE(cc_csf, m) {
    m.doc() = "Zero-copy Pybind11 wrapper for CloudCompare CSF with OpenMP";
    
    py::class_<CSFWrapper>(m, "CSF")
        .def(py::init<>())
        .def("set_params", &CSFWrapper::set_params)
        .def("set_num_threads", &CSFWrapper::set_num_threads)
        .def("set_point_cloud", &CSFWrapper::set_point_cloud)
        .def("do_filtering", &CSFWrapper::do_filtering);
}
