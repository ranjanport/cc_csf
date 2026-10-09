// ======================================================================================
// Copyright 2017 State Key Laboratory of Remote Sensing Science, 
// Institute of Remote Sensing Science and Engineering, Beijing Normal University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ======================================================================================

#include "point_cloud.h"
#ifdef CSF_USE_OPENMP
#include <omp.h>
#endif

void csf::PointCloud::computeBoundingBox(Point& bbMin, Point& bbMax) {
    if (empty()) {
        bbMin = bbMax = Point();
        return;
    }

    bbMin = bbMax = at(0);
    std::size_t n = size();

#ifdef CSF_USE_OPENMP
    double min_x = bbMin.x, min_y = bbMin.y, min_z = bbMin.z;
    double max_x = bbMax.x, max_y = bbMax.y, max_z = bbMax.z;

    #pragma omp parallel for reduction(min:min_x, min_y, min_z) reduction(max:max_x, max_y, max_z) schedule(static)
    for (std::size_t i = 1; i < n; i++) {
        const csf::Point& P = (*this)[i];
        if (P.x < min_x) min_x = P.x;
        if (P.x > max_x) max_x = P.x;
        if (P.y < min_y) min_y = P.y;
        if (P.y > max_y) max_y = P.y;
        if (P.z < min_z) min_z = P.z;
        if (P.z > max_z) max_z = P.z;
    }
    bbMin.x = min_x; bbMin.y = min_y; bbMin.z = min_z;
    bbMax.x = max_x; bbMax.y = max_y; bbMax.z = max_z;
#else
    for (std::size_t i = 1; i < n; i++) {
        const csf::Point& P = at(i);
        for (int d = 0; d < 3; ++d) {
            if (P.u[d] < bbMin.u[d]) {
                bbMin.u[d] = P.u[d];
            } else if (P.u[d] > bbMax.u[d]) {
                bbMax.u[d] = P.u[d];
            }
        }
    }
#endif
}
