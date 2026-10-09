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

#include "c2cdist.h"
#include <cmath>
#ifdef CSF_USE_OPENMP
#include <omp.h>
#endif

void c2cdist::calCloud2CloudDist(Cloth           & cloth,
                                 csf::PointCloud & pc,
                                 std::vector<int>& groundIndexes,
                                 std::vector<int>& offGroundIndexes) {
    groundIndexes.clear();
    offGroundIndexes.clear();

#ifdef CSF_USE_OPENMP
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> th_ground(max_threads);
    std::vector<std::vector<int>> th_off(max_threads);

    std::size_t pc_size = pc.size();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < pc_size; i++) {
            double pc_x = pc[i].x;
            double pc_z = pc[i].z;

            double deltaX = pc_x - cloth.origin_pos.f[0];
            double deltaZ = pc_z - cloth.origin_pos.f[2];

            int col0 = int(deltaX / cloth.step_x);
            int row0 = int(deltaZ / cloth.step_y);
            int col1 = col0 + 1;
            int row1 = row0;
            int col2 = col0 + 1;
            int row2 = row0 + 1;
            int col3 = col0;
            int row3 = row0 + 1;

            if (col0 < 0 || col2 >= cloth.num_particles_width ||
                row0 < 0 || row2 >= cloth.num_particles_height) {
                th_off[tid].push_back(static_cast<int>(i));
                continue;
            }

            double subdeltaX = (deltaX - col0 * cloth.step_x) / cloth.step_x;
            double subdeltaZ = (deltaZ - row0 * cloth.step_y) / cloth.step_y;

            double fxy
                = cloth.getParticle(col0, row0)->pos.f[1] * (1 - subdeltaX) * (1 - subdeltaZ) +
                  cloth.getParticle(col3, row3)->pos.f[1] * (1 - subdeltaX) * subdeltaZ +
                  cloth.getParticle(col2, row2)->pos.f[1] * subdeltaX * subdeltaZ +
                  cloth.getParticle(col1, row1)->pos.f[1] * subdeltaX * (1 - subdeltaZ);
            double height_var = fxy - pc[i].y;

            if (std::fabs(height_var) < class_treshold) {
                th_ground[tid].push_back(static_cast<int>(i));
            } else {
                th_off[tid].push_back(static_cast<int>(i));
            }
        }
    }

    std::size_t total_g = 0;
    std::size_t total_off = 0;
    for (int t = 0; t < max_threads; ++t) {
        total_g += th_ground[t].size();
        total_off += th_off[t].size();
    }
    groundIndexes.reserve(total_g);
    offGroundIndexes.reserve(total_off);
    for (int t = 0; t < max_threads; ++t) {
        groundIndexes.insert(groundIndexes.end(), th_ground[t].begin(), th_ground[t].end());
        offGroundIndexes.insert(offGroundIndexes.end(), th_off[t].begin(), th_off[t].end());
    }
#else
    for (std::size_t i = 0; i < pc.size(); i++) {
        double pc_x = pc[i].x;
        double pc_z = pc[i].z;

        double deltaX = pc_x - cloth.origin_pos.f[0];
        double deltaZ = pc_z - cloth.origin_pos.f[2];

        int col0 = int(deltaX / cloth.step_x);
        int row0 = int(deltaZ / cloth.step_y);
        int col1 = col0 + 1;
        int row1 = row0;
        int col2 = col0 + 1;
        int row2 = row0 + 1;
        int col3 = col0;
        int row3 = row0 + 1;

        if (col0 < 0 || col2 >= cloth.num_particles_width ||
            row0 < 0 || row2 >= cloth.num_particles_height) {
            offGroundIndexes.push_back(static_cast<int>(i));
            continue;
        }

        double subdeltaX = (deltaX - col0 * cloth.step_x) / cloth.step_x;
        double subdeltaZ = (deltaZ - row0 * cloth.step_y) / cloth.step_y;

        double fxy
            = cloth.getParticle(col0, row0)->pos.f[1] * (1 - subdeltaX) * (1 - subdeltaZ) +
              cloth.getParticle(col3, row3)->pos.f[1] * (1 - subdeltaX) * subdeltaZ +
              cloth.getParticle(col2, row2)->pos.f[1] * subdeltaX * subdeltaZ +
              cloth.getParticle(col1, row1)->pos.f[1] * subdeltaX * (1 - subdeltaZ);
        double height_var = fxy - pc[i].y;

        if (std::fabs(height_var) < class_treshold) {
            groundIndexes.push_back(static_cast<int>(i));
        } else {
            offGroundIndexes.push_back(static_cast<int>(i));
        }
    }
#endif
}
