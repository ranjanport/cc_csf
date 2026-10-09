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

#include "Rasterization.h"
#include <queue>
#include <vector>
#ifdef CSF_USE_OPENMP
#include <omp.h>
#endif

double Rasterization::findHeightValByScanline(Particle *p, Cloth& cloth) {
    int xpos = p->pos_x;
    int ypos = p->pos_y;

    for (int i = xpos + 1; i < cloth.num_particles_width; i++) {
        double crresHeight = cloth.getParticle(i, ypos)->nearestPointHeight;

        if (crresHeight > MIN_INF)
            return crresHeight;
    }

    for (int i = xpos - 1; i >= 0; i--) {
        double crresHeight = cloth.getParticle(i, ypos)->nearestPointHeight;

        if (crresHeight > MIN_INF)
            return crresHeight;
    }

    for (int j = ypos - 1; j >= 0; j--) {
        double crresHeight = cloth.getParticle(xpos, j)->nearestPointHeight;

        if (crresHeight > MIN_INF)
            return crresHeight;
    }

    for (int j = ypos + 1; j < cloth.num_particles_height; j++) {
        double crresHeight = cloth.getParticle(xpos, j)->nearestPointHeight;

        if (crresHeight > MIN_INF)
            return crresHeight;
    }

    return findHeightValByNeighbor(p, cloth);
}

double Rasterization::findHeightValByNeighbor(Particle *p, Cloth& cloth) {
    std::queue<Particle *>  nqueue;
    int cloth_size = cloth.getSize();
    std::vector<bool> visited(cloth_size, false);

    int p_idx = p->pos_y * cloth.num_particles_width + p->pos_x;
    visited[p_idx] = true;

    for (std::size_t i = 0; i < p->neighborsList.size(); i++) {
        Particle *nbr = p->neighborsList[i];
        int nbr_idx = nbr->pos_y * cloth.num_particles_width + nbr->pos_x;
        visited[nbr_idx] = true;
        nqueue.push(nbr);
    }

    while (!nqueue.empty()) {
        Particle *pneighbor = nqueue.front();
        nqueue.pop();

        if (pneighbor->nearestPointHeight > MIN_INF) {
            return pneighbor->nearestPointHeight;
        }

        for (std::size_t i = 0; i < pneighbor->neighborsList.size(); i++) {
            Particle *ptmp = pneighbor->neighborsList[i];
            int idx = ptmp->pos_y * cloth.num_particles_width + ptmp->pos_x;
            if (!visited[idx]) {
                visited[idx] = true;
                nqueue.push(ptmp);
            }
        }
    }

    return MIN_INF;
}

void Rasterization::RasterTerrian(Cloth          & cloth,
                                  csf::PointCloud& pc,
                                  std::vector<double> & heightVal) {
    std::size_t pc_size = pc.size();
    int cloth_size = cloth.getSize();

#ifdef CSF_USE_OPENMP
    int max_threads = omp_get_max_threads();

    std::vector<std::vector<double>> th_dist(max_threads, std::vector<double>(cloth_size, MAX_INF));
    std::vector<std::vector<double>> th_height(max_threads, std::vector<double>(cloth_size, MIN_INF));
    std::vector<std::vector<std::size_t>> th_idx(max_threads, std::vector<std::size_t>(cloth_size, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < pc_size; i++) {
            double pc_x = pc[i].x;
            double pc_z = pc[i].z;

            double deltaX = pc_x - cloth.origin_pos.f[0];
            double deltaZ = pc_z - cloth.origin_pos.f[2];
            int col = int(deltaX / cloth.step_x + 0.5);
            int row = int(deltaZ / cloth.step_y + 0.5);

            if (col >= 0 && col < cloth.num_particles_width &&
                row >= 0 && row < cloth.num_particles_height) {
                int p_idx = row * cloth.num_particles_width + col;
                Particle *pt = cloth.getParticle1d(p_idx);
                double pc2particleDist = SQUARE_DIST(
                    pc_x, pc_z,
                    pt->getPos().f[0],
                    pt->getPos().f[2]
                );

                if (pc2particleDist < th_dist[tid][p_idx]) {
                    th_dist[tid][p_idx] = pc2particleDist;
                    th_height[tid][p_idx] = pc[i].y;
                    th_idx[tid][p_idx] = i;
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < cloth_size; i++) {
        Particle *pt = cloth.getParticle1d(i);
        for (int t = 0; t < max_threads; ++t) {
            if (th_dist[t][i] < pt->tmpDist) {
                pt->tmpDist = th_dist[t][i];
                pt->nearestPointHeight = th_height[t][i];
                pt->nearestPointIndex = th_idx[t][i];
            }
        }
    }
#else
    for (std::size_t i = 0; i < pc_size; i++) {
        double pc_x = pc[i].x;
        double pc_z = pc[i].z;

        double deltaX = pc_x - cloth.origin_pos.f[0];
        double deltaZ = pc_z - cloth.origin_pos.f[2];
        int col = int(deltaX / cloth.step_x + 0.5);
        int row = int(deltaZ / cloth.step_y + 0.5);

        if (col >= 0 && col < cloth.num_particles_width &&
            row >= 0 && row < cloth.num_particles_height) {
            Particle *pt = cloth.getParticle(col, row);
            double pc2particleDist = SQUARE_DIST(
                pc_x, pc_z,
                pt->getPos().f[0],
                pt->getPos().f[2]
            );

            if (pc2particleDist < pt->tmpDist) {
                pt->tmpDist            = pc2particleDist;
                pt->nearestPointHeight = pc[i].y;
                pt->nearestPointIndex  = i;
            }
        }
    }
#endif

    heightVal.resize(cloth_size);

#ifdef CSF_USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < cloth_size; i++) {
        Particle *pcur          = cloth.getParticle1d(i);
        double    nearestHeight = pcur->nearestPointHeight;

        if (nearestHeight > MIN_INF) {
            heightVal[i] = nearestHeight;
        } else {
            heightVal[i] = findHeightValByScanline(pcur, cloth);
        }
    }
}
