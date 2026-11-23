// CAP009's 16 collision grids and eight-orientation bounded A*.
// The caller supplies the original Python trigonometry and hypot values.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define STEP 0.24
#define LIMIT 12000

#include "cap_heap.h"

static int clipped(double v, int n) {
    if (v <= 0) return 0;
    if (v >= n) return n;
    return (int)v;
}

static void rasterize(uint8_t *blocked, int nx, int ny, double width,
                      double height, const double *axes, const double *xy, int count) {
    for (int k = 0; k < 16; ++k) {
        uint8_t *grid = blocked + k*nx*ny;
        double ux = axes[2*k], uy = axes[2*k+1];
        double vx = -uy, vy = ux;
        double ex = .53*fabs(ux) + .155*fabs(vx);
        double ey = .53*fabs(uy) + .155*fabs(vy);
        for (int i = 0; i < nx; ++i) {
            double x = (i+.5)*STEP;
            for (int j = 0; j < ny; ++j) {
                double y = (j+.5)*STEP;
                grid[i*ny+j] = x < ex+.05 || x > width-ex-.05 ||
                               y < ey+.05 || y > height-ey-.05;
            }
        }
        for (int p = 0; p < count; ++p) {
            double px = xy[2*p], py = xy[2*p+1];
            int lo_x = clipped((px-ex)/STEP,nx), lo_y = clipped((py-ey)/STEP,ny);
            // Match min(n, int(upper) + 1), including truncation toward zero.
            int hi_x = clipped(trunc((px+ex)/STEP)+1,nx);
            int hi_y = clipped(trunc((py+ey)/STEP)+1,ny);
            for (int i = lo_x; i < hi_x; ++i) {
                double dx = px-(i+.5)*STEP;
                for (int j = lo_y; j < hi_y; ++j) {
                    double dy = py-(j+.5)*STEP;
                    if (fabs(dx*ux+dy*uy)<.53 && fabs(dx*vx+dy*vy)<.155)
                        grid[i*ny+j] = 1;
                }
            }
        }
    }
}

// path is returned in reverse order (goal to start); stats are real search events.
int cap009_route(int nx, int ny, double width, double height, int start,
                 const double *axes, const int *dirs, const double *xy, int count,
                 const double *heuristic, int *path, int *stats) {
    if (nx < 1 || ny < 1 || nx > 128 || ny > 128 || count < 0) return -1;
    int cells = nx*ny, nodes = 8*cells;
    if (start < 0 || start >= cells) return -1;
    uint8_t *blocked = malloc((size_t)16*cells);
    uint8_t *closed = calloc(nodes,1);
    double *costs = malloc(nodes*sizeof(*costs));
    int *parents = malloc(nodes*sizeof(*parents));
    // Each expanded node can improve at most four translations and two rotations.
    int expansions = nodes < LIMIT ? nodes : LIMIT;
    Heap heap = {.entries=malloc((1+6*expansions)*sizeof(Entry)), .last_positive_pop_size=-1};
    int length = -1;
    if (!blocked || !closed || !costs || !parents || !heap.entries) goto done;
    rasterize(blocked,nx,ny,width,height,axes,xy,count);
    for (int i = 0; i < nodes; ++i) costs[i] = 1e9;
    blocked[start] = 0;
    costs[start] = 0;
    push(&heap,0,start);
    int best = start, found = -1;
    double best_h = 1e9;
    for (int it = 0; it < LIMIT && heap.size; ++it) {
        int node = pop(&heap);
        if (closed[node]) continue;
        closed[node] = 1;
        int k = node/cells, cell = node%cells, i = cell/ny, j = cell%ny;
        double h = heuristic[cell];
        if (h < best_h) { best = node; best_h = h; }
        if (h < .80) { found = node; break; }
        const uint8_t *grid = blocked+2*k*cells;
        for (int d = 0; d < 4; ++d) {
            int di = dirs[8*k+2*d], dj = dirs[8*k+2*d+1];
            int ni = i+di, nj = j+dj;
            if (ni<0 || ni>=nx || nj<0 || nj>=ny) continue;
            int nc = ni*ny+nj, nn = k*cells+nc;
            if (grid[nc] || closed[nn]) continue;
            if (di && dj && (grid[i*ny+nj] || grid[ni*ny+j])) continue;
            double cost = costs[node]+(di && dj ? 1.41421356 : 1.);
            if (cost < costs[nn]) {
                costs[nn] = cost; parents[nn] = node;
                push(&heap,cost+heuristic[nc],nn);
            }
        }
        for (int sign = -1; sign <= 1; sign+=2) {
            int nk = (k+sign+8)%8, mid = (2*k+sign+16)%16, nn = nk*cells+cell;
            if (blocked[2*nk*cells+cell] || blocked[mid*cells+cell] || closed[nn]) continue;
            double cost = costs[node]+2.5;
            if (cost < costs[nn]) {
                costs[nn] = cost; parents[nn] = node;
                push(&heap,cost+h,nn);
            }
        }
    }
    stats[0]=found; stats[1]=best; stats[2]=heap.last_push_index;
    stats[3]=heap.last_pop_size; stats[4]=heap.last_positive_pop_size;
    int last = found < 0 ? best : found;
    length = 0;
    while (last != start) {
        path[length++] = last;
        last = parents[last];
    }
    path[length++] = start;
done:
    free(heap.entries); free(parents); free(costs); free(closed); free(blocked);
    return length;
}
