/* Analytic lidar arithmetic matches lidar.scan_lidar_batch. No fast math/FMA.
 * Geometry is owned by the host; no simulator pointers or mutable static state.
 * Each primitive preserves the NumPy expression's operation order. */
#include <stddef.h>
#include <stdint.h>
#include <math.h>

static void accept(double t, int64_t entry, double *best, int64_t *owner) {
    if (t < *best || (t == *best && entry < *owner)) {
        *best = t;
        *owner = entry;
    }
}

void coimagine_lidar_scan(size_t n, const double *directions,
        double ox, double oy, double range,
        size_t ns, const double *segments, const int64_t *segment_owners,
        size_t nc, const double *circles, const int64_t *circle_owners,
        double *distances, int64_t *owners) {
    for (size_t i = 0; i < n; ++i) {
        double dx = directions[2*i], dy = directions[2*i+1];
        double best = INFINITY;
        int64_t owner = INT64_MAX;
        for (size_t j = 0; j < ns; ++j) {
            /* ax, ay, ex, ey, radius, length, unit_x, unit_y */
            const double *s = segments + 8*j;
            double fx = s[0]-ox, fy = s[1]-oy;
            double denom = dx*s[3]-dy*s[2];
            double safe = fabs(denom) >= 1e-12 ? denom : 1.;
            double t = (fx*s[3]-fy*s[2])/safe;
            double u = (fx*dy-fy*dx)/safe;
            if (!(fabs(denom) >= 1e-12 && t >= 0. && u >= -1e-9 && u <= 1.+1e-9)) continue;
            if (s[4] > 0.) {
                double sine = fabs(dx*s[7]-dy*s[6]);
                double candidate = t-s[4]/fmax(sine,1e-30);
                double px = ox+candidate*dx, py = oy+candidate*dy;
                double along = ((px-s[0])*s[6]+(py-s[1])*s[7])/fmax(s[5],1e-12);
                if (!(s[5] > 1e-12 && sine > 1e-9 && along >= -1e-9 && along <= 1.+1e-9)) continue;
                t = candidate;
            }
            if (t >= 0. && t <= range) accept(t,segment_owners[j],&best,&owner);
        }
        for (size_t j = 0; j < nc; ++j) {
            const double *c = circles + 3*j;
            double fx = ox-c[0], fy = oy-c[1];
            double bb = 2.*(dx*fx+dy*fy);
            double cc = fx*fx+fy*fy-c[2]*c[2];
            double disc = bb*bb-4.*cc;
            double root = sqrt(fmax(disc,0.));
            double first = (-bb-root)/2., second = (-bb+root)/2.;
            double t = first >= 0. ? first : second;
            if (disc >= 0. && t >= 0. && t <= range) accept(t,circle_owners[j],&best,&owner);
        }
        distances[i] = isfinite(best) ? best : 0.;
        owners[i] = isfinite(best) ? owner : -1;
    }
}
