// Numeric loops shared by the pinned random-room CAP policies.
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include "cap_heap.h"

static double movement_gap(double px, double py, double ux, double uy, double rad, double clear) {
    double f = px*ux + py*uy, lateral = fabs(px*uy-py*ux);
    if (f > 0 && lateral < rad) {
        double gap = f-sqrt(fmax(0.,rad*rad-lateral*lateral));
        if (gap < clear) return gap;
    }
    return clear;
}

// xy contains x,y,is_target,is_teammate for each measured/map point.
void cap_clearances(const double *xy, int n, const double *g, const double *own,
                    const double *dirs, double *out) {
    for (int k=0; k<8; ++k) {
        double ux=dirs[2*k], uy=dirs[2*k+1];
        out[k]=2.;
        for (int step=1; step<=20; ++step) {
            double x=g[0]+ux*step*.1, y=g[1]+uy*step*.1;
            int hit=0;
            for (int i=0; i<n; ++i) {
                const double *p=xy+4*i;
                if (p[2]) continue;
                double dx=p[0]-x, dy=p[1]-y;
                if (fabs(dx*g[2]+dy*g[3])<.52 && fabs(dx*(-g[3])+dy*g[2])<.145) {
                    hit=1; break;
                }
            }
            if (hit) { out[k]=(step-1)*.1; break; }
        }
        for (int j=0; j<2; ++j) {
            double x=own[2*j], y=own[2*j+1], clear=2.;
            for (int i=0; i<n; ++i) {
                const double *p=xy+4*i;
                clear=movement_gap(p[0]-x,p[1]-y,ux,uy,p[3] ? .38 : .2,clear);
            }
            out[8+8*j+k]=movement_gap(own[2*(1-j)]-x,own[2*(1-j)+1]-y,ux,uy,.38,clear);
        }
    }
}

typedef double (*ExactHypot)(double,double);

// Squared distances are well separated on most cells. At a possible rounding
// boundary ask the same CPython math.hypot used by the original policy.
static int near_comparison(double a,double b) {
    return !isfinite(a) || !isfinite(b) || fabs(a-b)<=16*DBL_EPSILON*fmax(fabs(a),fabs(b));
}
static int inside(double x,double y,double r,ExactHypot exact) {
    double q=x*x+y*y, rr=r*r;
    return near_comparison(q,rr) ? exact(x,y)<r : q<rr;
}

static int build_grid(uint8_t *grid,int nx,int ny,const double *arena,const double *g,
                      const double *xy,int n,const double *peer,const double *goal,ExactHypot exact) {
    const double step=.12;
    for (int i=0;i<nx;++i) {
        double x=(i+.5)*step;
        for (int j=0;j<ny;++j) {
            double y=(j+.5)*step,dx=x-g[0],dy=y-g[1];
            grid[i*ny+j]=(x<.23 || y<.23 || x>arena[0]-.23 || y>arena[1]-.23 ||
                (fabs(dx*g[2]+dy*g[3])<.705 && fabs(dx*(-g[3])+dy*g[2])<.330) ||
                inside(x-peer[0],y-peer[1],.40,exact));
        }
    }
    for (int p=0;p<n;++p) {
        double x=xy[2*p],y=xy[2*p+1];
        // Outside this range the clipped loops in the frozen program are empty.
        if (x/step < -3 || y/step < -3 || x/step >= nx+2 || y/step >= ny+2) continue;
        int ii=(int)(x/step),jj=(int)(y/step);
        for(int i=ii-2;i<=ii+2;++i) {
            if(i<0 || i>=nx) continue;
            for(int j=jj-2;j<=jj+2;++j) {
                if(j>=0 && j<ny && inside((i+.5)*step-x,(j+.5)*step-y,.205,exact))
                    grid[i*ny+j]=1;
            }
        }
    }
    int best=-1;
    double best_q=0.,best_x=0.,best_y=0.;
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) {
        int cell=i*ny+j;
        if(grid[cell]) continue;
        double dx=(i+.5)*step-goal[0],dy=(j+.5)*step-goal[1],q=dx*dx+dy*dy;
        if(best<0 || (near_comparison(q,best_q) ? exact(dx,dy)<exact(best_x,best_y) : q<best_q)) {
            best=cell;best_q=q;best_x=dx;best_y=dy;
        }
    }
    return best;
}

// Paths are returned end-to-start; zero is an exhausted search, -2 an empty menu.
int cap_navigate(int nx,int ny,const double *arena,const double *g,const double *xy,int n,
                 const double *pos,const double *peer,const double *goal,int start,
                 const double *heuristic,ExactHypot exact,uint8_t *grid,int *path,int *stats) {
    if(nx<1 || ny<1 || nx>128 || ny>128 || start<0 || start>=nx*ny || n<0) return -1;
    int cells=nx*ny, si=start/ny,sj=start%ny;
    int end=build_grid(grid,nx,ny,arena,g,xy,n,peer,goal,exact);
    if(end<0) return -2;
    grid[start]=0;
    for(int i=si-1;i<=si+1;++i) for(int j=sj-1;j<=sj+1;++j) {
        if(i>=0 && i<nx && j>=0 && j<ny && inside((i+.5)*.12-pos[0],(j+.5)*.12-pos[1],.17,exact))
            grid[i*ny+j]=0;
    }
    double *cost=malloc(cells*sizeof(*cost));
    int *parent=malloc(cells*sizeof(*parent));
    uint8_t *closed=calloc(cells,1);
    Heap heap={.entries=malloc((1+8*cells)*sizeof(Entry)),.last_positive_pop_size=-1};
    int length=-1;
    if(!cost || !parent || !closed || !heap.entries) goto done;
    for(int i=0;i<cells;++i) cost[i]=1e9;
    cost[start]=0.;push(&heap,0.,start);
    static const int moves[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    length=0;
    for(int it=0;it<cells && heap.size;++it) {
        int k=pop(&heap);
        if(closed[k]) continue;
        closed[k]=1;
        if(k==end) {
            for(int last=k;last!=start;last=parent[last]) path[length++]=last;
            path[length++]=start; break;
        }
        int i=k/ny,j=k%ny;
        for(int d=0;d<8;++d) {
            int di=moves[d][0],dj=moves[d][1],ni=i+di,nj=j+dj;
            if(ni<0 || ni>=nx || nj<0 || nj>=ny) continue;
            int nk=ni*ny+nj;
            if(grid[nk] || closed[nk] || (di && dj && (grid[i*ny+nj] || grid[ni*ny+j]))) continue;
            double next=cost[k]+(di && dj ? 1.41421356 : 1.);
            if(next<cost[nk]) {
                cost[nk]=next;parent[nk]=k;
                push(&heap,next+heuristic[abs(ni-end/ny)*ny+abs(nj-end%ny)],nk);
            }
        }
    }
    stats[0]=end;stats[1]=heap.last_push_index;stats[2]=heap.last_pop_size;stats[3]=heap.last_positive_pop_size;
done:
    free(heap.entries);free(closed);free(parent);free(cost);
    return length;
}
