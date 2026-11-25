// Incremental, bounded-resolution public evidence. No simulator state API.
// Closed triangles/quadrilaterals are bounded entirely by observed free rays.
// Their area must be smaller than the declared minimum static component area.
// A grid cell becomes FREE only after polygon subtraction covers its WHOLE area.
// Incomplete, over-complex, and out-of-map cells stay UNKNOWN.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>

namespace {
using F = long double;
struct P { F x, y; };
using Poly = std::vector<P>;
P operator+(P a,P b){return {a.x+b.x,a.y+b.y};}
P operator-(P a,P b){return {a.x-b.x,a.y-b.y};}
P operator*(P a,F b){return {a.x*b,a.y*b};}
F cross(P a,P b){return a.x*b.y-a.y*b.x;}
F area(const Poly& p){F a=0;for(size_t i=0;i<p.size();++i)a+=cross(p[i],p[(i+1)%p.size()]);return a/2;}
Poly half(const Poly& p,P a,P b,bool inside){
  Poly out;if(p.empty())return out;
  P prev=p.back();F dp=cross(b-a,prev-a);bool pin=inside?dp>=0:dp<=0;
  for(P now:p){F dn=cross(b-a,now-a);bool nin=inside?dn>=0:dn<=0;
    if(pin!=nin){F t=dp/(dp-dn);out.push_back(prev+(now-prev)*t);}
    if(nin)out.push_back(now);prev=now;dp=dn;pin=nin;
  }
  if(out.size()<3 || std::abs(area(out))<1e-24L)out.clear();
  return out;
}
bool convex(Poly& p){
  if(p.size()<3)return false;
  if(area(p)<0)std::reverse(p.begin(),p.end());
  if(area(p)<1e-12L)return false;
  for(size_t i=0;i<p.size();++i)
    if(cross(p[(i+1)%p.size()]-p[i],p[(i+2)%p.size()]-p[(i+1)%p.size()]) < -1e-18L)return false;
  return true;
}
struct Ray {P a,b;};
bool intersection(const Ray& a,const Ray& b,P& p){
  P u=a.b-a.a,v=b.b-b.a,w=b.a-a.a;F d=cross(u,v);
  if(std::abs(d)<1e-14L)return false;
  F t=cross(w,v)/d,s=cross(w,u)/d;
  if(t<0 || t>1 || s<0 || s>1)return false;
  p=a.a+u*t;return true;
}
struct Cell {bool free=false;bool initialized=false;std::vector<Poly> remain;};
struct Grid {
  double h,w,height;int nx,ny;F min_area;
  std::vector<Cell> cells;std::vector<std::vector<Ray>> scans;
  F wx0=0,wy0=0,wx1,wy1;
  uint64_t certificates=0,resets=0;
  Grid(double x,double y,double s,double m):h(s),w(x),height(y),nx(std::ceil(x/s)),ny(std::ceil(y/s)),min_area(m),cells(nx*ny),wx1(x),wy1(y){}
  Poly square(int x,int y)const{F a=x*F(h),b=y*F(h);return {{a,b},{a+h,b},{a+h,b+h},{a,b+h}};}
  void add(Poly poly,bool initial=false){
    if(!convex(poly))return;
    if(!initial && !(area(poly)<min_area-1e-8L))return;
    ++certificates;
    F x0=poly[0].x,x1=x0,y0=poly[0].y,y1=y0;
    for(P p:poly){x0=std::min(x0,p.x);x1=std::max(x1,p.x);y0=std::min(y0,p.y);y1=std::max(y1,p.y);}
    if(!initial){x0=std::max(x0,wx0);x1=std::min(x1,wx1);y0=std::max(y0,wy0);y1=std::min(y1,wy1);if(x0>x1||y0>y1)return;}
    int ax=std::max(0,int(std::floor(x0/h))),bx=std::min(nx-1,int(std::floor(x1/h)));
    int ay=std::max(0,int(std::floor(y0/h))),by=std::min(ny-1,int(std::floor(y1/h)));
    for(int y=ay;y<=by;++y)for(int x=ax;x<=bx;++x){
      Cell& c=cells[y*nx+x];if(c.free)continue;
      // Most bounding-box cells miss a thin ray polygon entirely. Native
      // separating-axis tests avoid allocating/clipping them. Fully enclosed
      // cells are certified directly; only boundary cells need subtraction.
      F cx=(x+.5L)*h,cy=(y+.5L)*h;bool outside=false,inside=true;
      for(size_t i=0;i<poly.size();++i){P d=poly[(i+1)%poly.size()]-poly[i];
        F side=cross(d,{cx-poly[i].x,cy-poly[i].y});F radius=(std::abs(d.x)+std::abs(d.y))*h/2;
        if(side+radius<0){outside=true;break;}if(side-radius<0)inside=false;}
      if(outside)continue;
      if(inside){c.free=true;c.initialized=true;c.remain.clear();continue;}
      if(!c.initialized){c.remain={square(x,y)};c.initialized=true;}
      std::vector<Poly> remain;
      for(const Poly& p:c.remain){
        Poly pending=p;
        for(size_t i=0;i<poly.size()&&!pending.empty();++i){
          Poly outside=half(pending,poly[i],poly[(i+1)%poly.size()],false);
          if(!outside.empty())remain.push_back(std::move(outside));
          pending=half(pending,poly[i],poly[(i+1)%poly.size()],true);
        }
      }
      if(remain.size()>48){c.remain={square(x,y)};++resets;} // lose evidence, never invent free cells
      else {c.remain=std::move(remain);if(c.remain.empty()){c.free=true;c.remain.shrink_to_fit();}}
    }
  }
  void pair(const std::vector<Ray>& a,const std::vector<Ray>& b){
    if(std::hypot(double(a[0].a.x-b[0].a.x),double(a[0].a.y-b[0].a.y))<1e-8)return;
    std::array<P,72*72> points;std::array<bool,72*72> ok;
    for(int i=0;i<72;++i)for(int j=0;j<72;++j)ok[i*72+j]=intersection(a[i],b[j],points[i*72+j]);
    for(int i=0;i<72;++i)for(int j=0;j<72;++j){
      int ni=(i+1)%72,nj=(j+1)%72;
      if(ok[i*72+j] && ok[ni*72+j])add({a[i].a,points[i*72+j],points[ni*72+j]});
      if(ok[i*72+j] && ok[i*72+nj])add({b[j].a,points[i*72+j],points[i*72+nj]});
      if(ok[i*72+j] && ok[ni*72+j] && ok[ni*72+nj] && ok[i*72+nj])
        add({points[i*72+j],points[ni*72+j],points[ni*72+nj],points[i*72+nj]});
    }
  }
  void update(const double* v){
    for(int r=0;r<2;++r){std::vector<Ray> scan;scan.reserve(72);
      for(int i=0;i<72;++i){const double* p=v+(r*72+i)*4;scan.push_back({{p[0],p[1]},{p[2],p[3]}});}
      // Only new scan pairs; historical certificates remain in the grid.
      for(const auto& old:scans)pair(old,scan);
      scans.push_back(std::move(scan));if(scans.size()>2)scans.erase(scans.begin());
    }
  }
  // Test a convex outer envelope. Every intersected cell must have a complete
  // certificate. Native separating-axis rejection avoids polygon construction.
  double missing(const double* v,int n)const{
    if(n<3)return std::numeric_limits<double>::infinity();
    Poly p;for(int i=0;i<n;++i)p.push_back({v[2*i],v[2*i+1]});if(!convex(p))return std::numeric_limits<double>::infinity();
    F x0=p[0].x,x1=x0,y0=p[0].y,y1=y0;for(P q:p){x0=std::min(x0,q.x);x1=std::max(x1,q.x);y0=std::min(y0,q.y);y1=std::max(y1,q.y);}
    if(x0<0||y0<0||x1>w||y1>height)return double(area(p));
    int ax=std::max(0,int(std::floor(x0/h))),bx=std::min(nx-1,int(std::floor(x1/h)));
    int ay=std::max(0,int(std::floor(y0/h))),by=std::min(ny-1,int(std::floor(y1/h)));
    int unknown=0;
    for(int y=ay;y<=by;++y)for(int x=ax;x<=bx;++x){
      if(cells[y*nx+x].free)continue;
      F cx=(x+.5L)*h,cy=(y+.5L)*h;bool separate=false;
      for(int i=0;i<n;++i){P d=p[(i+1)%n]-p[i];F side=cross(d,{cx-p[i].x,cy-p[i].y});
        F radius=(std::abs(d.x)+std::abs(d.y))*h/2;
        if(side+radius<0){separate=true;break;}}
      if(!separate)++unknown;
    }
    return unknown*h*h;
  }
};
}

extern "C" {
void* cap_grid_new(double w,double h,double resolution,double min_area){
  if(!(w>0&&w<=50&&h>0&&h<=50&&resolution>=.005&&resolution<=.1&&min_area>0))return nullptr;
  try{return new Grid(w,h,resolution,min_area);}catch(...){return nullptr;}
}
void cap_grid_delete(void* p){delete static_cast<Grid*>(p);}
void* cap_grid_clone(void* p){try{return new Grid(*static_cast<Grid*>(p));}catch(...){return nullptr;}}
void cap_grid_window(void* p,double x0,double y0,double x1,double y1){auto* g=static_cast<Grid*>(p);g->wx0=x0;g->wy0=y0;g->wx1=x1;g->wy1=y1;}
int cap_grid_update(void* p,const double* rays){try{static_cast<Grid*>(p)->update(rays);return 0;}catch(...){return -1;}}
int cap_grid_certificate(void* p,const double* xy,int n){try{Poly q;for(int i=0;i<n;++i)q.push_back({xy[2*i],xy[2*i+1]});static_cast<Grid*>(p)->add(q,true);return 0;}catch(...){return -1;}}
double cap_grid_missing(void* p,const double* xy,int n){try{return static_cast<Grid*>(p)->missing(xy,n);}catch(...){return std::numeric_limits<double>::infinity();}}
void cap_grid_mask(void* p,uint8_t* out){auto* g=static_cast<Grid*>(p);for(size_t i=0;i<g->cells.size();++i)out[i]=g->cells[i].free;}
void cap_grid_stats(void* p,uint64_t* out){auto* g=static_cast<Grid*>(p);out[0]=g->certificates;out[1]=g->resets;out[2]=g->cells.size();out[3]=0;for(auto& c:g->cells)out[3]+=c.free;}
}
