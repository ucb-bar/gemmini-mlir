#include <stdint.h>
extern void cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
static float af[1024*512],bf[1024*512],cf[1024*1024];
static void run(const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k){
 for(int t=0;t<m*k;t++)af[t]=a[t];for(int t=0;t<k*n;t++)bf[t]=b[t];
 cblas_sgemm(101,111,111,m,n,k,1.0f,af,k,bf,n,0.0f,cf,n);
 for(int t=0;t<m*n;t++)c[t]=(int32_t)cf[t];}
void qk_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,1024,64);}
void pv512_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,512);}
void pv192_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,192);}
void pv128_plane(const int8_t*a,const int8_t*b,int32_t*c){run(a,b,c,1024,64,128);}
