#include <math.h>
#include <stddef.h>
void ordered_source_dot(const float*a,const float*b,float*out,int m,int n,int k){
 for(int r=0;r<m;r++)for(int j=0;j<n;j++){
  float acc=0.0f;
  for(int z=0;z<k;z++)acc=fmaf(a[(size_t)r*k+z],b[(size_t)j*k+z],acc);
  out[(size_t)r*n+j]=acc;
 }
}
