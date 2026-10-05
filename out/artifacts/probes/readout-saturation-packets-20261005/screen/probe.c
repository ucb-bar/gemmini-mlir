#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
static uint64_t clock_now(void){
#ifdef __riscv
uint64_t x;__asm__ volatile("csrr %0,mcycle":"=r"(x)::"memory");return x;
#else
return 0;
#endif
}
extern const int32_t input0[];extern const int8_t expected0[];static int8_t output0[50176+2048] __attribute__((aligned(64)));
static int8_t original0(int32_t x){volatile float value=(float)x;value=value*0x1.3852e80000000p-3f;value=value*0x1.2aa6a40000000p-10f;value=value*0x1.0f0b580000000p+2f;float q=nearbyintf(value);if(q<0)q=0;if(q>127)q=127;return (int8_t)q;}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t control_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t control_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=control_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<control_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void control_0(const int32_t*acc,int8_t*out,size_t count){
 for(size_t i=0;i<count;i++)out[i]=control_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t saturation_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 if((int64_t)x<680LL)return (int8_t)0;
 if((int64_t)x>=171922LL)return (int8_t)127;
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation_0(const int32_t*acc,int8_t*out,size_t count){
 for(size_t i=0;i<count;i++)out[i]=saturation_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet2_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t packet2_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet2_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet2_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet2_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=2;i+=2){
 out[i+0]=packet2_0_scalar(acc[i+0]);
 out[i+1]=packet2_0_scalar(acc[i+1]);
 }
 for(;i<count;i++)out[i]=packet2_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet4_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t packet4_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet4_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet4_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet4_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=4;i+=4){
 out[i+0]=packet4_0_scalar(acc[i+0]);
 out[i+1]=packet4_0_scalar(acc[i+1]);
 out[i+2]=packet4_0_scalar(acc[i+2]);
 out[i+3]=packet4_0_scalar(acc[i+3]);
 }
 for(;i<count;i++)out[i]=packet4_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet8_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t packet8_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet8_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet8_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet8_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=packet8_0_scalar(acc[i+0]);
 out[i+1]=packet8_0_scalar(acc[i+1]);
 out[i+2]=packet8_0_scalar(acc[i+2]);
 out[i+3]=packet8_0_scalar(acc[i+3]);
 out[i+4]=packet8_0_scalar(acc[i+4]);
 out[i+5]=packet8_0_scalar(acc[i+5]);
 out[i+6]=packet8_0_scalar(acc[i+6]);
 out[i+7]=packet8_0_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=packet8_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation2_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t saturation2_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 if((int64_t)x<680LL)return (int8_t)0;
 if((int64_t)x>=171922LL)return (int8_t)127;
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation2_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation2_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation2_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=2;i+=2){
 out[i+0]=saturation2_0_scalar(acc[i+0]);
 out[i+1]=saturation2_0_scalar(acc[i+1]);
 }
 for(;i<count;i++)out[i]=saturation2_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation4_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t saturation4_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 if((int64_t)x<680LL)return (int8_t)0;
 if((int64_t)x>=171922LL)return (int8_t)127;
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation4_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation4_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation4_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=4;i+=4){
 out[i+0]=saturation4_0_scalar(acc[i+0]);
 out[i+1]=saturation4_0_scalar(acc[i+1]);
 out[i+2]=saturation4_0_scalar(acc[i+2]);
 out[i+3]=saturation4_0_scalar(acc[i+3]);
 }
 for(;i<count;i++)out[i]=saturation4_0_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation8_0_thresholds[127]={680LL,2039LL,3398LL,4757LL,6116LL,7475LL,8834LL,10193LL,11553LL,12912LL,14271LL,15630LL,16989LL,18348LL,19707LL,21066LL,22425LL,23784LL,25143LL,26502LL,27861LL,29220LL,30579LL,31939LL,33298LL,34657LL,36016LL,37375LL,38734LL,40093LL,41452LL,42811LL,44170LL,45529LL,46888LL,48247LL,49606LL,50965LL,52325LL,53684LL,55043LL,56402LL,57761LL,59120LL,60479LL,61838LL,63197LL,64556LL,65915LL,67274LL,68633LL,69992LL,71351LL,72711LL,74070LL,75429LL,76788LL,78147LL,79506LL,80865LL,82224LL,83583LL,84942LL,86301LL,87660LL,89019LL,90378LL,91737LL,93097LL,94456LL,95815LL,97174LL,98533LL,99892LL,101251LL,102610LL,103969LL,105328LL,106687LL,108046LL,109405LL,110764LL,112123LL,113482LL,114842LL,116201LL,117560LL,118919LL,120278LL,121637LL,122996LL,124355LL,125714LL,127073LL,128432LL,129791LL,131150LL,132509LL,133869LL,135228LL,136587LL,137946LL,139305LL,140664LL,142023LL,143382LL,144741LL,146100LL,147459LL,148818LL,150177LL,151536LL,152895LL,154254LL,155614LL,156973LL,158332LL,159691LL,161050LL,162409LL,163768LL,165127LL,166486LL,167845LL,169204LL,170563LL,171922LL};
static inline int8_t saturation8_0_scalar(int32_t x){
 if((int64_t)x<-37748736LL || (int64_t)x>37748736LL)__builtin_trap();
 if((int64_t)x<680LL)return (int8_t)0;
 if((int64_t)x>=171922LL)return (int8_t)127;
 int64_t q=((int64_t)x*790059LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation8_0_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation8_0_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation8_0(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=saturation8_0_scalar(acc[i+0]);
 out[i+1]=saturation8_0_scalar(acc[i+1]);
 out[i+2]=saturation8_0_scalar(acc[i+2]);
 out[i+3]=saturation8_0_scalar(acc[i+3]);
 out[i+4]=saturation8_0_scalar(acc[i+4]);
 out[i+5]=saturation8_0_scalar(acc[i+5]);
 out[i+6]=saturation8_0_scalar(acc[i+6]);
 out[i+7]=saturation8_0_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=saturation8_0_scalar(acc[i]);
}
extern const int32_t input1[];extern const int8_t expected1[];static int8_t output1[25088+2048] __attribute__((aligned(64)));
static int8_t original1(int32_t x){volatile float value=(float)x;value=value*0x1.cbc4520000000p+0f;value=value*0x1.b965120000000p-11f;value=value*0x1.ceedee0000000p-2f;float q=nearbyintf(value);if(q<0)q=0;if(q>127)q=127;return (int8_t)q;}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t control_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t control_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=control_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<control_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void control_1(const int32_t*acc,int8_t*out,size_t count){
 for(size_t i=0;i<count;i++)out[i]=control_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t saturation_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 if((int64_t)x<732LL)return (int8_t)0;
 if((int64_t)x>=185064LL)return (int8_t)127;
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation_1(const int32_t*acc,int8_t*out,size_t count){
 for(size_t i=0;i<count;i++)out[i]=saturation_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet2_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t packet2_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet2_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet2_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet2_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=2;i+=2){
 out[i+0]=packet2_1_scalar(acc[i+0]);
 out[i+1]=packet2_1_scalar(acc[i+1]);
 }
 for(;i<count;i++)out[i]=packet2_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet4_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t packet4_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet4_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet4_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet4_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=4;i+=4){
 out[i+0]=packet4_1_scalar(acc[i+0]);
 out[i+1]=packet4_1_scalar(acc[i+1]);
 out[i+2]=packet4_1_scalar(acc[i+2]);
 out[i+3]=packet4_1_scalar(acc[i+3]);
 }
 for(;i<count;i++)out[i]=packet4_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t packet8_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t packet8_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=packet8_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<packet8_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void packet8_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=packet8_1_scalar(acc[i+0]);
 out[i+1]=packet8_1_scalar(acc[i+1]);
 out[i+2]=packet8_1_scalar(acc[i+2]);
 out[i+3]=packet8_1_scalar(acc[i+3]);
 out[i+4]=packet8_1_scalar(acc[i+4]);
 out[i+5]=packet8_1_scalar(acc[i+5]);
 out[i+6]=packet8_1_scalar(acc[i+6]);
 out[i+7]=packet8_1_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=packet8_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation2_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t saturation2_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 if((int64_t)x<732LL)return (int8_t)0;
 if((int64_t)x>=185064LL)return (int8_t)127;
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation2_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation2_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation2_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=2;i+=2){
 out[i+0]=saturation2_1_scalar(acc[i+0]);
 out[i+1]=saturation2_1_scalar(acc[i+1]);
 }
 for(;i<count;i++)out[i]=saturation2_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation4_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t saturation4_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 if((int64_t)x<732LL)return (int8_t)0;
 if((int64_t)x>=185064LL)return (int8_t)127;
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation4_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation4_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation4_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=4;i+=4){
 out[i+0]=saturation4_1_scalar(acc[i+0]);
 out[i+1]=saturation4_1_scalar(acc[i+1]);
 out[i+2]=saturation4_1_scalar(acc[i+2]);
 out[i+3]=saturation4_1_scalar(acc[i+3]);
 }
 for(;i<count;i++)out[i]=saturation4_1_scalar(acc[i]);
}
#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t saturation8_1_thresholds[127]={732LL,2195LL,3658LL,5121LL,6584LL,8047LL,9510LL,10973LL,12436LL,13899LL,15362LL,16824LL,18287LL,19750LL,21213LL,22676LL,24139LL,25602LL,27065LL,28528LL,29991LL,31454LL,32917LL,34380LL,35843LL,37306LL,38769LL,40232LL,41695LL,43158LL,44621LL,46083LL,47546LL,49009LL,50472LL,51935LL,53398LL,54861LL,56324LL,57787LL,59250LL,60713LL,62176LL,63639LL,65102LL,66565LL,68028LL,69491LL,70954LL,72417LL,73880LL,75343LL,76805LL,78268LL,79731LL,81194LL,82657LL,84120LL,85583LL,87046LL,88509LL,89972LL,91435LL,92898LL,94361LL,95824LL,97287LL,98750LL,100213LL,101676LL,103139LL,104602LL,106065LL,107527LL,108990LL,110453LL,111916LL,113379LL,114842LL,116305LL,117768LL,119231LL,120694LL,122157LL,123620LL,125083LL,126546LL,128009LL,129472LL,130935LL,132398LL,133861LL,135324LL,136787LL,138249LL,139712LL,141175LL,142638LL,144101LL,145564LL,147027LL,148490LL,149953LL,151416LL,152879LL,154342LL,155805LL,157268LL,158731LL,160194LL,161657LL,163120LL,164583LL,166046LL,167509LL,168971LL,170434LL,171897LL,173360LL,174823LL,176286LL,177749LL,179212LL,180675LL,182138LL,183601LL,185064LL};
static inline int8_t saturation8_1_scalar(int32_t x){
 if((int64_t)x<-75497472LL || (int64_t)x>75497472LL)__builtin_trap();
 if((int64_t)x<732LL)return (int8_t)0;
 if((int64_t)x>=185064LL)return (int8_t)127;
 int64_t q=((int64_t)x*733955LL+536870912LL)>>30;
 if(q<0)q=0;if(q>127)q=127;
 if(q<127 && (int64_t)x>=saturation8_1_thresholds[q-(0)])q++;
 else if(q>0 && (int64_t)x<saturation8_1_thresholds[q-(0)-1])q--;
 return (int8_t)q;
}
void saturation8_1(const int32_t*acc,int8_t*out,size_t count){
 size_t i=0;
 for(;count-i>=8;i+=8){
 out[i+0]=saturation8_1_scalar(acc[i+0]);
 out[i+1]=saturation8_1_scalar(acc[i+1]);
 out[i+2]=saturation8_1_scalar(acc[i+2]);
 out[i+3]=saturation8_1_scalar(acc[i+3]);
 out[i+4]=saturation8_1_scalar(acc[i+4]);
 out[i+5]=saturation8_1_scalar(acc[i+5]);
 out[i+6]=saturation8_1_scalar(acc[i+6]);
 out[i+7]=saturation8_1_scalar(acc[i+7]);
 }
 for(;i<count;i++)out[i]=saturation8_1_scalar(acc[i]);
}
int main(void){
#ifndef __riscv
for(size_t i=0;i<50176;i++)if(original0(input0[i])!=expected0[i])return 3;
#endif
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();control_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 control %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 control %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();saturation_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 saturation %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 saturation %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();packet2_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 packet2 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 packet2 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();packet4_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 packet4 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 packet4 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();packet8_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 packet8 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 packet8 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();saturation2_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 saturation2 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 saturation2 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();saturation4_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 saturation4 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 saturation4 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<50176+2048;i++)output0[i]=-77;
{uint64_t a=clock_now();saturation8_0(input0,output0,50176);uint64_t b=clock_now();
for(size_t i=0;i<50176;i++)if(output0[i]!=expected0[i]){printf("FAIL 0 saturation8 %lu\n",(unsigned long)i);return 1;}
for(size_t i=50176;i<50176+2048;i++)if(output0[i]!=-77)return 2;
printf("READOUT 0 saturation8 %llu\n",(unsigned long long)(b-a));}

#ifndef __riscv
for(size_t i=0;i<25088;i++)if(original1(input1[i])!=expected1[i])return 3;
#endif
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();control_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 control %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 control %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();saturation_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 saturation %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 saturation %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();packet2_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 packet2 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 packet2 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();packet4_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 packet4 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 packet4 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();packet8_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 packet8 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 packet8 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();saturation2_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 saturation2 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 saturation2 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();saturation4_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 saturation4 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 saturation4 %llu\n",(unsigned long long)(b-a));}
for(size_t i=0;i<25088+2048;i++)output1[i]=-77;
{uint64_t a=clock_now();saturation8_1(input1,output1,25088);uint64_t b=clock_now();
for(size_t i=0;i<25088;i++)if(output1[i]!=expected1[i]){printf("FAIL 1 saturation8 %lu\n",(unsigned long)i);return 1;}
for(size_t i=25088;i<25088+2048;i++)if(output1[i]!=-77)return 2;
printf("READOUT 1 saturation8 %llu\n",(unsigned long long)(b-a));}
printf("READOUT PASS\n");return 0;}
