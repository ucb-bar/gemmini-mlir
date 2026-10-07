#include <stdint.h>
extern void control_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern void cells_prepared_M8(int32_t*,float*,int32_t*,float*,int8_t*);
void cells_M8(int32_t*a,float*sa,int32_t*b,float*sb,int8_t*out){unsigned frm;__asm__ volatile("csrr %0,frm":"=r"(frm)::"memory");if(frm)control_M8(a,sa,b,sb,out);else cells_prepared_M8(a,sa,b,sb,out);}
