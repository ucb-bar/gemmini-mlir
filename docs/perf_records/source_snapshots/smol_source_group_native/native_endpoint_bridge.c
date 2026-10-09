#include <stdint.h>
#include <stdlib.h>
typedef struct { void *allocated,*aligned; int64_t offset,size[4],stride[4]; } desc4;
typedef int (*callback_type)(void **);
static callback_type callback;
void native_endpoint_register(callback_type value) { callback=value; }
void _mlir_ciface_native_bf16_endpoint_ab831e3044d54744_borrowed(desc4 *a0,desc4 *a1,desc4 *a2,desc4 *a3,desc4 *a4,desc4 *a5,desc4 *a6,desc4 *a7,desc4 *a8,desc4 *a9,desc4 *a10,desc4 *a11) {
 void *descriptors[12]={a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11};
 if(!callback || callback(descriptors)) abort();
}
