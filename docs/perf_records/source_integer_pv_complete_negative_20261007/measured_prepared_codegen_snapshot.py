"""Explicit immutable-column preparation for the exact integer observer.

This alternative owns only the generated executor's coefficient schedule. It
grants no typed source-use closure, provider admission or automatic routing.
The unchanged source replay remains mandatory for ambiguous integer bins.
"""

from .scaled_integer_observer_codegen import ScaledIntegerObserverPlan, emit_scaled_integer_observer


def _replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError("owned exact-observer emission grammar changed")
    return source.replace(before, after, 1)


def emit_prepared_scaled_integer_observer(plan: ScaledIntegerObserverPlan, *, symbol: str) -> str:
    """Prepare exact scale products and conservative column radii once.

    Positive finite binary32 alpha and beta have at most 48 product-significand
    bits; their exact product's exponent fits normal binary64. Thus multiplying
    these source constants together in binary64 changes no reconstructed real
    dot. The source's two rounded binary32 multiplies are still materialized
    separately and included in the representation-error summary.

    Distributing the upward source error coefficient gives
    norm(P)*(max(V error)+gamma*max(abs(source V)))+eta. Its preparation and
    application are outward. A maximum row norm proves every source prefix
    finite once per immutable column; unsafe columns use source replay.

    For nonzero signed-i64 sums, adjacent binary64 conversion endpoints times
    the admitted binary32 power of two are exact normal binary64 values: their
    magnitude lies between 2**-149 and 2**191. Only the final exact scale-product
    multiplication requires outward rounding. Integer zero is an exact point.
    Every final bound and original integer observer remains checked. Source
    arithmetic order, source flags permission and full writer contracts are
    inherited from the explicit plan; no target capability is implied.
    """
    result = emit_scaled_integer_observer(plan, symbol=symbol)
    result = _replace_once(
        result,
        " double lhs_mass[H*M],rhs_abs[H*N],rhs_error[H*N];int8_t private_output[H*M*N];",
        " double lhs_mass[H*M],rhs_abs[H*N],rhs_error[H*N];int8_t private_output[H*M*N];\n"
        " double rhs_scale[H*N],rhs_radius[H*N];uint8_t prefix_safe[H*N];",
    )
    entry = f"int {symbol}(const float*p,const int32_t*v,const float*beta,int8_t*out,void*workspace,size_t capacity,product_callback product,void*opaque,observer_counts*counts){{"
    prepare = r"""static void prepare_coefficients(observer_workspace*w,const float*beta){
 const double gamma=merlin_fma_next_up((2*K*0x1p-24)/(1-2*K*0x1p-24));
 const double tiny=merlin_fma_next_up((2*K*0x1p-150)/(1-2*K*0x1p-24));
 const double reciprocal=merlin_fma_next_up(1/(1-gamma));
 for(size_t h=0;h<H;h++){
  double maximum=0;for(size_t m=0;m<M;m++)if(w->lhs_mass[h*M+m]>maximum)maximum=w->lhs_mass[h*M+m];
  for(size_t n=0;n<N;n++){
   size_t j=h*N+n;
   /* The exact product of two finite positive binary32 constants fits f64. */
   w->rhs_scale[j]=(double)@ALPHA@*(double)beta[j];
   w->rhs_radius[j]=merlin_fma_up_add(w->rhs_error[j],merlin_fma_up_mul(gamma,w->rhs_abs[j]));
   double mass=merlin_fma_up_mul(maximum,w->rhs_abs[j]);
   double bound=merlin_fma_up_add(merlin_fma_up_mul(reciprocal,mass),tiny);
   w->prefix_safe[j]=(uint8_t)(finite_f64(bound)&&bound<=FLT_MAX);
  }
 }
}
"""
    # The validated original plan owns the exact literal spelling. Reuse it
    # rather than converting source constants through decimal representations.
    alpha_line = next(line for line in result.splitlines() if "float first=(float)code*" in line)
    alpha = alpha_line.split("float first=(float)code*", 1)[1].split(";", 1)[0]
    prepare = prepare.replace("@ALPHA@", alpha)
    result = _replace_once(result, entry, prepare + entry)
    result = _replace_once(
        result,
        " if(!prepare_lattice(w,p)||!prepare_rhs(w,v,beta))return 0;",
        " if(!prepare_lattice(w,p)||!prepare_rhs(w,v,beta))return 0;\n prepare_coefficients(w,beta);",
    )
    result = _replace_once(result, _ORIGINAL_FINISH.replace("@ALPHA@", alpha), _PREPARED_FINISH)
    return result


_ORIGINAL_FINISH = r"""   if(w->admitted[row]){
    double middle=(double)w->dot[i],lo=merlin_fma_next_down(middle),hi=merlin_fma_next_up(middle);
    double scales[3]={(double)power2(w->exponents[row]),(double)@ALPHA@,(double)beta[h*N+n]};
    for(unsigned s=0;s<3;s++){middle*=scales[s];lo=merlin_fma_next_down(lo*scales[s]);hi=merlin_fma_next_up(hi*scales[s]);}
    double mass=merlin_fma_up_mul(w->lhs_mass[row],w->rhs_abs[h*N+n]);
    double error=merlin_fma_up_add(merlin_fma_up_mul(w->lhs_mass[row],w->rhs_error[h*N+n]),merlin_fma_up_mul(gamma,mass));
    error=merlin_fma_up_add(error,tiny);
    if(merlin_fma_next_up(mass/(1-gamma))<=FLT_MAX){
     float low=merlin_fma_down_cast_f32(merlin_fma_next_down(lo-error)),high=merlin_fma_up_cast_f32(merlin_fma_next_up(hi+error));
     if(finite_f32(low)&&finite_f32(high)&&finite_f64(error)){
      int8_t a=observe(low),b=observe(high);if(a==b){observed=a;certified=1;}
     }
    }
   }
"""

_PREPARED_FINISH = r"""   if(w->admitted[row]&&w->prefix_safe[h*N+n]){
    double lo=0,hi=0;
    if(w->dot[i]){
     double middle=(double)w->dot[i],scale=(double)power2(w->exponents[row]);
     lo=merlin_fma_next_down(middle)*scale;hi=merlin_fma_next_up(middle)*scale;
     lo=-merlin_fma_up_mul(-lo,w->rhs_scale[h*N+n]);hi=merlin_fma_up_mul(hi,w->rhs_scale[h*N+n]);
    }
    double error=merlin_fma_up_add(merlin_fma_up_mul(w->lhs_mass[row],w->rhs_radius[h*N+n]),tiny);
    float low=merlin_fma_down_cast_f32(merlin_fma_down_add(lo,-error)),high=merlin_fma_up_cast_f32(merlin_fma_up_add(hi,error));
    if(finite_f32(low)&&finite_f32(high)&&finite_f64(error)){
     int8_t a=observe(low),b=observe(high);if(a==b){observed=a;certified=1;}
    }
   }
"""
