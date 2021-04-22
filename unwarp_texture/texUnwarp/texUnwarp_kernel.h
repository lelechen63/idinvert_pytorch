#ifndef tex_unwarp_kernel_h
#define tex_unwarp_kernel_h

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
void test();
void unwarpTexNor_cuda( const std::vector<unsigned char>& img_vec, int imgWidth, int imgHeight,
                        const std::vector<float>& ptUVWs_vec, const std::vector<int>& posIdxs_vec,
                        const std::vector<float>& idxMap_vec, int texWidth, int texHeight );
#ifdef __cplusplus
}
#endif

#endif