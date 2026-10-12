#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_rot_in_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&);
void exl3_rot_in2_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&, at::Tensor&);
void exl3_linear_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                      const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&, at::Tensor&,
                      int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3_linear_pair_cuda(const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                           const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&,
                           at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, const at::Tensor&,
                           const c10::optional<at::Tensor>&, at::Tensor&, const c10::optional<at::Tensor>&,
                           at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3_unpack_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// an output: 2-d, rows contiguous, any row stride (a column range of a fused projection's buffer)
static void check_out(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1 &&
                    (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat),
                name, ": expected a 2-d row-contiguous fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

static void check_io(const at::Tensor& x, const char* name) {
    const auto t = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (t == at::kHalf || t == at::kBFloat16 || t == at::kFloat),
                name, ": expected a contiguous 2-d fp16, bf16 or fp32 CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, name, ": must be 16-byte aligned");
}

// xh [M, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)); x [M, K] fp16, bf16 or fp32.
void rot_in(const at::Tensor& x, const at::Tensor& suh, at::Tensor xh) {
    check_io(x, "x");
    check(suh, at::kHalf, "suh");
    check(xh, at::kHalf, "xh");
    TORCH_CHECK(x.size(1) % 128 == 0 && suh.numel() == x.size(1) && xh.sizes() == x.sizes(),
                "x and xh must be [M, K], K a multiple of 128, suh [K]");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in_cuda(x, suh, xh);
}

// Two rotations of the same rows into two xh buffers, one launch (a fused projection's two parts).
void rot_in2(const at::Tensor& x, const at::Tensor& suh1, at::Tensor xh1, const at::Tensor& suh2, at::Tensor xh2) {
    check_io(x, "x");
    check(suh1, at::kHalf, "suh1");
    check(suh2, at::kHalf, "suh2");
    check(xh1, at::kHalf, "xh1");
    check(xh2, at::kHalf, "xh2");
    TORCH_CHECK(suh1.numel() == x.size(1) && suh2.numel() == x.size(1) && xh1.sizes() == x.sizes()
                && xh2.sizes() == x.sizes(), "rot_in2: both parts must be [M, K] with a [K] suh");
    c10::cuda::CUDAGuard guard(x.device());
    exl3_rot_in2_cuda(x, suh1, xh1, suh2, xh2);
}

// y [M, N] = (xh @ W_q) @ H * svh + bias; Z [SK, M, N] fp32 when SK > 1; counters int32 [8 * N / 128], left zero.
void linear(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb, const at::Tensor& svh,
            const c10::optional<at::Tensor>& bias, at::Tensor y, const c10::optional<at::Tensor>& Z,
            at::Tensor counters, int64_t K2, int64_t cb, int64_t SK, int64_t WK, int64_t mode) {
    TORCH_CHECK(mode == 0 || mode == 1 || mode == 3 || mode == 7, "linear mode must be 0, 1, 3 or 7");
    check(xh, at::kHalf, "xh");
    check_out(y, "y");
    check(svh, at::kHalf, "svh");
    check(T, at::kInt, "T");
    check(counters, at::kInt, "counters");
    const int64_t M = xh.size(0), K = xh.size(1), N = y.size(1);
    TORCH_CHECK(xh.dim() == 2 && y.size(0) == M && M >= 1 && M <= 128, "xh and y must have the same 1 to 128 rows");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    TORCH_CHECK(svh.numel() == N, "svh must have N elements");
    TORCH_CHECK(T.numel() == K * N * K2 / 64, "T must hold K * N * bits / 32 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(T.data_ptr()) % 16 == 0, "T must be 16-byte aligned");
    TORCH_CHECK(counters.numel() >= 8 * (N / 128), "counters must hold 8 * N / 128 ints");
    if (bias) check(*bias, at::kHalf, "bias");
    if (SK > 1) {
        TORCH_CHECK(Z.has_value(), "Z is needed with more than one split");
        check(*Z, at::kFloat, "Z");
        TORCH_CHECK(Z->numel() >= SK * M * N, "Z too small");
    }
    c10::cuda::CUDAGuard guard(xh.device());
    exl3_linear_cuda(xh, T, stride_k, stride_nb, svh, bias, y, Z, counters, K2, cb, SK, WK, mode);
}

// Two projections of one input, the same width and codebook, in a single launch (plan K1-b). Each part keeps its
// own K, scales, bias, output and split-K scratch; the two parts' K splits and outputs are independent.
void linear2(const at::Tensor& xh1, const at::Tensor& T1, int64_t stride_k1, int64_t stride_nb1,
             const at::Tensor& svh1, const c10::optional<at::Tensor>& bias1, at::Tensor y1,
             const c10::optional<at::Tensor>& Z1, at::Tensor counters1, const at::Tensor& xh2, const at::Tensor& T2,
             int64_t stride_k2, int64_t stride_nb2, const at::Tensor& svh2, const c10::optional<at::Tensor>& bias2,
             at::Tensor y2, const c10::optional<at::Tensor>& Z2, at::Tensor counters2, int64_t K2, int64_t cb,
             int64_t SK1, int64_t SK2, int64_t WK, int64_t mode) {
    TORCH_CHECK(mode == 0 || mode == 1 || mode == 3 || mode == 7, "linear mode must be 0, 1, 3 or 7");
    check(xh1, at::kHalf, "xh1");
    check(xh2, at::kHalf, "xh2");
    check_out(y1, "y1");
    check_out(y2, "y2");
    check(svh1, at::kHalf, "svh1");
    check(svh2, at::kHalf, "svh2");
    check(T1, at::kInt, "T1");
    check(T2, at::kInt, "T2");
    check(counters1, at::kInt, "counters1");
    check(counters2, at::kInt, "counters2");
    const int64_t M = xh1.size(0), K = xh1.size(1), N1 = y1.size(1), N2 = y2.size(1);
    TORCH_CHECK(xh2.size(0) == M && xh2.size(1) == K, "a pair launch needs both parts on the same input rows and K");
    TORCH_CHECK(xh1.dim() == 2 && y1.size(0) == M && M >= 1 && M <= 128, "xh and y must have the same 1 to 128 rows");
    TORCH_CHECK(K % 128 == 0 && N1 % 128 == 0 && N2 % 128 == 0, "K and both N must be multiples of 128");
    TORCH_CHECK(svh1.numel() == N1 && svh2.numel() == N2, "each svh must have its part's N elements");
    TORCH_CHECK(T1.numel() == K * N1 * K2 / 64 && T2.numel() == K * N2 * K2 / 64,
                "each T must hold K * N * bits / 32 words");
    TORCH_CHECK(counters1.numel() >= 8 * (N1 / 128) && counters2.numel() >= 8 * (N2 / 128),
                "each counters must hold 8 * N / 128 ints");
    if (bias1) check(*bias1, at::kHalf, "bias1");
    if (bias2) check(*bias2, at::kHalf, "bias2");
    if (SK1 > 1) {
        TORCH_CHECK(Z1.has_value(), "Z1 is needed with more than one split");
        check(*Z1, at::kFloat, "Z1");
        TORCH_CHECK(Z1->numel() >= SK1 * M * N1, "Z1 too small");
    }
    if (SK2 > 1) {
        TORCH_CHECK(Z2.has_value(), "Z2 is needed with more than one split");
        check(*Z2, at::kFloat, "Z2");
        TORCH_CHECK(Z2->numel() >= SK2 * M * N2, "Z2 too small");
    }
    c10::cuda::CUDAGuard guard(xh1.device());
    exl3_linear_pair_cuda(xh1, T1, stride_k1, stride_nb1, svh1, bias1, y1, Z1, counters1, xh2, T2, stride_k2,
                          stride_nb2, svh2, bias2, y2, Z2, counters2, K2, cb, SK1, SK2, WK, mode);
}

// W [K, N] fp16 = W_q, the trellis tiles decoded; tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb words.
void unpack(const at::Tensor& T, at::Tensor W, int64_t stride_k, int64_t stride_nb, int64_t K2, int64_t cb) {
    check(T, at::kInt, "T");
    check(W, at::kHalf, "W");
    TORCH_CHECK(W.dim() == 2 && W.size(0) % 128 == 0 && W.size(1) % 128 == 0, "W must be [K, N], multiples of 128");
    TORCH_CHECK(T.numel() == W.numel() * K2 / 64, "T must hold K * N * bits / 32 words");
    c10::cuda::CUDAGuard guard(T.device());
    exl3_unpack_cuda(T, W, stride_k, stride_nb, K2, cb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rot_in", &rot_in);
    m.def("rot_in2", &rot_in2);
    m.def("linear", &linear, pybind11::arg("xh"), pybind11::arg("T"), pybind11::arg("stride_k"), pybind11::arg("stride_nb"), pybind11::arg("svh"), pybind11::arg("bias"), pybind11::arg("y"), pybind11::arg("Z"), pybind11::arg("counters"), pybind11::arg("K2"), pybind11::arg("cb"), pybind11::arg("SK"), pybind11::arg("WK"), pybind11::arg("mode") = 0);
    m.def("linear2", &linear2, pybind11::arg("xh1"), pybind11::arg("T1"), pybind11::arg("stride_k1"), pybind11::arg("stride_nb1"), pybind11::arg("svh1"), pybind11::arg("bias1"), pybind11::arg("y1"), pybind11::arg("Z1"), pybind11::arg("counters1"), pybind11::arg("xh2"), pybind11::arg("T2"), pybind11::arg("stride_k2"), pybind11::arg("stride_nb2"), pybind11::arg("svh2"), pybind11::arg("bias2"), pybind11::arg("y2"), pybind11::arg("Z2"), pybind11::arg("counters2"), pybind11::arg("K2"), pybind11::arg("cb"), pybind11::arg("SK1"), pybind11::arg("SK2"), pybind11::arg("WK"), pybind11::arg("mode") = 0);
    m.def("unpack", &unpack);
}
