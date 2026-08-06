// Minimal device kernel for fatbin multi-arch packaging demo.
// Host main is intentionally omitted — we only need the device image container.

extern "C" __global__ void add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
