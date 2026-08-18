# ANE native projection sweep v8

This branch adds a diagnostic-only device probe. It does not change production generation behavior.

V7's standalone layout failure was invalid as a correctness verdict: its 37x64 `A12ANESurface` was only 8192 bytes on an A12 device with 16384-byte VM pages, so `A12ANESurface` rejected the fixture before either layout kernel ran.

V8 first re-tests layout with page-valid production shapes and a page-valid padded small shape. It then validates all native projection families in all 28 DiT blocks against the native pack's independent `U8 * FP32 scale + FP32 bias` per-row contract. The projection sweep writes and reads the plane-major IOSurface buffers directly, deliberately bypassing the layout kernels. Fused QKV is checked as three separate outputs, and the rectangular cross K/V plus both MLP shapes are included.

If layout and all projection families pass, the next diagnostic boundary is the real hybrid block glue: RMSNorm/RoPE/attention/gates/residuals at self, cross, and MLP boundaries.
