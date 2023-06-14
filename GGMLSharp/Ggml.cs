using GGMLSharp;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace GGMLSharp;

public static unsafe class Ggml
{
    private static ggml_state* g_state = null;
    private static volatile int g_state_barrier = 0;
    private static Stopwatch timer;
    private static bool is_first_call = true;

    const int GGML_MEM_ALIGN = 16;
    const int GGML_MAX_DIMS = 4;
    const int GGML_MAX_NODES = 4096;
    const int GGML_MAX_PARAMS = 16;
    const int GGML_MAX_CONTEXTS = 64;
    const int GGML_MAX_OPT = 4;
    const int GGML_DEFAULT_N_THREADS = 4;

    const int GGML_LOCK_INITIALIZER = 0;

    const int GGML_SOFT_MAX_UNROLL = 4;
    const int GGML_VEC_DOT_UNROLL = 2;

    const int CACHE_LINE_SIZE = 64;
    const int CLOCKS_PER_SEC = 1000;

    const int QK4_0 = 32;
    const int QK4_1 = 32;
    const int QK4_2 = 16;
    const int QK4_3 = 16;
    const int QK5_0 = 32;
    const int QK5_1 = 32;
    const int QK8_0 = 32;
    const int QK8_1 = 32;
    private static ulong GGML_OBJECT_SIZE = (ulong)sizeof(ggml_object);

    // precomputed gelu table for f16 (128 KB)
    private static ushort[] table_gelu_f16 = new ushort[1 << 16];

    // precomputed silu table for f16 (128 KB)
    private static ushort[] table_silu_f16 = new ushort[1 << 16];

    // precomputed exp table for f16 (128 KB)
    private static ushort[] table_exp_f16 = new ushort[1 << 16];

    // precomputed f32 table for f16 (256 KB)
    private static float[] table_f32_f16 = new float[1 << 16];

    private static int[] GGML_BLCK_SIZE = new int[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_F32]  = */1,
        /*[GGML_TYPE_F16]  = */1,
        /*[GGML_TYPE_Q4_0] = */QK4_0,
        /*[GGML_TYPE_Q4_1] = */QK4_1,
        /*[GGML_TYPE_Q4_2] = */QK4_2,
        /*[GGML_TYPE_Q4_3] = */QK4_3,
        /*[GGML_TYPE_Q5_0] = */QK5_0,
        /*[GGML_TYPE_Q5_1] = */QK5_1,
        /*[GGML_TYPE_Q8_0] = */QK8_0,
        /*[GGML_TYPE_Q8_1] = */QK8_1,
        /*[GGML_TYPE_I8]   = */1,
        /*[GGML_TYPE_I16]  = */1,
        /*[GGML_TYPE_I32]  = */1,
    };

    private static ulong[] GGML_TYPE_SIZE = new ulong[(int)ggml_type.GGML_TYPE_COUNT] {
        /* [GGML_TYPE_F32]  = */ sizeof(float),
        /* [GGML_TYPE_F16]  = */ sizeof(short)                                  /*sizeof(ggml_fp16_t)*/,
        /* [GGML_TYPE_Q4_0] = */ sizeof(float) + QK4_0 / 2                      /*sizeof(block_q4_0) */,
        /* [GGML_TYPE_Q4_1] = */ 2 * sizeof(float) + QK4_1 / 2                  /*sizeof(block_q4_1)*/,
        /* [GGML_TYPE_Q4_2] = */ sizeof(short) + QK4_2 / 2                      /*sizeof(block_q4_2)*/,
        /* [GGML_TYPE_Q4_3] = */ 2 * sizeof(short) + QK4_3 / 2                  /*sizeof(block_q4_3)*/,
        /* [GGML_TYPE_Q5_0] = */ sizeof(short) + sizeof(uint) + QK5_0 / 2       /*sizeof(block_q5_0)*/,
        /* [GGML_TYPE_Q5_1] = */ 2 * sizeof(short) + sizeof(uint) + QK5_1 / 2   /*sizeof(block_q5_1)*/,
        /* [GGML_TYPE_Q8_0] = */ sizeof(float) + QK8_0                          /*sizeof(block_q8_0)*/,
        /* [GGML_TYPE_Q8_1] = */ 3*sizeof(float) + QK8_1                        /*sizeof(block_q8_1)*/,
        /* [GGML_TYPE_I8]   = */ sizeof(byte),
        /* [GGML_TYPE_I16]  = */ sizeof(short),
        /* [GGML_TYPE_I32]  = */ sizeof(int),
    };

    private static string[] GGML_TYPE_NAME = new string[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_F32]  = */ "f32",
        /*[GGML_TYPE_F16]  = */ "f16",
        /*[GGML_TYPE_Q4_0] = */ "q4_0",
        /*[GGML_TYPE_Q4_1] = */ "q4_1",
        /*[GGML_TYPE_Q4_2] = */ "q4_2",
        /*[GGML_TYPE_Q4_3] = */ "q4_3",
        /*[GGML_TYPE_Q5_0] = */ "q5_0",
        /*[GGML_TYPE_Q5_1] = */ "q5_1",
        /*[GGML_TYPE_Q8_0] = */ "q8_0",
        /*[GGML_TYPE_Q8_1] = */ "q8_1",
        /*[GGML_TYPE_I8]   = */ "i8",
        /*[GGML_TYPE_I16]  = */ "i16",
        /*[GGML_TYPE_I32]  = */ "i32",
    };

    private static string[] GGML_OP_SYMBOL = new string[(int)ggml_op.GGML_OP_COUNT] {
        "none",

        "x",
        "x+y",
        "x-y",
        "x*y",
        "x/y",
        "x^2",
        "√x",
        "Σx",
        "Σx/n",
        "repeat(x)",
        "abs(x)",
        "sgn(x)",
        "-x",
        "step(x)",
        "relu(x)",
        "gelu(x)",
        "silu(x)",
        "norm(x)",
        "rms_norm(x)",

        "X*Y",

        "x*v",
        "x-\\>y",
        "cont(x)",
        "reshape(x)",
        "view(x)",
        "permute(x)",
        "transpose(x)",
        "get_rows(x)",
        "diag_mask_inf(x)",
        "soft_max(x)",
        "rope(x)",
        "alibi(x)",
        "conv_1d_1s(x)",
        "conv_1d_2s(x)",

        "flash_attn(x)",
        "flash_ff(x)",

        "f(x)",
        "f(x,y)",
    };

    static bool[] GGML_IS_QUANTIZED = new bool[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_F32]  = */ false,
        /*[GGML_TYPE_F16]  = */ false,
        /*[GGML_TYPE_Q4_0] = */ true,
        /*[GGML_TYPE_Q4_1] = */ true,
        /*[GGML_TYPE_Q4_2] = */ true,
        /*[GGML_TYPE_Q4_3] = */ true,
        /*[GGML_TYPE_Q5_0] = */ true,
        /*[GGML_TYPE_Q5_1] = */ true,
        /*[GGML_TYPE_Q8_0] = */ true,
        /*[GGML_TYPE_Q8_1] = */ true,
        /*[GGML_TYPE_I8]   = */ false,
        /*[GGML_TYPE_I16]  = */ false,
        /*[GGML_TYPE_I32]  = */ false,
    };

    static quantize_fns_t[] quantize_fns = new quantize_fns_t[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_Q4_0] =*/ new quantize_fns_t
        {
            dequantize_row_q         = &dequantize_row_q4_0,
            quantize_row_q           = &quantize_row_q4_0,
            quantize_row_q_reference = &quantize_row_q4_0_reference,
            quantize_row_q_dot       = &quantize_row_q8_0,
            vec_dot_q                = &ggml_vec_dot_q4_0_q8_0,
            vec_dot_type             = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q4_1] =*/ new quantize_fns_t{
            dequantize_row_q         = &dequantize_row_q4_1,
            quantize_row_q           = &quantize_row_q4_1,
            quantize_row_q_reference = &quantize_row_q4_1_reference,
            quantize_row_q_dot       = &quantize_row_q8_1,
            vec_dot_q                = &ggml_vec_dot_q4_1_q8_1,
            vec_dot_type             = ggml_type.GGML_TYPE_Q8_1,
        },
        /*[GGML_TYPE_Q4_2] = */ new quantize_fns_t {
            dequantize_row_q = &dequantize_row_q4_2,
            quantize_row_q = &quantize_row_q4_2,
            quantize_row_q_reference = &quantize_row_q4_2_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q4_2_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q4_3] = */ default,
        /*[GGML_TYPE_Q5_0] = */ new quantize_fns_t {
            dequantize_row_q = &dequantize_row_q5_0,
            quantize_row_q = &quantize_row_q5_0,
            quantize_row_q_reference = &quantize_row_q5_0_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q5_0_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q5_1] = */ new quantize_fns_t{
            dequantize_row_q = &dequantize_row_q5_1,
            quantize_row_q = &quantize_row_q5_1,
            quantize_row_q_reference = &quantize_row_q5_1_reference,
            quantize_row_q_dot = &quantize_row_q8_1,
            vec_dot_q = &ggml_vec_dot_q5_1_q8_1,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_1,
        },
        /*[GGML_TYPE_Q8_0] = */ new quantize_fns_t{
            dequantize_row_q = &dequantize_row_q8_0,
            quantize_row_q = &quantize_row_q8_0,
            quantize_row_q_reference = &quantize_row_q8_0_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q8_0_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q8_1] = */ new quantize_fns_t{
            dequantize_row_q = null,   // TODO
            quantize_row_q = &quantize_row_q8_1,
            quantize_row_q_reference = &quantize_row_q8_1_reference,
            quantize_row_q_dot = &quantize_row_q8_1,
            vec_dot_q = null,   // TODO
            vec_dot_type = ggml_type.GGML_TYPE_Q8_1,
        },
        default,
        default,
        default,
        default,
        default,
    };

    // Unpack 32 4-bit fields into 32 bytes
    // The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
    static Vector256<byte> bytes_from_nibbles_32(byte* rsi)
    {
        // Load 16 bytes from memory
        var tmp = Vector128.Load(rsi);

        // Expand bytes into uint16_t values
        var bytes = Avx2.ConvertToVector256Int16(tmp);

        // Unpack values into individual bytes
        var lowMask = Vector256.Create<byte>(0xF).AsInt16();
        var high = Avx2.AndNot(lowMask, bytes);
        var low = Avx2.And(lowMask, bytes);
        high = Avx2.ShiftLeftLogical(high, 4);
        bytes = Avx2.Or(low, high);
        return bytes.AsByte();
    }

    private static Vector128<byte> packNibbles(Vector128<short> bytes1, Vector128<short> bytes2)
    {
        // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
        var lowByte = Vector128.Create<short>(0xFF);
        var high = Vector128.AndNot(lowByte, bytes1);
        var low = Vector128.AndNot(lowByte, bytes1);
        high = Vector128.ShiftRightLogical(high, 4);
        bytes1 = Sse2.Or(low, high);
        high = Vector128.AndNot(lowByte, bytes2);
        low = Sse2.And(lowByte, bytes2);
        high = Vector128.ShiftRightLogical(high, 4);
        bytes2 = Sse2.Or(low, high);

        return Sse2.PackUnsignedSaturate(bytes1, bytes2);
    }

    // reference implementation for deterministic creation of model files
    [UnmanagedCallersOnly]
    static void quantize_row_q4_0_reference(float* x, void* y, int k)
    {
        quantize_row_q4_0_reference_impl(x, (block_q4_0*)y, k);
    }

    static void quantize_row_q4_0_reference_impl(float* x, block_q4_0* y, int k)
    {
        Debug.Assert(k % QK4_0 == 0);
        int nb = k / QK4_0;

        byte* pp = stackalloc byte[QK4_0/2];

        for (int i = 0; i < nb; i++) {
            float amax = 0.0f; // absolute max
            float max = 0.0f;

            for (int l = 0; l < QK4_0; l++) {
                float v = x[i*QK4_0 + l];
                if (amax < Math.Abs(v)) {
                    amax = Math.Abs(v);
                    max = v;
                }
            }

            float d = max / -8;
            float id = d != 0.0f ? 1.0f/d : 0.0f;

            y[i].d = d;

            for (int l = 0; l < QK4_0; l += 2) {
                float v0 = x[i*QK4_0 + l + 0]*id;
                float v1 = x[i*QK4_0 + l + 1]*id;

                byte vi0 = (byte)Math.Min(15, Math.Round(v0) + 8);
                byte vi1 = (byte)Math.Min(15, Math.Round(v1) + 8);

                Debug.Assert(vi0 < 16);
                Debug.Assert(vi1 < 16);

                pp[l/2] = (byte)(vi0 | (vi1 << 4));
            }

            Buffer.MemoryCopy(pp, y[i].qs, QK4_0 / 2, QK4_0 / 2);
        }
    }

    [UnmanagedCallersOnly]
    private static void quantize_row_q4_0(float* x, void* vy, int k) {
        Debug.Assert(k % QK4_0 == 0);
        int nb = k / QK4_0;

        block_q4_0* y = (block_q4_0*)vy;
        if (Avx.IsSupported)
        {
            for (int i = 0; i < nb; i++)
            {
                // Load elements into 4 AVX vectors
                var v0 = Vector256.Load(x);
                var v1 = Vector256.Load(x + 8);
                var v2 = Vector256.Load(x + 16);
                var v3 = Vector256.Load(x + 24);
                x += 32;

                // Compute max for the block
                var max = Vector256.Max(v0, v1);
                var maxTmp = Vector256.Max(v2, v3);
                max = Vector256.Max(max, maxTmp);

                var max4 = Vector128.Max(max.GetUpper(), max.GetLower());
                max4 = Vector128.Max(max4, Sse.MoveHighToLow(max4, max4));
                max4 = Sse.MaxScalar(max4, Sse3.MoveHighAndDuplicate(max4));
                float maxScalar = max4.ToScalar();

                // Compute min for the block
                var min = Vector256.Min(v0, v1);
                var minTmp = Vector256.Min(v2, v3);
                min = Vector256.Min(min, minTmp);

                var min4 = Vector128.Min(min.GetUpper(), min.GetLower());
                min4 = Vector128.Min(min4, Sse.MoveHighToLow(min4, min4));
                min4 = Sse.MinScalar(min4, Sse3.MoveHighAndDuplicate(min4));
                float minScalar = min4.ToScalar();

                // Quantize these floats
                float magnitude = maxScalar >= Math.Abs(minScalar) ? maxScalar : minScalar;
                float d = magnitude / -8.0f;
                y[i].d = d;
                float id = (magnitude != 0.0f) ? -8.0f / magnitude : 0.0f;
                var mul = Vector256.Create(id);

                // Apply the multiplier
                v0 = Vector256.Multiply(v0, mul);
                v1 = Vector256.Multiply(v1, mul);
                v2 = Vector256.Multiply(v2, mul);
                v3 = Vector256.Multiply(v3, mul);

                // Round to nearest integer
                v0 = Avx.RoundToNearestInteger(v0);
                v1 = Avx.RoundToNearestInteger(v1);
                v2 = Avx.RoundToNearestInteger(v2);
                v3 = Avx.RoundToNearestInteger(v3);

                // Convert floats to integers
                var i0 = Avx.ConvertToVector256Int32(v0);
                var i1 = Avx.ConvertToVector256Int32(v1);
                var i2 = Avx.ConvertToVector256Int32(v2);
                var i3 = Avx.ConvertToVector256Int32(v3);

                // Since we don't have in AVX some necessary functions,
                // we split the registers in half and call AVX2 analogs from SSE
                var ni0 = i0.GetLower();
                var ni1 = i0.GetUpper();
                var ni2 = i1.GetLower();
                var ni3 = i1.GetUpper();
                var ni4 = i2.GetLower();
                var ni5 = i2.GetUpper();
                var ni6 = i3.GetLower();
                var ni7 = i3.GetUpper();

                // Convert int32 to int16
                var si0 = Sse2.PackSignedSaturate(ni0, ni1);
                var si2 = Sse2.PackSignedSaturate(ni2, ni3);
                var si4 = Sse2.PackSignedSaturate(ni4, ni5);
                var si6 = Sse2.PackSignedSaturate(ni6, ni7);
                // Convert int16 to int8
                var bi0 = Sse2.PackSignedSaturate(si0, si2).AsByte();
                var bi4 = Sse2.PackSignedSaturate(si4, si6).AsByte();

                // Apply offset and clamp to translate the range from [ -8 .. +8 ] into [ +0 .. +15 ]
                var off = Vector128.Create<byte>(8);
                bi0 = Vector128.Add(bi0, off);
                bi4 = Vector128.Add(bi4, off);
                var maxNibble = Vector128.Create<byte>(15);
                bi0 = Vector128.Min(bi0, maxNibble);
                bi4 = Vector128.Min(bi4, maxNibble);

                // Compress the vector into 4 bit/value, and store
                var res = packNibbles(bi0.AsInt16(), bi4.AsInt16());
                res.Store(y[i].qs);
            }
        }
        else
        {
            quantize_row_q4_0_reference_impl(x, y, k);
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q4_1_reference(float* x, void* y, int k)
    {
        quantize_row_q4_1_reference_impl(x, (block_q4_1*)y, k);
    }

    static void quantize_row_q4_1_reference_impl(float* x, block_q4_1* y, int k)
    {
        Debug.Assert(k % QK4_1 == 0);
        int nb = k / QK4_1;

        byte* pp = stackalloc byte[QK4_1 / 2];

        for (int i = 0; i < nb; i++)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int l = 0; l < QK4_1; l++)
            {
                float v = x[i * QK4_1 + l];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float d = (max - min) / ((1 << 4) - 1);
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = d;
            y[i].m = min;

            for (int l = 0; l < QK4_1; l += 2)
            {
                float v0 = (x[i * QK4_1 + l + 0] - min) * id;
                float v1 = (x[i * QK4_1 + l + 1] - min) * id;

                byte vi0 = (byte)Math.Round(v0);
                byte vi1 = (byte)Math.Round(v1);

                Debug.Assert(vi0 < 16);
                Debug.Assert(vi1 < 16);

                pp[l / 2] = (byte)(vi0 | (vi1 << 4));
            }

            Buffer.MemoryCopy(pp, y[i].qs, QK4_1 / 2, QK4_1 / 2);
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q4_1(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK4_1 == 0);

        int nb = k / QK4_1;

        block_q4_1* y = (block_q4_1*)vy;
        quantize_row_q4_1_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q4_2_reference(float* x, void* y, int k)
    {
        quantize_row_q4_2_reference_impl(x, (block_q4_2*)y, k);
    }

    static void quantize_row_q4_2_reference_impl(float* x, block_q4_2* y, int k)
    {
        Debug.Assert(k % QK4_2 == 0);
        int nb = k / QK4_2;

        byte* pp = stackalloc byte[QK4_2 / 2];

        for (int i = 0; i < nb; i++)
        {
            float amax = 0.0f; // absolute max
            float max = 0.0f;

            for (int l = 0; l < QK4_2; l++)
            {
                float v = x[i * QK4_2 + l];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                    max = v;
                }
            }

            float d = max / -8;
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = (ushort)(Half)d;

            for (int l = 0; l < QK4_2; l += 2)
            {
                float v0 = x[i * QK4_2 + l + 0] * id;
                float v1 = x[i * QK4_2 + l + 1] * id;

                byte vi0 = (byte)Math.Min(15, Math.Round(v0) + 8);
                byte vi1 = (byte)Math.Min(15, Math.Round(v1) + 8);

                Debug.Assert(vi0 < 16);
                Debug.Assert(vi1 < 16);

                pp[l / 2] = (byte)(vi0 | (vi1 << 4));
            }

            Buffer.MemoryCopy(pp, y[i].qs, QK4_2 / 2, QK4_2 / 2);
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q4_2(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK4_2 == 0);

        int nb = k / QK4_2;

        block_q4_2* y = (block_q4_2*)vy;
        quantize_row_q4_2_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q5_0_reference(float* x, void* y, int k)
    {
        quantize_row_q5_0_reference_impl(x, (block_q5_0*)y, k);
    }

    static void quantize_row_q5_0_reference_impl(float* x, block_q5_0* y, int k)
    {
        Debug.Assert(k % QK5_0 == 0);
        int nb = k / QK5_0;

        for (int i = 0; i < nb; i++)
        {
            float amax = 0.0f; // absolute max
            float max = 0.0f;

            for (int l = 0; l < QK5_0; l++)
            {
                float v = x[i * QK5_0 + l];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                    max = v;
                }
            }

            float d = max / -16;
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = (Half)d;

            uint qh = 0;

            for (int l = 0; l < QK5_0; l += 2)
            {
                float v0 = x[i * QK5_0 + l + 0] * id;
                float v1 = x[i * QK5_0 + l + 1] * id;

                uint vi0 = (uint)Math.Min(31, (int)(v0 + 16.5f));
                uint vi1 = (uint)Math.Min(31, (int)(v1 + 16.5f));

                y[i].qs[l / 2] = (byte)((vi0 & 0x0F) | ((vi1 & 0x0F) << 4));

                // get the 5-th bit and store it in qh at the right position
                qh |= ((vi0 & 0x10) >> 4) << (l + 0);
                qh |= ((vi1 & 0x10) >> 4) << (l + 1);
            }

            *(uint*)y[i].qh = qh;
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q5_0(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK5_0 == 0);

        int nb = k / QK5_0;

        block_q5_0* y = (block_q5_0*)vy;
        quantize_row_q5_0_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q5_1_reference(float* x, void* y, int k)
    {
        quantize_row_q5_1_reference_impl(x, (block_q5_1*)y, k);
    }

    static void quantize_row_q5_1_reference_impl(float* x, block_q5_1* y, int k)
    {
        Debug.Assert(k % QK5_1 == 0);
        int nb = k / QK5_1;

        for (int i = 0; i < nb; i++)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int l = 0; l < QK5_1; l++)
            {
                float v = x[i * QK5_1 + l];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float d = (max - min) / ((1 << 5) - 1);
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = (ushort)(Half)d;
            y[i].m = (ushort)(Half)min;

            uint qh = 0;

            for (int l = 0; l < QK5_1; l += 2)
            {
                float v0 = (x[i * QK5_1 + l + 0] - min) * id;
                float v1 = (x[i * QK5_1 + l + 1] - min) * id;

                uint vi0 = (uint)(v0 + 0.5f);
                uint vi1 = (uint)(v1 + 0.5f);

                y[i].qs[l / 2] = (byte)((vi0 & 0x0F) | ((vi1 & 0x0F) << 4));

                // get the 5-th bit and store it in qh at the right position
                qh |= ((vi0 & 0x10) >> 4) << (l + 0);
                qh |= ((vi1 & 0x10) >> 4) << (l + 1);
            }

            *(uint*)y[i].qh = qh;
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q5_1(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK5_1 == 0);

        int nb = k / QK5_1;

        block_q5_1* y = (block_q5_1*)vy;
        quantize_row_q5_1_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q8_0_reference(float* x, void* y, int k)
    {
        quantize_row_q8_0_reference_impl(x, (block_q8_0*)y, k);
    }

    static void quantize_row_q8_0_reference_impl(float* x, block_q8_0* y, int k)
    {
        Debug.Assert(k % QK8_0 == 0);
        int nb = k / QK8_0;

        for (int i = 0; i < nb; i++)
        {
            float amax = 0.0f; // absolute max

            for (int l = 0; l < QK8_0; l++)
            {
                float v = x[i * QK8_0 + l];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                }
            }

            float d = amax / ((1 << 7) - 1);
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = d;

            for (int l = 0; l < QK8_0; l += 2)
            {
                float v0 = x[i * QK8_0 + l] * id;
                y[i].qs[l] = (byte)Math.Round(v0);
            }
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q8_0(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK8_0 == 0);

        int nb = k / QK8_0;

        block_q8_0* y = (block_q8_0*)vy;
        quantize_row_q8_0_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q8_1_reference(float* x, void* y, int k)
    {
        quantize_row_q8_1_reference_impl(x, (block_q8_1*)y, k);
    }

    static void quantize_row_q8_1_reference_impl(float* x, block_q8_1* y, int k)
    {
        Debug.Assert(k % QK8_1 == 0);
        int nb = k / QK8_1;

        for (int i = 0; i < nb; i++)
        {
            float amax = 0.0f; // absolute max
            float max = 0.0f;

            for (int l = 0; l < QK8_1; l++)
            {
                float v = x[i * QK8_1 + l];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                }
            }

            float d = amax / ((1 << 7) - 1);
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = d;

            int sum0 = 0;
            int sum1 = 0;

            for (int l = 0; l < QK8_1; l += 2)
            {
                float v0 = x[i * QK8_1             + l] * id;
                float v1 = x[i * QK8_1 + QK8_1 / 2 + l] * id;

                y[i].qs[l] = (byte)Math.Round(v0);
                y[i].qs[QK8_1 / 2 + l] = (byte)Math.Round(v1);

                sum0 += y[i].qs[l];
                sum1 += y[i].qs[QK8_1 / 2 + l];
            }

            y[i].s0 = d * sum0;
            y[i].s1 = d * sum1;
        }
    }

    [UnmanagedCallersOnly]
    static void quantize_row_q8_1(float* x, void* vy, int k)
    {
        Debug.Assert(k % QK8_1 == 0);

        int nb = k / QK8_1;

        block_q8_1* y = (block_q8_1*)vy;
        quantize_row_q8_1_reference_impl(x, y, k);
    }

    [UnmanagedCallersOnly]
    private unsafe static void dequantize_row_q4_0(void* vx, float* y, int k) {
        Debug.Assert(k % QK4_0 == 0);
        int nb = k / QK4_0;

        block_q4_0* x = (block_q4_0*)vx;

        if (Avx2.IsSupported)
        {
            for (int i = 0; i < nb; i++)
            {
                // scale factor
                Vector256<float> d_v = Vector256.Create(x[i].d);

                byte* pp = x[i].qs;

                for (int l = 0; l < QK4_0; l += 32)
                {
                    // Load 32x4-bit integers into 32x8-bit integers
                    var vx8 = bytes_from_nibbles_32(pp + l / 2);

                    // Subtract 8 from the integers
                    vx8 = Vector256.Subtract(vx8, Vector256.Create<byte>(8));

                    // Convert to 16-bit int
                    var vx16_lo = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vx8, 0));
                    var vx16_hi = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vx8, 1));

                    // Convert to 32-bit int -> float 32
                    var vf = new Vector256<float>[4] {
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_lo, 0))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_lo, 1))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_hi, 0))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_hi, 1)))
                };

                    // Scale and store
                    for (int j = 0; j < 4; j++)
                    {
                        var result = Avx.Multiply(vf[j], d_v);
                        Avx.Store(y + i * QK4_0 + l + j * 8, result);
                    }
                }
            }
        }
        else
        {
            // scalar
            for (int i = 0; i < nb; i++)
            {
                float d = x[i].d;

                byte* pp = x[i].qs;

                for (int l = 0; l < QK4_0; l += 2)
                {
                    byte vi = pp[l / 2];

                    byte vi0 = (byte)(vi & 0x0F);
                    byte vi1 = (byte)(vi >> 4);

                    float v0 = (vi0 - 8) * d;
                    float v1 = (vi1 - 8) * d;

                    //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

                    y[i * QK4_0 + l + 0] = v0;
                    y[i * QK4_0 + l + 1] = v1;

                    Debug.Assert(!float.IsNaN(y[i * QK4_0 + l + 0]));
                    Debug.Assert(!float.IsNaN(y[i * QK4_0 + l + 1]));
                }
            }
        }
    }

    [UnmanagedCallersOnly]
    private unsafe static void dequantize_row_q4_1(void* vx, float* y, int k)
    {
        Debug.Assert(k % QK4_1 == 0);
        int nb = k / QK4_1;

        block_q4_1* x = (block_q4_1*)vx;

        if (Avx2.IsSupported)
        {
            for (int i = 0; i < nb; i++)
            {
                // scale factor
                Vector256<float> d_v = Vector256.Create(x[i].d);
                Vector256<float> d_m = Vector256.Create(x[i].m);

                byte* pp = x[i].qs;

                for (int l = 0; l < QK4_1; l += 32)
                {
                    // Load 32x4-bit integers into 32x8-bit integers
                    var vx8 = bytes_from_nibbles_32(pp + l / 2);

                    // Convert to 16-bit int
                    var vx16_lo = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vx8, 0));
                    var vx16_hi = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vx8, 1));

                    // Convert to 32-bit int -> float 32
                    var vf = new Vector256<float>[4] {
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_lo, 0))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_lo, 1))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_hi, 0))),
                    Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Avx2.ExtractVector128(vx16_hi, 1)))
                };

                    // Scale and store
                    for (int j = 0; j < 4; j++)
                    {
                        var result = Vector256.Add(Vector256.Multiply(vf[j], d_v), d_m);
                        Avx.Store(y + i * QK4_0 + l + j * 8, result);
                    }
                }
            }
        }
        else
        {
            // scalar
            for (int i = 0; i < nb; i++)
            {
                float d = x[i].d;
                float m = x[i].m;

                byte* pp = x[i].qs;

                for (int l = 0; l < QK4_1; l += 2)
                {
                    byte vi = pp[l / 2];

                    byte vi0 = (byte)(vi & 0x0F);
                    byte vi1 = (byte)(vi >> 4);

                    float v0 = vi0 * d + m;
                    float v1 = vi1 * d + m;

                    //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

                    y[i * QK4_1 + l + 0] = v0;
                    y[i * QK4_1 + l + 1] = v1;

                    Debug.Assert(!float.IsNaN(y[i * QK4_1 + l + 0]));
                    Debug.Assert(!float.IsNaN(y[i * QK4_1 + l + 1]));
                }
            }
        }
    }

    [UnmanagedCallersOnly]
    static void dequantize_row_q4_2(void* vx, float* y, int k) {
        Debug.Assert(k % QK4_2 == 0);
        int nb = k / QK4_2;

        block_q4_2* x = (block_q4_2*)vx;

        for (int i = 0; i < nb; i++) {
            float d = (float)(Half)(x[i].d);

            byte* pp = x[i].qs;

            for (int l = 0; l < QK4_2; l += 2) {
                byte vi = pp[l / 2];

                byte vi0 = (byte)(vi & 0x0F);
                byte vi1 = (byte)(vi >> 4);

                float v0 = (vi0 - 8) * d;
                float v1 = (vi1 - 8) * d;

                y[i * QK4_2 + l + 0] = v0;
                y[i * QK4_2 + l + 1] = v1;

                Debug.Assert(!float.IsNaN(y[i * QK4_2 + l + 0]));
                Debug.Assert(!float.IsNaN(y[i * QK4_2 + l + 1]));
            }
        }
    }

    [UnmanagedCallersOnly]
    static void dequantize_row_q5_0(void* vx, float* y, int k) {
        Debug.Assert(k % QK5_0 == 0);
        int nb = k / QK5_0;

        block_q5_0* x = (block_q5_0*)vx;

        for (int i = 0; i<nb; i++) {
            float d = (float)(Half)(x[i].d);

            byte* pp = x[i].qs;

            uint qh = *(uint*)x[i].qh;

            for (int l = 0; l<QK5_0; l += 2) {
                byte vi = pp[l / 2];

                // extract the 5-th bit from qh
                byte vh0 = (byte)(((qh & (1 << (l + 0))) >> (l + 0)) << 4);
                byte vh1 = (byte)(((qh & (1 << (l + 1))) >> (l + 1)) << 4);

                byte vi0 = (byte)((vi & 0x0F) | vh0);
                byte vi1 = (byte)((vi >> 4) | vh1);

                float v0 = (vi0 - 16) * d;
                float v1 = (vi1 - 16) * d;

                y[i * QK5_0 + l + 0] = v0;
                y[i * QK5_0 + l + 1] = v1;

                Debug.Assert(!float.IsNaN(y[i * QK5_0 + l + 0]));
                Debug.Assert(!float.IsNaN(y[i * QK5_0 + l + 1]));
            }
        }
    }

    [UnmanagedCallersOnly]
    static void dequantize_row_q5_1(void* vx, float* y, int k)
    {
        Debug.Assert(k % QK5_1 == 0);
        int nb = k / QK5_1;

        block_q5_1* x = (block_q5_1*)vx;

        for (int i = 0; i<nb; i++)
        {
            float d = (float)(Half)(x[i].d);
            float m = (float)(Half)(x[i].m);

            byte* pp = x[i].qs;

            uint qh = *(uint*)x[i].qh;

            for (int l = 0; l<QK5_1; l += 2)
            {
                byte vi = pp[l / 2];

                // extract the 5-th bit from qh
                byte vh0 = (byte)(((qh & (1 << (l + 0))) >> (l + 0)) << 4);
                byte vh1 = (byte)(((qh & (1 << (l + 1))) >> (l + 1)) << 4);

                byte vi0 = (byte)((vi & 0x0F) | vh0);
                byte vi1 = (byte)((vi >> 4) | vh1);

                float v0 = vi0 * d + m;
                float v1 = vi1 * d + m;

                y[i * QK5_1 + l + 0] = v0;
                y[i * QK5_1 + l + 1] = v1;

                Debug.Assert(!float.IsNaN(y[i * QK5_1 + l + 0]));
                Debug.Assert(!float.IsNaN(y[i * QK5_1 + l + 1]));
            }
        }
    }

    [UnmanagedCallersOnly]
    static void dequantize_row_q8_0(void* vx, float* y, int k)
    {
        Debug.Assert(k % QK8_0 == 0);
        int nb = k / QK8_0;

        block_q8_0* x = (block_q8_0*)vx;

        for (int i = 0; i < nb; i++)
        {
            float d = x[i].d;

            byte* pp = x[i].qs;

            for (int l = 0; l < QK8_0; ++l)
            {
                y[i * QK8_0 + l] = pp[l] * d;
            }
        }
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q4_0_q8_0(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_0;

        Debug.Assert(n % QK8_0 == 0);
        Debug.Assert(nb % 2 == 0);

        block_q4_0* x = (block_q4_0*)vx;
        block_q8_0* y = (block_q8_0*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            float d0 = x[i].d;
            float d1 = y[i].d;

            byte* p0 = x[i].qs;
            byte* p1 = y[i].qs;

            int sumi = 0;
            for (int j = 0; j < QK8_0 / 2; j++)
            {
                byte v0 = p0[j];

                int i0 = (byte)(v0 & 0x0F) - 8;
                int i1 = (byte)(v0 >> 4) - 8;

                int i2 = p1[2 * j + 0];
                int i3 = p1[2 * j + 1];

                sumi += i0 * i2 + i1 * i3;
            }

            sumf += d0 * d1 * sumi;
        }
        *s = sumf;
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q4_1_q8_1(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_1;

        Debug.Assert(n % QK8_1 == 0);
        Debug.Assert(nb % 2 == 0);

        block_q4_1* x = (block_q4_1*)vx;
        block_q8_1* y = (block_q8_1*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            float d0 = x[i].d;
            float m0 = x[i].m;
            float d1 = y[i].d;

            byte* p0 = x[i].qs;
            byte* p1 = y[i].qs;

            // TODO: this is very slow ..
            for (int j = 0; j < QK8_1 / 2; j++)
            {
                byte v0 = p0[j];

                float f0 = d0 * (v0 & 0x0F) + m0;
                float f1 = d0 * (v0 >> 4) + m0;

                float f2 = d1 * p1[2 * j + 0];
                float f3 = d1 * p1[2 * j + 1];

                sumf += f0 * f2 + f1 * f3;
            }
        }
        *s = sumf;
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q4_2_q8_0(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_0;

        Debug.Assert(n % QK8_0 == 0);
        Debug.Assert(nb % 2 == 0);
        Debug.Assert(QK8_0 == 2 * QK4_2);

        block_q4_2* x = (block_q4_2*)vx;
        block_q8_0* y = (block_q8_0*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            byte* x0 = x[2 * i + 0].qs;
            byte* x1 = x[2 * i + 1].qs;
            byte* y0 = y[i].qs;

            float d0 = (float)(Half)x[2 * i + 0].d;
            float d1 = (float)(Half)x[2 * i + 1].d;

            int sumi_0 = 0;
            int sumi_1 = 0;

            for (int j = 0; j < QK8_0 / 4; j++)
            {
                byte v0 = x0[j];
                byte v1 = x1[j];

                int i0_0 = (byte)(v0 & 0x0F) - 8;
                int i1_0 = (byte)(v0 >> 4) - 8;

                int i0_1 = (byte)(v1 & 0x0F) - 8;
                int i1_1 = (byte)(v1 >> 4) - 8;


                int i2_0 = y0[2 * j + 0];
                int i3_0 = y0[2 * j + 1];

                int i2_1 = y0[2 * (j + QK8_0 / 4) + 0];
                int i3_1 = y0[2 * (j + QK8_0 / 4) + 1];

                sumi_0 += i0_0 * i2_0 + i1_0 * i3_0;
                sumi_1 += i0_1 * i2_1 + i1_1 * i3_1;
            }

            sumf += (d0 * y[i].d) * sumi_0;
            sumf += (d1 * y[i].d) * sumi_1;
        }
        *s = sumf;
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q5_0_q8_0(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_0;

        Debug.Assert(n % QK8_0 == 0);
        Debug.Assert(nb % 2 == 0);
        Debug.Assert(QK8_0 == QK5_0);

        block_q5_0* x = (block_q5_0*)vx;
        block_q8_0* y = (block_q8_0*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            byte* x0 = x[i].qs;
            byte* y0 = y[i].qs;

            uint qh = *(uint*)x[i].qh;

            float d = (float)x[i].d;

            int sxy = 0;

            for (int j = 0; j < QK8_0 / 2; j++)
            {
                byte v0 = x0[j];

                int x0_0h = (int)(((qh & (1 << (2 * j + 0))) >> (2 * j + 0)) << 4);
                int x1_0h = (int)(((qh & (1 << (2 * j + 1))) >> (2 * j + 1)) << 4);

                int x0_0 = ((v0 & 0x0F) | x0_0h) - 16;
                int x1_0 = ((v0 >> 4) | x1_0h) - 16;

                int y0_0 = y0[2 * j + 0];
                int y1_0 = y0[2 * j + 1];

                sxy += x0_0 * y0_0 + x1_0 * y1_0;
            }

            sumf += (d * sxy) * y[i].d;
        }
        *s = sumf;
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q5_1_q8_1(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_1;

        Debug.Assert(n % QK8_1 == 0);
        Debug.Assert(nb % 2 == 0);
        Debug.Assert(QK8_1 == QK5_1);

        block_q5_1* x = (block_q5_1*)vx;
        block_q8_1* y = (block_q8_1*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            byte* x0 = x[i].qs;
            byte* y0 = y[i].qs;

            uint qh = *(uint*)x[i].qh;

            float d = (float)x[i].d;
            float m = (float)x[i].m;

            int sxy = 0;

            for (int j = 0; j < QK8_0 / 2; j++)
            {
                byte v0 = x0[j];

                int x0_0h = (int)(((qh & (1 << (2 * j + 0))) >> (2 * j + 0)) << 4);
                int x1_0h = (int)(((qh & (1 << (2 * j + 1))) >> (2 * j + 1)) << 4);

                int x0_0 = (v0 & 0x0F) | x0_0h;
                int x1_0 = (v0 >> 4) | x1_0h;

                int y0_0 = y0[2 * j + 0];
                int y1_0 = y0[2 * j + 1];

                sxy += x0_0 * y0_0 + x1_0 * y1_0;
            }

            sumf += (d * sxy) * y[i].d + m * (y[i].s0 + y[i].s1);
        }
        *s = sumf;
    }

    [UnmanagedCallersOnly]
    static void ggml_vec_dot_q8_0_q8_0(int n, float* s, void* vx, void* vy)
    {
        int nb = n / QK8_0;

        Debug.Assert(n % QK8_0 == 0);
        Debug.Assert(nb % 2 == 0);

        block_q8_0* x = (block_q8_0*)vx;
        block_q8_0* y = (block_q8_0*)vy;
        // scalar
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++)
        {
            byte* x0 = x[i].qs;
            byte* y0 = y[i].qs;

            int sumi = 0;

            for (int j = 0; j < QK8_0; j++)
            { 
                int v0 = x0[j];
                int v1 = y0[j];

                sumi += v0*v1;
            }

            sumf += (x[i].d * y[i].d) * sumi;
        }
        *s = sumf;
    }

    public static ggml_context* ggml_init(ggml_init_params @params)
    {
        // make this function thread safe
        ggml_critical_section_start();

        if (is_first_call)
        {
            // initialize time system (required on Windows)
            ggml_time_init();

            // initialize GELU, SILU and EXP F32 tables
            {
                long t_start = ggml_time_us(); //UNUSED(t_start);

                ushort ii;
                for (int i = 0; i < (1 << 16); ++i)
                {
                    ushort ui = (ushort)i;
                    //memcpy(&ii, &ui, sizeof(ushort));
                    ii = ui; // Probably different configurations define differently, that's why C code use memcpy.
                    float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
#if INIT_TABLES
                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
                table_exp_f16[i] = GGML_FP32_TO_FP16(expf(f));
#endif
                }

                long t_end = ggml_time_us();

                GGML_PRINT_DEBUG($"{nameof(ggml_init)}: GELU, SILU and EXP tables initialized in {(t_end - t_start) / 1000.0f} ms\n");
            }

            // initialize g_state
            unsafe
            {
                long t_start = ggml_time_us();

                g_state = (ggml_state*)NativeMemory.AlignedAlloc((nuint)sizeof(ggml_state), GGML_MEM_ALIGN);

                // https://github.com/dotnet/csharplang/blob/main/proposals/inline-arrays.md
                ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
                for (int i = 0; i < GGML_MAX_CONTEXTS; ++i)
                    unsafe
                    {
                        var itemPtr = basePtr + i;
                        itemPtr->used = false;
                    }

                long t_end = ggml_time_us();
                GGML_PRINT_DEBUG($"{nameof(ggml_init)}: g_state initialized in {(t_end - t_start) / 1000.0f} ms\n");
            }

            // initialize cuBLAS
#if GGML_USE_CUBLAS
        ggml_init_cublas();
#elif GGML_USE_CLBLAST
        ggml_cl_init();
#endif

            is_first_call = false;
        }

        // find non-used context in g_state
        ggml_context* ctx = null;

        {
            ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
            for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
                unsafe
                {
                    var itemPtr = basePtr + i;
                    if (!itemPtr->used)
                    {
                        itemPtr->used = true;
                        ctx = &itemPtr->context;

                        GGML_PRINT_DEBUG($"{nameof(ggml_init)}: found unused context {i}\n");
                        break;
                    }
                }
        }

        if (ctx == null)
        {
            GGML_PRINT_DEBUG($"{nameof(ggml_init)}: no unused context found\n");

            ggml_critical_section_end();

            return null;
        }

        ulong mem_size = (@params.mem_size + GGML_MEM_ALIGN - 1) & ~(ulong)(GGML_MEM_ALIGN - 1);

        *ctx = new ggml_context
        {
            mem_size = mem_size,
            mem_buffer = @params.mem_buffer is not null ? @params.mem_buffer : NativeMemory.AlignedAlloc((nuint)mem_size, GGML_MEM_ALIGN),
            mem_buffer_owned = @params.mem_buffer is not null ? false : true,
            no_alloc = @params.no_alloc,
            n_objects = 0,
            objects_begin = null,
            objects_end = null,
            //scratch            = { 0, 0, NULL, },
            //scratch_save       = { 0, 0, NULL, },
        };

        Debug.Assert(ctx->mem_buffer != null);

        ggml_assert_aligned(ctx->mem_buffer);

        GGML_PRINT_DEBUG($"{nameof(ggml_init)}: context initialized\n");

        ggml_critical_section_end();

        return ctx;
    }

    public static void ggml_free(ggml_context* ctx)
    {
        // make this function thread safe
        ggml_critical_section_start();

        bool found = false;
        ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
        for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
            unsafe
            {
                var itemPtr = basePtr + i;
                if (&itemPtr->context == ctx)
                {
                    itemPtr->used = false;

                    GGML_PRINT_DEBUG($"{nameof(ggml_free)}: context {i} with {ctx->n_objects} objects has been freed. memory used = {ctx->objects_end->offs + ctx->objects_end->size}\n");

                    if (ctx->mem_buffer_owned)
                    {
                        //GGML_ALIGNED_FREE(ctx->mem_buffer);
                        NativeMemory.AlignedFree(ctx->mem_buffer);
                    }

                    found = true;
                    break;
                }
            }

        if (!found)
        {
            GGML_PRINT_DEBUG($"{nameof(ggml_free)}: context not found\n");
        }

        ggml_critical_section_end();
    }

    public static void ggml_print_object(in ggml_object* obj)
    {
        GGML_PRINT($" - ggml_object: offset = {obj->offs}, size = {obj->size}, next = {(nint)obj->next}\n");
    }

    public static void ggml_print_objects(in ggml_context* ctx)
    {
        ggml_object* obj = ctx->objects_begin;

        GGML_PRINT($"{nameof(ggml_print_objects)}: objects in context {(nint)ctx}:\n");

        while (obj != null)
        {
            ggml_print_object(obj);
            obj = obj->next;
        }

        GGML_PRINT($"{nameof(ggml_print_objects)}: --- end ---\n");
    }

    public static ggml_tensor* ggml_new_tensor(
            ggml_context* ctx,
              ggml_type type,
            int n_dims,
            long* ne)
    {
        return ggml_new_tensor_impl(ctx, type, n_dims, ne, null);
    }

    public static ggml_tensor* ggml_new_tensor_1d(
            ggml_context* ctx,
            ggml_type type,
            long ne0)
    {
        return ggml_new_tensor(ctx, type, 1, &ne0);
    }

    public static ggml_tensor* ggml_new_tensor_2d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1)
    {
        long* ne = stackalloc long[2] { ne0, ne1 };
        return ggml_new_tensor(ctx, type, 2, ne);
    }

    public static ggml_tensor* ggml_new_tensor_3d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1,
            long ne2)
    {
        long* ne = stackalloc long[3] { ne0, ne1, ne2 };
        return ggml_new_tensor(ctx, type, 3, ne);
    }

    public static ggml_tensor* ggml_new_tensor_4d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1,
            long ne2,
            long ne3)
    {
        long* ne = stackalloc long[4] { ne0, ne1, ne2, ne3 };
        return ggml_new_tensor(ctx, type, 4, ne);
    }

    public static ggml_tensor* ggml_new_i32(ggml_context* ctx, int value)
    {
        ctx->scratch_save = ctx->scratch;
        ctx->scratch.data = null;

        ggml_tensor* result = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I32, 1);

        ctx->scratch = ctx->scratch_save;

        ggml_set_i32(result, value);

        return result;
    }

    public static ggml_tensor* ggml_new_f32(ggml_context* ctx, float value)
    {
        ctx->scratch_save = ctx->scratch;
        ctx->scratch.data = null;

        ggml_tensor* result = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I32, 1);

        ctx->scratch = ctx->scratch_save;

        ggml_set_f32(result, value);

        return result;
    }

    public static ggml_tensor* ggml_dup_tensor(ggml_context* ctx, in ggml_tensor* src)
    {
        return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, null);
    }

    public static ggml_tensor* ggml_set_zero(ggml_tensor* tensor)
    {
        NativeMemory.Fill(tensor->data, ggml_nbytes(tensor), 0);
        return tensor;
    }

    public static ggml_tensor* ggml_set_i32(ggml_tensor* tensor, int value)
    {
        nuint n = (nuint)ggml_nrows(tensor);
        int nc = (int)tensor->ne[0];
        nuint n1 = (nuint)tensor->nb[1];

        byte* data = (byte*)tensor->data;

        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i8(nc, (byte*)(data + i * n1), (byte)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i16(nc, (short*)(data + i * n1), (short)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i32(nc, (int*)(data + i * n1), value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f16(nc, (Half*)(data + i * n1), value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
                    }
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return tensor;
    }

    public static ggml_tensor* ggml_set_f32(ggml_tensor* tensor, float value)
    {
        nuint n = (nuint)ggml_nrows(tensor);
        int nc = (int)tensor->ne[0];
        nuint n1 = (nuint)tensor->nb[1];

        byte* data = (byte*)tensor->data;

        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i8(nc, (byte*)(data + i * n1), (byte)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i16(nc, (short*)(data + i * n1), (short)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i32(nc, (int*)(data + i * n1), (int)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f16(nc, (Half*)(data + i * n1), (int)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
                    }
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return tensor;
    }

    public static void ggml_vec_set_i8(int n, byte* x, byte v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_i16(int n, short* x, short v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_i32(int n, int* x, int v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_f16(int n, Half* x, int v) { for (int i = 0; i < n; ++i) x[i] = (Half)v; }
    public static void ggml_vec_set_f32(int n, float* x, float v) { for (int i = 0; i < n; ++i) x[i] = v; }


    public static float ggml_get_f32_1d(ggml_tensor* tensor, int i)
    {
        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    return ((byte*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    return ((short*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    return ((int*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    return (float)(((Half*)(tensor->data))[i]);
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    return ((float*)(tensor->data))[i];
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return 0.0f;
    }

    // check if node is part of the graph
    private static bool ggml_graph_find(ggml_cgraph* cgraph, ggml_tensor* node)
    {
        if (cgraph == null)
        {
            return true;
        }

        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            if (cgraph->get_node(i) == node)
            {
                return true;
            }
        }

        return false;
    }

    private static ggml_tensor* ggml_graph_get_parent(ggml_cgraph* cgraph, ggml_tensor* node)
    {
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            ggml_tensor* parent = cgraph->get_node(i);

            if (parent->grad == node)
            {
                return parent;
            }
        }

        return null;
    }

#if GGML_PERF
    public static long ggml_perf_time_ms() => ggml_time_ms();
    public static long ggml_perf_time_us() => ggml_time_us();
    public static long ggml_perf_cycles() => ggml_cycles();
    public static long ggml_perf_cycles_per_ms() => ggml_cycles_per_ms();
#else
    public static long ggml_perf_time_ms() => 0;
    public static long ggml_perf_time_us() => 0;
    public static long ggml_perf_cycles() => 0;
    public static long ggml_perf_cycles_per_ms() => 0;
#endif

    private static void ggml_lock_init(int* spin) { }
    private static void ggml_lock_destroy(int* spin) { }
    private static void ggml_lock_lock(int* spin) { }
    private static void ggml_lock_unlock(int* spin) { }
    public static void ggml_graph_dump_dot(ggml_cgraph* gb, ggml_cgraph* gf, string filename)
    {
        string color;

        using var fp = new StreamWriter(File.OpenWrite(filename));

        fp.WriteLine("digraph G {");
        fp.WriteLine("  newrank = true;");
        fp.WriteLine("  rankdir = LR;");

        for (int i = 0; i < gb->n_nodes; i++)
        {
            ggml_tensor* node = gb->get_node(i);

            if (ggml_graph_get_parent(gb, node) != null)
            {
                continue;
            }

            if (node->is_param)
            {
                color = "yellow";
            }
            else if (node->grad is not null)
            {
                if (ggml_graph_find(gf, node))
                {
                    color = "green";
                }
                else
                {
                    color = "lightblue";
                }
            }
            else
            {
                color = "white";
            }

            fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"{i} [{node->ne[0]}, {node->ne[1]} | <x>{GGML_OP_SYMBOL[(int)node->op]}");

            if (node->grad is not null)
            {
                fp.WriteLine($" | <g>{GGML_OP_SYMBOL[(int)node->grad->op]}\"; ]");
            }
            else
            {
                fp.WriteLine("\"; ]");
            }
        }

        for (int i = 0; i < gb->n_leafs; i++)
        {
            ggml_tensor* node = gb->get_leaf(i);

            color = "pink";

            if (ggml_nelements(node) == 1)
            {
                fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>{(double)ggml_get_f32_1d(node, 0)}\"; ]");
            }
            else
            {
                fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>CONST {i} [{node->ne[0]}, {node->ne[1]}]\"; ]");
            }
        }

        for (int i = 0; i < gb->n_nodes; i++)
        {
            ggml_tensor* node = gb->get_node(i);

            ggml_tensor* parent = ggml_graph_get_parent(gb, node);

            if (node->src0 is not null)
            {
                ggml_tensor* parent0 = ggml_graph_get_parent(gb, node->src0);

                fp.WriteLine("  \"{0}\":{1} -> \"{2}\":{3} [ arrowhead = {4}; style = {5}; label = \"x\"; ]",
                    parent0 is not null ? (nint)parent0 : (nint)node->src0,
                    parent0 is not null ? "g" : "x",
                    parent is not null ? (nint)parent : (nint)node,
                    parent is not null ? "g" : "x",
                    parent is not null ? "empty" : "vee",
                    parent is not null ? "dashed" : "solid");
            }

            if (node->src1 is not null)
            {
                ggml_tensor* parent1 = ggml_graph_get_parent(gb, node->src1);

                fp.WriteLine("  \"{0}\":{1} -> \"{2}\":{3} [ arrowhead = {4}; style = {5}; label = \"x\"; ]",
                        parent1 is not null ? (nint)parent1 : (nint)node->src1,
                        parent1 is not null ? "g" : "x",
                        parent is not null ? (nint)parent : (nint)node,
                        parent is not null ? "g" : "x",
                        parent is not null ? "empty" : "vee",
                        parent is not null ? "dashed" : "solid");
            }
        }

        for (int i = 0; i < gb->n_leafs; i++)
        {
            ggml_tensor* node = gb->get_leaf(i);

            if (node->src0 is not null)
            {
                fp.WriteLine("  \"{0}\":x -> \"{1}\":x [ label = \"x\"; ]\n",
                        (nint)node->src0,
                        (nint)node);
            }

            if (node->src1 is not null)
            {
                fp.WriteLine("  \"{0}\":x -> \"{1}\":x [ label = \"y\"; ]\n",
                        (nint)node->src1,
                        (nint)node);
            }
        }

        fp.WriteLine("}\n");

        GGML_PRINT($"{nameof(ggml_graph_dump_dot)}: dot -Tpng {filename} -o {filename}.png && open {filename}.png\n");
    }
    private static void atomic_store(ref int ptr, int value)
    {
        Interlocked.Exchange(ref ptr, value);
    }
    private static int atomic_load(ref int ptr)
    {
        return Interlocked.CompareExchange(ref ptr, 0, 0);
    }
    private static int atomic_fetch_add(ref int ptr, int inc)
    {
        return Interlocked.Add(ref ptr, inc);
    }
    private static int atomic_fetch_sub(ref int ptr, int dec)
    {
        return atomic_fetch_add(ref ptr, -dec);
    }
    static int ggml_up32(int n)
    {
        return (n + 31) & ~31;
    }
    static int ggml_up64(int n)
    {
        return (n + 63) & ~63;
    }
    static int ggml_up(int n, int m)
    {
        // assert m is a power of 2
        Debug.Assert((m & (m - 1)) == 0);
        return (n + m - 1) & ~(m - 1);
    }
    public static void ggml_graph_compute(ggml_context* ctx, ggml_cgraph* cgraph)
    {
        int n_threads = cgraph->n_threads;

        ggml_compute_state_shared state_shared = new ggml_compute_state_shared {
            spin      = GGML_LOCK_INITIALIZER,
            n_threads = n_threads,
            n_ready   = 0,
            has_work  = 0,
            stop      = 0,
        };
        //ggml_compute_state* workers = n_threads > 1 ? stackalloc ggml_compute_state[(n_threads - 1)] : null;
        ggml_compute_state* workers = n_threads > 1 ? (ggml_compute_state*)NativeMemory.Alloc((nuint)sizeof(ggml_compute_state) * (nuint)(n_threads - 1)) : null;

        // create thread pool
        if (n_threads > 1)
        {
            ggml_lock_init(&state_shared.spin);

            Interlocked.Exchange(ref state_shared.has_work, 1);

            for (int j = 0; j < n_threads - 1; j++)
            {
                workers[j] = new ggml_compute_state {
                    thrd = 0,
                    @params = new ggml_compute_params
                    {
                        type = ggml_task_type.GGML_TASK_COMPUTE,
                        ith = j + 1,
                        nth = n_threads,
                        wsize = cgraph->work is not null ? ggml_nbytes(cgraph->work) : 0,
                        wdata = cgraph->work is not null ? cgraph->work->data : null,
                    },
                    node = null,
                    shared = &state_shared,
                };

                int rc = ggml_thread_create(&workers[j].thrd, null, &ggml_graph_compute_thread, &workers[j]);
                Debug.Assert(rc == 0);
            }
        }

        // initialize tasks + work buffer
        {
            nuint work_size = 0;

            // thread scheduling for the different operations
            for (int i = 0; i < cgraph->n_nodes; i++)
            {
                ggml_tensor* node = cgraph->get_node(i);

                switch (node->op)
                {
                    case ggml_op.GGML_OP_CPY:
                    case ggml_op.GGML_OP_DUP:
                        {
                            node->n_tasks = n_threads;

                            nuint cur = 0;
                            if (ggml_is_quantized(node->type))
                            {
                                cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F32] * (nuint)node->ne[0] * (nuint)n_threads;
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_ADD:
                        {
                            node->n_tasks = n_threads;

                            nuint cur = 0;

                            if (ggml_is_quantized(node->src0->type))
                            {
                                cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F32] * (nuint)node->src0->ne[0] * (nuint)n_threads;
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_SUB:
                    case ggml_op.GGML_OP_MUL:
                    case ggml_op.GGML_OP_DIV:
                    case ggml_op.GGML_OP_SQR:
                    case ggml_op.GGML_OP_SQRT:
                    case ggml_op.GGML_OP_SUM:
                    case ggml_op.GGML_OP_MEAN:
                    case ggml_op.GGML_OP_REPEAT:
                    case ggml_op.GGML_OP_ABS:
                    case ggml_op.GGML_OP_SGN:
                    case ggml_op.GGML_OP_NEG:
                    case ggml_op.GGML_OP_STEP:
                    case ggml_op.GGML_OP_RELU:
                        {
                            node->n_tasks = 1;
                        }
                        break;
                    case ggml_op.GGML_OP_GELU:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_SILU:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_NORM:
                    case ggml_op.GGML_OP_RMS_NORM:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_MUL_MAT:
                        {
                            node->n_tasks = n_threads;

                            // TODO: use different scheduling for different matrix sizes
                            //const int nr0 = ggml_nrows(node->src0);
                            //const int nr1 = ggml_nrows(node->src1);

                            //node->n_tasks = MIN(n_threads, MAX(1, nr0/128));
                            //printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

                            nuint cur = 0;

                            if (node->src0->type == ggml_type.GGML_TYPE_F16 && node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
                            if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                node->n_tasks = 1; // TODO: this actually is doing nothing
                                                   //       the threads are still spinning
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                //printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
                                //printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
                                //printf("cur = %zu\n", cur);
                            } else {
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F16]*ggml_nelements(node->src1);
                            }
#else
                                cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F16] * (nuint)ggml_nelements(node->src1);
#endif
                            }
                            else if (node->src0->type == ggml_type.GGML_TYPE_F32 && node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
                                cur = 0;
                            }
                            else if (ggml_is_quantized(node->src0->type) && node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
                            if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                node->n_tasks = 1;
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                            } else
#endif
                                {
                                    ggml_type type_q = quantize_fns[(int)node->src0->type].vec_dot_type;
                                    cur = (nuint)GGML_TYPE_SIZE[(int)type_q] * (nuint)ggml_nelements(node->src1) / (nuint)GGML_BLCK_SIZE[(int)type_q];
                                }
                            }
                            else
                            {
                                Debug.Assert(false);
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_SCALE:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_CONT:
                    case ggml_op.GGML_OP_RESHAPE:
                    case ggml_op.GGML_OP_VIEW:
                    case ggml_op.GGML_OP_PERMUTE:
                    case ggml_op.GGML_OP_TRANSPOSE:
                    case ggml_op.GGML_OP_GET_ROWS:
                    case ggml_op.GGML_OP_DIAG_MASK_INF:
                        {
                            node->n_tasks = 1;
                        }
                        break;
                    case ggml_op.GGML_OP_SOFT_MAX:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_ROPE:
                        {
                            node->n_tasks = n_threads;
                        }
                        break;
                    case ggml_op.GGML_OP_ALIBI:
                        {
                            node->n_tasks = 1; //TODO
                        }
                        break;
                    case ggml_op.GGML_OP_CONV_1D_1S:
                    case ggml_op.GGML_OP_CONV_1D_2S:
                        {
                            node->n_tasks = n_threads;

                            Debug.Assert(node->src0->ne[3] == 1);
                            Debug.Assert(node->src1->ne[2] == 1);
                            Debug.Assert(node->src1->ne[3] == 1);

                            nuint cur = 0;
                            int nk = (int)node->src0->ne[0];

                            if (node->src0->type == ggml_type.GGML_TYPE_F16 &&
                                node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
                                cur = (nuint)(sizeof(Half) * (
                                        nk * ggml_up32((int)node->src0->ne[1]) * node->src0->ne[2] +
                                        (2 * (nk / 2) + node->src1->ne[0]) * node->src1->ne[1]
                                        ));
                            }
                            else if (node->src0->type == ggml_type.GGML_TYPE_F32 &&
                                       node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
                                cur = (nuint)(sizeof(float) * (
                                        nk * ggml_up32((int)node->src0->ne[1]) * node->src0->ne[2] +
                                        (2 * (nk / 2) + node->src1->ne[0]) * node->src1->ne[1]
                                        ));
                            }
                            else
                            {
                                Debug.Assert(false);
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_FLASH_ATTN:
                        {
                            node->n_tasks = n_threads;

                            nuint cur = 0;

                            long ne11 = ggml_up((int)node->src1->ne[1], GGML_SOFT_MAX_UNROLL);

                            if (node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
                                cur = (nuint)(sizeof(float) * ne11 * node->n_tasks); // TODO: this can become (n_tasks-1)
                                cur += (nuint)(sizeof(float) * ne11 * node->n_tasks); // this is overestimated by x2
                            }

                            if (node->src1->type == ggml_type.GGML_TYPE_F16)
                            {
                                cur = (nuint)(sizeof(float) * ne11 * node->n_tasks); // TODO: this can become (n_tasks-1)
                                cur += (nuint)(sizeof(float) * ne11 * node->n_tasks); // this is overestimated by x2
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_FLASH_FF:
                        {
                            node->n_tasks = n_threads;

                            nuint cur = 0;

                            if (node->src1->type == ggml_type.GGML_TYPE_F32)
                            {
                                cur = (nuint)(sizeof(float) * node->src1->ne[1] * node->n_tasks); // TODO: this can become (n_tasks-1)
                                cur += (nuint)(sizeof(float) * node->src1->ne[1] * node->n_tasks); // this is overestimated by x2
                            }

                            if (node->src1->type == ggml_type.GGML_TYPE_F16)
                            {
                                cur = (nuint)(sizeof(float) * node->src1->ne[1] * node->n_tasks); // TODO: this can become (n_tasks-1)
                                cur += (nuint)(sizeof(float) * node->src1->ne[1] * node->n_tasks); // this is overestimated by x2
                            }

                            work_size = Math.Max(work_size, cur);
                        }
                        break;
                    case ggml_op.GGML_OP_MAP_UNARY:
                    case ggml_op.GGML_OP_MAP_BINARY:
                        {
                            node->n_tasks = 1;
                        }
                        break;
                    case ggml_op.GGML_OP_NONE:
                        {
                            node->n_tasks = 1;
                        }
                        break;
                    case ggml_op.GGML_OP_COUNT:
                        {
                            Debug.Assert(false);
                        }
                        break;
                }
            }

            if (cgraph->work != null && work_size > cgraph->work_size)
            {
                Debug.Assert(false); // TODO: better handling
            }

            if (work_size > 0 && cgraph->work == null)
            {
                cgraph->work_size = work_size + (nuint)CACHE_LINE_SIZE * (nuint)(n_threads - 1);

                GGML_PRINT_DEBUG($"{nameof(ggml_graph_compute)}: allocating work buffer for graph ({cgraph->work_size} bytes)\n");
                cgraph->work = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I8, (long)cgraph->work_size);
            }
        }

        long perf_start_cycles = ggml_perf_cycles();
        long perf_start_time_us = ggml_perf_time_us();

        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            GGML_PRINT_DEBUG_5($"{nameof(ggml_graph_compute)}: {i}/{cgraph->n_nodes}\n");

            ggml_tensor* node = cgraph->get_node(i);

            // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
            //if (node->grad == NULL && node->perf_runs > 0) {
            //    continue;
            //}

            long perf_node_start_cycles = ggml_perf_cycles();
            long perf_node_start_time_us = ggml_perf_time_us();

            // INIT
            ggml_compute_params @params = new ggml_compute_params
            {
                type  = ggml_task_type.GGML_TASK_INIT,
                ith   = 0,
                nth   = node->n_tasks,
                wsize = cgraph->work is not null ? ggml_nbytes(cgraph->work) : 0,
                wdata = cgraph->work is not null ? cgraph->work->data : null,
            };

            ggml_compute_forward(&@params, node);

            // COMPUTE
            if (node->n_tasks > 1)
            {
                if (atomic_fetch_add(ref state_shared.n_ready, 1) == n_threads - 1)
                {
                    Interlocked.Exchange(ref state_shared.has_work, 0);
                }

                while (atomic_load(ref state_shared.has_work) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                // launch thread pool
                for (int j = 0; j < n_threads - 1; j++)
                {
                    workers[j].@params = new ggml_compute_params
                    {
                        type = ggml_task_type.GGML_TASK_COMPUTE,
                        ith = j + 1,
                        nth = node->n_tasks,
                        wsize = cgraph->work is not null ? ggml_nbytes(cgraph->work) : 0,
                        wdata = cgraph->work is not null ? cgraph->work->data : null,
                    };
                    workers[j].node = node;
                }

                atomic_fetch_sub(ref state_shared.n_ready, 1);

                while (atomic_load(ref state_shared.n_ready) > 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                atomic_store(ref state_shared.has_work, 1);
            }

            @params.type = ggml_task_type.GGML_TASK_COMPUTE;
            ggml_compute_forward(&@params, node);

            // wait for thread pool
            if (node->n_tasks > 1)
            {
                if (atomic_fetch_add(ref state_shared.n_ready, 1) == n_threads - 1)
                {
                    atomic_store(ref state_shared.has_work, 0);
                }

                while (atomic_load(ref state_shared.has_work) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                atomic_fetch_sub(ref state_shared.n_ready, 1);

                while (atomic_load(ref state_shared.n_ready) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }
            }

            // FINALIZE
            if (node->n_tasks > 1)
            {
                if (atomic_fetch_add(ref state_shared.n_ready, 1) == n_threads - 1)
                {
                    atomic_store(ref state_shared.has_work, 0);
                }

                while (atomic_load(ref state_shared.has_work) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                // launch thread pool
                for (int j = 0; j < n_threads - 1; j++)
                {
                    workers[j].@params = new ggml_compute_params {
                        type = ggml_task_type.GGML_TASK_FINALIZE,
                        ith = j + 1,
                        nth = node->n_tasks,
                        wsize = cgraph->work is not null ? ggml_nbytes(cgraph->work) : 0,
                        wdata = cgraph->work is not null ? cgraph->work->data : null,
                    };
                    workers[j].node = node;
                }

                atomic_fetch_sub(ref state_shared.n_ready, 1);

                while (atomic_load(ref state_shared.n_ready) > 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                atomic_store(ref state_shared.has_work, 1);
            }

            @params.type = ggml_task_type.GGML_TASK_FINALIZE;
            ggml_compute_forward(&@params, node);

            // wait for thread pool
            if (node->n_tasks > 1)
            {
                if (atomic_fetch_add(ref state_shared.n_ready, 1) == n_threads - 1)
                {
                    atomic_store(ref state_shared.has_work, 0);
                }

                while (atomic_load(ref state_shared.has_work) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }

                atomic_fetch_sub(ref state_shared.n_ready, 1);

                while (atomic_load(ref state_shared.n_ready) != 0)
                {
                    ggml_lock_lock(&state_shared.spin);
                    ggml_lock_unlock(&state_shared.spin);
                }
            }

            // performance stats (node)
            {
                long perf_cycles_cur = ggml_perf_cycles() - perf_node_start_cycles;
                long perf_time_us_cur = ggml_perf_time_us() - perf_node_start_time_us;

                node->perf_runs++;
                node->perf_cycles += perf_cycles_cur;
                node->perf_time_us += perf_time_us_cur;
            }
        }

        // join thread pool
        if (n_threads > 1)
        {
            atomic_store(ref state_shared.stop, 1);
            atomic_store(ref state_shared.has_work, 1);

            for (int j = 0; j < n_threads - 1; j++)
            {
                int rc = ggml_thread_join(workers[j].thrd, null);
                Debug.Assert(rc == 0);
            }

            ggml_lock_destroy(&state_shared.spin);
        }

        // performance stats (graph)
        {
            long perf_cycles_cur = ggml_perf_cycles() - perf_start_cycles;
            long perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

            cgraph->perf_runs++;
            cgraph->perf_cycles += perf_cycles_cur;
            cgraph->perf_time_us += perf_time_us_cur;

            GGML_PRINT_DEBUG($"{nameof(ggml_graph_compute)}: perf ({cgraph->perf_runs}) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                    (double)perf_cycles_cur / (double)ggml_cycles_per_ms(),
                    (double)cgraph->perf_cycles / (double)ggml_cycles_per_ms() / (double)cgraph->perf_runs,
                    (double)perf_time_us_cur / 1000.0,
                    (double)cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
        }
    }

    public static void ggml_graph_reset(ggml_cgraph* cgraph)
    {
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            ggml_tensor* grad = cgraph->get_grad(i);

            if (grad is not null)
            {
                ggml_set_zero(grad);
            }
        }
    }

    public static ggml_tensor* ggml_view_tensor(
        ggml_context* ctx,
        ggml_tensor* src)
    {
        ggml_tensor* result = ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);

        result->nb[0] = src->nb[0];
        result->nb[1] = src->nb[1];
        result->nb[2] = src->nb[2];
        result->nb[3] = src->nb[3];

        return result;
    }


    public static long ggml_nelements(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
        return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    }

    public static int ggml_nrows(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return (int)(tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
    }

    public static nuint ggml_nbytes(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return ((nuint)ggml_nelements(tensor) * (nuint)GGML_TYPE_SIZE[(int)tensor->type]) / (nuint)GGML_BLCK_SIZE[(int)tensor->type];
    }

    public static void ggml_set_param(
        ggml_context* ctx,
        ggml_tensor* tensor)
    {
        tensor->is_param = true;

        Debug.Assert(tensor->grad == null);
        tensor->grad = ggml_dup_tensor(ctx, tensor);
    }

    public static ggml_tensor* ggml_add(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_add_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_add_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_add_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_sub(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_sub_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_sub_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_sub_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_mul(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_mul_inplace(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_div(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_div_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_div_inplace(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_div_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_sqr(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sqr_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sqr_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sqr_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sqrt(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sqrt_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sqrt_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sqrt_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sum(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            is_node = true;
        }

        ggml_tensor* result = ggml_new_tensor_1d(ctx, a->type, 1);

        result->op = ggml_op.GGML_OP_SUM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    public static ggml_tensor* ggml_mean(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement
            is_node = true;
        }

        long* ne = stackalloc long[GGML_MAX_DIMS] { 1, a->ne[1], a->ne[2], a->ne[3] };
        ggml_tensor* result = ggml_new_tensor(ctx, ggml_type.GGML_TYPE_F32, a->n_dims, ne);

        result->op = ggml_op.GGML_OP_MEAN;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    public static ggml_tensor* ggml_repeat(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        Debug.Assert(ggml_can_repeat(a, b));

        bool is_node = false;

        if (a->grad is not null)
        {
            is_node = true;
        }

        if (ggml_are_same_shape(a, b) && !is_node)
        {
            return a;
        }

        ggml_tensor* result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

        result->op = ggml_op.GGML_OP_REPEAT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    public static ggml_tensor* ggml_abs(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_abs_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_abs_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_abs_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sgn(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sgn_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sgn_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sgn_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_neg(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_neg_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_neg_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_neg_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_step(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_step_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_step_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_step_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_relu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_relu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_relu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_relu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_gelu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_gelu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_gelu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_gelu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_silu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_silu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_silu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_silu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_norm(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_norm_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_norm_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_norm_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_rms_norm(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_rms_norm_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_rms_norm_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_rms_norm_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_mul_mat(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_mat_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_mul_mat_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_mul_mat_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_scale(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_scale_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_scale_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_scale_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_cpy(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_cpy_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_cpy_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_cpy_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_cont(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_cont_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_cont_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_cont_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_transpose(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = ggml_view_tensor(ctx, a);

        result->ne[0] = a->ne[1];
        result->ne[1] = a->ne[0];

        result->nb[0] = a->nb[1];
        result->nb[1] = a->nb[0];

        result->op = ggml_op.GGML_OP_TRANSPOSE;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static void ggml_compute_backward(ggml_context* ctx, ggml_tensor* tensor, bool inplace)
    {
        ggml_tensor* src0 = tensor->src0;
        ggml_tensor* src1 = tensor->src1;

        switch (tensor->op)
        {
            case ggml_op.GGML_OP_DUP:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_ADD:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad = ggml_add_impl(ctx, src1->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SUB:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad = ggml_sub_impl(ctx, src1->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_MUL:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx, src1, tensor->grad),
                                    inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_add_impl(ctx,
                                    src1->grad,
                                    ggml_mul(ctx, src0, tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_DIV:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_div(ctx, tensor->grad, src1),
                                    inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_sub_impl(ctx,
                                    src1->grad,
                                    ggml_mul(ctx,
                                        tensor->grad,
                                        ggml_div(ctx, tensor, src1)),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SQR:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx,
                                        ggml_mul(ctx, src0, tensor->grad),
                                        ggml_repeat(ctx, ggml_new_f32(ctx, 2.0f), src0)),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SQRT:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_div(ctx,
                                        ggml_repeat(ctx, ggml_new_f32(ctx, 0.5f), tensor),
                                        tensor),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SUM:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_repeat(ctx, tensor->grad, src0->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_MEAN:
                {
                    Debug.Assert(false); // TODO: implement
                }
                break;
            case ggml_op.GGML_OP_REPEAT:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_sum(ctx, tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_ABS:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx,
                                        ggml_sgn(ctx, src0),
                                        tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SGN:
                {
                    if (src0->grad is not null)
                    {
                        // noop
                    }
                }
                break;
            case ggml_op.GGML_OP_NEG:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_sub_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_STEP:
                {
                    if (src0->grad is not null)
                    {
                        // noop
                    }
                }
                break;
            case ggml_op.GGML_OP_RELU:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_sub_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_step(ctx, src0),
                                    tensor->grad),
                                inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_GELU:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_ALIBI:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_SILU:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_NORM:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_RMS_NORM:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_MUL_MAT:
                {
                    if (src0->grad is not null)
                    {
                        // TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor->grad);
                        Debug.Assert(false);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_add_impl(ctx,
                                    src1->grad,
                                    ggml_mul_mat(ctx,
                                        ggml_cont(ctx, ggml_transpose(ctx, src0)),
                                        tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SCALE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CPY:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONT:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_RESHAPE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_VIEW:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_PERMUTE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_TRANSPOSE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_GET_ROWS:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_DIAG_MASK_INF:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_SOFT_MAX:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_ROPE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONV_1D_1S:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONV_1D_2S:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_FLASH_ATTN:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_FLASH_FF:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_MAP_UNARY:
            case ggml_op.GGML_OP_MAP_BINARY:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_NONE:
                {
                    // nop
                }
                break;
            case ggml_op.GGML_OP_COUNT:
                {
                    Debug.Assert(false);
                }
                break;
        }
    }

    private static void ggml_visit_parents(ggml_cgraph* cgraph, ggml_tensor* node)
    {
        if (node->grad == null)
        {
            // this usually happens when we generate intermediate nodes from constants in the backward pass
            // it can also happen during forward pass, if the user performs computations with constants
            if (node->op != ggml_op.GGML_OP_NONE)
            {
                //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
            }
        }

        // check if already visited
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            if (cgraph->get_node(i) == node)
            {
                return;
            }
        }

        for (int i = 0; i < cgraph->n_leafs; i++)
        {
            if (cgraph->get_leaf(i) == node)
            {
                return;
            }
        }

        if (node->src0 is not null)
        {
            ggml_visit_parents(cgraph, node->src0);
        }

        if (node->src1 is not null)
        {
            ggml_visit_parents(cgraph, node->src1);
        }

        for (int i = 0; i < GGML_MAX_OPT; ++i)
        {
            if (node->opt[i] != 0)
            {
                ggml_visit_parents(cgraph, (ggml_tensor*)node->opt[i]);
            }
        }

        if (node->op == ggml_op.GGML_OP_NONE && node->grad == null)
        {
            // reached a leaf node, not part of the gradient graph (e.g. a constant)
            Debug.Assert(cgraph->n_leafs < GGML_MAX_NODES);

            cgraph->set_leaf(cgraph->n_leafs, node);
            cgraph->n_leafs++;
        }
        else
        {
            Debug.Assert(cgraph->n_nodes < GGML_MAX_NODES);

            cgraph->set_node(cgraph->n_nodes, node);
            cgraph->set_grad(cgraph->n_nodes, node->grad);
            cgraph->n_nodes++;
        }
    }

    private static void ggml_build_forward_impl(ggml_cgraph* cgraph, ggml_tensor* tensor, bool expand)
    {
        if (!expand)
        {
            cgraph->n_nodes = 0;
            cgraph->n_leafs = 0;
        }

        int n0 = cgraph->n_nodes;

        ggml_visit_parents(cgraph, tensor);

        int n_new = cgraph->n_nodes - n0;
        GGML_PRINT_DEBUG($"{nameof(ggml_build_forward_impl)}: visited {n_new} new nodes\n");

        if (n_new > 0)
        {
            // the last added node should always be starting point
            Debug.Assert(cgraph->get_node(cgraph->n_nodes - 1) == tensor);
        }
    }

    public static void ggml_build_forward_expand(ggml_cgraph* cgraph, ggml_tensor* tensor)
    {
        ggml_build_forward_impl(cgraph, tensor, true);
    }

    public static ggml_cgraph ggml_build_forward(ggml_tensor* tensor)
    {
        ggml_cgraph result = new ggml_cgraph()
        {
            n_nodes = 0,
            n_leafs = 0,
            n_threads = GGML_DEFAULT_N_THREADS,
            work_size = 0,
            work = null,
            // nodes        = { NULL },
            // grads        = { NULL },
            // leafs        = { NULL },
            perf_runs = 0,
            perf_cycles = 0,
            perf_time_us = 0,
        };

        ggml_build_forward_impl(&result, tensor, false);

        return result;
    }

    public static ggml_cgraph ggml_build_backward(ggml_context* ctx, ggml_cgraph* gf, bool keep)
    {
        ggml_cgraph result = *gf;

        Debug.Assert(gf->n_nodes > 0);

        // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
        if (keep)
        {
            for (int i = 0; i < gf->n_nodes; i++)
            {
                ggml_tensor* node = gf->get_node(i);

                if (node->grad is not null)
                {
                    node->grad = ggml_dup_tensor(ctx, node);
                    gf->set_grad(i, node->grad);
                }
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = gf->get_node(i);

            // because we detached the grad nodes from the original graph, we can afford inplace operations
            if (node->grad is not null)
            {
                ggml_compute_backward(ctx, node, keep);
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = gf->get_node(i);

            if (node->is_param)
            {
                GGML_PRINT_DEBUG($"{nameof(ggml_build_backward)}: found root node {(nuint)node}\n");
                ggml_build_forward_impl(&result, node->grad, true);
            }
        }

        return result;
    }
    ////////////////////////////////////////////////////////////////////////////////

    static unsafe ggml_tensor* ggml_new_tensor_impl(
            ggml_context* ctx,
            ggml_type type,
            int n_dims,
            in long* ne,
            void* data)
    {
        // always insert objects at the end of the context's memory pool
        ggml_object* obj_cur = ctx->objects_end;

        ulong cur_offs = obj_cur == null ? 0 : obj_cur->offs;
        ulong cur_size = obj_cur == null ? 0 : obj_cur->size;
        ulong cur_end = cur_offs + cur_size;

        ulong size_needed = 0;

        if (data == null && !ctx->no_alloc)
        {
            size_needed += GGML_TYPE_SIZE[(int)type] * (ulong)(ne[0] / GGML_BLCK_SIZE[(int)type]);
            for (int i = 1; i < n_dims; i++)
            {
                size_needed *= (ulong)ne[i];
            }
            // align to GGML_MEM_ALIGN
            size_needed = ((size_needed + GGML_MEM_ALIGN - 1) / GGML_MEM_ALIGN) * GGML_MEM_ALIGN;
        }

        byte* mem_buffer = (byte*)ctx->mem_buffer;
        ggml_object* obj_new = (ggml_object*)(mem_buffer + cur_end);

        if (ctx->scratch.data == null || data != null)
        {
            size_needed += (ulong)sizeof(ggml_tensor);

            if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + size_needed + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
                Debug.Assert(false);
                return null;
            }

            *obj_new = new ggml_object
            {
                offs = cur_end + GGML_OBJECT_SIZE,
                size = size_needed,
                next = null,
            };
        }
        else
        {
            if (ctx->scratch.offs + size_needed > ctx->scratch.size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the scratch memory\n");
                Debug.Assert(false);
                return null;
            }

            if (cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE > ctx->mem_size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
                Debug.Assert(false);
                return null;
            }

            data = (byte*)ctx->scratch.data + ctx->scratch.offs;

            *obj_new = new ggml_object
            {
                offs = cur_end + GGML_OBJECT_SIZE,
                size = (ulong)sizeof(ggml_tensor),
                next = null,
            };

            //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

            ctx->scratch.offs += size_needed;
        }

        if (obj_cur != null)
        {
            obj_cur->next = obj_new;
        }
        else
        {
            // this is the first object in this context
            ctx->objects_begin = obj_new;
        }

        ctx->objects_end = obj_new;

        //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

        ggml_tensor* result = (ggml_tensor*)(mem_buffer + obj_new->offs);

        ggml_assert_aligned(result);

        ggml_tensor res = new ggml_tensor
        {
            /*.type         =*/
            type = type,
            /*.n_dims       =*/
            n_dims = n_dims,
            //ne = { 1, 1, 1, 1 },
            //nb = { 0, 0, 0, 0 },
            op = ggml_op.GGML_OP_NONE,
            is_param = false,
            grad = null,
            src0 = null,
            src1 = null,
            //opt = { NULL },
            n_tasks = 0,
            perf_runs = 0,
            perf_cycles = 0,
            perf_time_us = 0,
            data = (data == null && !ctx->no_alloc) ? (void*)(result + 1) : data,
            //pad = { 0 },
        };
        res.ne[0] = 1;
        res.ne[1] = 1;
        res.ne[2] = 1;
        res.ne[3] = 1;
        *result = res;

        // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
        //ggml_assert_aligned(result->data);

        for (int i = 0; i < n_dims; i++)
        {
            result->ne[i] = ne[i];
        }

        result->nb[0] = GGML_TYPE_SIZE[(int)type];
        result->nb[1] = result->nb[0] * (ulong)(result->ne[0] / GGML_BLCK_SIZE[(int)type]);
        for (int i = 2; i < GGML_MAX_DIMS; i++)
        {
            result->nb[i] = result->nb[i - 1] * (ulong)result->ne[i - 1];
        }

        ctx->n_objects++;

        return result;
    }

    private static ggml_tensor* ggml_add_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_ADD;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_sub_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SUB;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_mul_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        if (inplace)
        {
            Debug.Assert(is_node == false);
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_MUL;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_div_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        if (inplace)
        {
            Debug.Assert(is_node == false);
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_DIV;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_sqr_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SQR;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_sqrt_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SQRT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_abs_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_ABS;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_sgn_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SGN;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_neg_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_NEG;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_step_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_STEP;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_relu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_RELU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_gelu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_GELU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_silu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SILU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_norm_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_NORM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null; // TODO: maybe store epsilon here?

        return result;
    }

    private static ggml_tensor* ggml_rms_norm_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_RMS_NORM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null; // TODO: maybe store epsilon here?

        return result;
    }

    private static ggml_tensor* ggml_mul_mat_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_can_mul_mat(a, b));
        Debug.Assert(!ggml_is_transposed(a));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            is_node = true;
        }

        long* ne = stackalloc long[4] { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
        ggml_tensor* result = ggml_new_tensor(ctx, ggml_type.GGML_TYPE_F32, Math.Min(a->n_dims, b->n_dims), ne);

        result->op = ggml_op.GGML_OP_MUL_MAT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_scale_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_is_scalar(b));
        Debug.Assert(ggml_is_padded_1d(a));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            is_node = true;
        }

        // TODO: when implement backward, fix this:
        //ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
        ggml_tensor* result = ggml_view_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SCALE;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_cpy_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_nelements(a) == ggml_nelements(b));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        // make a view of the destination
        ggml_tensor* result = ggml_view_tensor(ctx, b);

        result->op = ggml_op.GGML_OP_CPY;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_cont_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_CONT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    static bool ggml_is_scalar(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_is_vector(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_is_matrix(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_can_mul_mat(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
    }

    static bool ggml_is_quantized(ggml_type type) {
        return GGML_IS_QUANTIZED[(int)type];
    }

    static bool ggml_is_transposed(ggml_tensor* tensor)
    {
        return tensor->nb[0] > tensor->nb[1];
    }

    static bool ggml_is_contiguous(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            tensor->nb[0] == GGML_TYPE_SIZE[(int)tensor->type] &&
            tensor->nb[1] == (tensor->nb[0] * (ulong)tensor->ne[0]) / (ulong)GGML_BLCK_SIZE[(int)tensor->type] &&
            tensor->nb[2] == tensor->nb[1] * (ulong)tensor->ne[1] &&
            tensor->nb[3] == tensor->nb[2] * (ulong)tensor->ne[2];
    }

    static bool ggml_is_padded_1d(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            tensor->nb[0] == GGML_TYPE_SIZE[(int)tensor->type] &&
            tensor->nb[2] == tensor->nb[1] * (ulong)tensor->ne[1] &&
            tensor->nb[3] == tensor->nb[2] * (ulong)tensor->ne[2];
    }

    private static bool ggml_are_same_shape(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
    }

    // check if t1 can be represented as a repeatition of t0
    private static bool ggml_can_repeat(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t1->ne[0] % t0->ne[0] == 0) &&
            (t1->ne[1] % t0->ne[1] == 0) &&
            (t1->ne[2] % t0->ne[2] == 0) &&
            (t1->ne[3] % t0->ne[3] == 0);
    }

    static void ggml_assert_aligned(void* ptr)
    {
        Debug.Assert(((nint)ptr) % GGML_MEM_ALIGN == 0);
    }

    static void ggml_time_init()
    {
        timer = new Stopwatch();
        timer.Start();
    }
    static long ggml_time_ms()
    {
        return timer.ElapsedMilliseconds;
    }
    static long ggml_time_us()
    {
        return timer.ElapsedTicks * 100;
    }

    static long ggml_cycles_per_ms()
    {
        return CLOCKS_PER_SEC / 1000;
    }

    static long ggml_cycles()
    {
        return Environment.TickCount64;
    }
    static System.Span<TElement> InlineArrayAsSpan<TBuffer, TElement>(ref TBuffer buffer, int size) where TBuffer : struct
    {
        return MemoryMarshal.CreateSpan(ref Unsafe.As<TBuffer, TElement>(ref buffer), size);
    }

    static float GGML_COMPUTE_FP16_TO_FP32(ushort value)
    {
        return (float)(System.Half)value;
    }

    static void ggml_critical_section_start()
    {
        int processing = Interlocked.Exchange(ref g_state_barrier, g_state_barrier + 1);

        while (processing > 0)
        {
            // wait for other threads to finish
            Interlocked.Decrement(ref g_state_barrier);
            //sched_yield(); // TODO: reconsider this
            Thread.Yield();
            processing = Interlocked.Exchange(ref g_state_barrier, g_state_barrier + 1);
        }
    }

    // TODO: make this somehow automatically executed
    //       some sort of "sentry" mechanism
    static void ggml_critical_section_end()
    {
        Interlocked.Decrement(ref g_state_barrier);
    }

    static void GGML_PRINT_DEBUG(string format, params object?[]? args)
    {
        Console.Write(format, args);
    }

    static void GGML_PRINT_DEBUG_5(string format, params object?[]? args)
    {
        Console.Write(format, args);
    }

    static void GGML_PRINT(string format, params object?[]? args)
    {
        Console.Write(format, args);
    }

    static int ggml_thread_create(int* @out, void* unused, delegate*unmanaged <void*, int> func, void* arg)
    {
        throw new NotImplementedException();
    }

    static int ggml_thread_join(int thread, void* unused)
    {
        throw new NotImplementedException();
    }

    [UnmanagedCallersOnly]
    static int ggml_graph_compute_thread(void* data)
    {
        throw new NotImplementedException();
    }
    static void ggml_compute_forward(ggml_compute_params* @params, ggml_tensor* tensor)
    {
        throw new NotImplementedException();
    }
}
