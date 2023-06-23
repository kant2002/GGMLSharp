using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
    const int CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE / sizeof(float);
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
    private static Half[] table_gelu_f16 = new Half[1 << 16];

    // precomputed silu table for f16 (128 KB)
    private static Half[] table_silu_f16 = new Half[1 << 16];

    // precomputed exp table for f16 (128 KB)
    private static Half[] table_exp_f16 = new Half[1 << 16];

    // precomputed f32 table for f16 (256 KB)
    private static float[] table_f32_f16 = new float[1 << 16];

    private static int[] GGML_BLCK_SIZE = new int[(int)ggml_type.GGML_TYPE_COUNT]
    {
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

    private static ulong[] GGML_TYPE_SIZE = new ulong[(int)ggml_type.GGML_TYPE_COUNT]
    {
        /* [GGML_TYPE_F32]  = */ sizeof(float),
        /* [GGML_TYPE_F16]  = */ sizeof(short) /*sizeof(ggml_fp16_t)*/,
        /* [GGML_TYPE_Q4_0] = */ sizeof(float) + QK4_0 / 2 /*sizeof(block_q4_0) */,
        /* [GGML_TYPE_Q4_1] = */ 2 * sizeof(float) + QK4_1 / 2 /*sizeof(block_q4_1)*/,
        /* [GGML_TYPE_Q4_2] = */ sizeof(short) + QK4_2 / 2 /*sizeof(block_q4_2)*/,
        /* [GGML_TYPE_Q4_3] = */ 2 * sizeof(short) + QK4_3 / 2 /*sizeof(block_q4_3)*/,
        /* [GGML_TYPE_Q5_0] = */ sizeof(short) + sizeof(uint) + QK5_0 / 2 /*sizeof(block_q5_0)*/,
        /* [GGML_TYPE_Q5_1] = */ 2 * sizeof(short) + sizeof(uint) + QK5_1 / 2 /*sizeof(block_q5_1)*/,
        /* [GGML_TYPE_Q8_0] = */ sizeof(float) + QK8_0 /*sizeof(block_q8_0)*/,
        /* [GGML_TYPE_Q8_1] = */ 3 * sizeof(float) + QK8_1 /*sizeof(block_q8_1)*/,
        /* [GGML_TYPE_I8]   = */ sizeof(byte),
        /* [GGML_TYPE_I16]  = */ sizeof(short),
        /* [GGML_TYPE_I32]  = */ sizeof(int),
    };

    private static string[] GGML_TYPE_NAME = new string[(int)ggml_type.GGML_TYPE_COUNT]
    {
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

    private static string[] GGML_OP_LABEL = new string[(int)ggml_op.GGML_OP_COUNT]
    {
        "NONE",

        "DUP",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "SQR",
        "SQRT",
        "SUM",
        "MEAN",
        "REPEAT",
        "ABS",
        "SGN",
        "NEG",
        "STEP",
        "RELU",
        "GELU",
        "SILU",
        "NORM",
        "RMS_NORM",

        "MUL_MAT",

        "SCALE",
        "CPY",
        "CONT",
        "RESHAPE",
        "VIEW",
        "PERMUTE",
        "TRANSPOSE",
        "GET_ROWS",
        "DIAG_MASK_INF",
        "SOFT_MAX",
        "ROPE",
        "ALIBI",
        "CONV_1D_1S",
        "CONV_1D_2S",

        "FLASH_ATTN",
        "FLASH_FF",

        "MAP_UNARY",
        "MAP_BINARY",
    };

    private static string[] GGML_OP_SYMBOL = new string[(int)ggml_op.GGML_OP_COUNT]
    {
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

    static bool[] GGML_IS_QUANTIZED = new bool[(int)ggml_type.GGML_TYPE_COUNT]
    {
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

    static quantize_fns_t[] quantize_fns = new quantize_fns_t[(int)ggml_type.GGML_TYPE_COUNT]
    {
        /*[GGML_TYPE_Q4_0] =*/ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q4_0,
            quantize_row_q = &quantize_row_q4_0,
            quantize_row_q_reference = &quantize_row_q4_0_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q4_0_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q4_1] =*/ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q4_1,
            quantize_row_q = &quantize_row_q4_1,
            quantize_row_q_reference = &quantize_row_q4_1_reference,
            quantize_row_q_dot = &quantize_row_q8_1,
            vec_dot_q = &ggml_vec_dot_q4_1_q8_1,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_1,
        },
        /*[GGML_TYPE_Q4_2] = */ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q4_2,
            quantize_row_q = &quantize_row_q4_2,
            quantize_row_q_reference = &quantize_row_q4_2_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q4_2_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q4_3] = */ default,
        /*[GGML_TYPE_Q5_0] = */ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q5_0,
            quantize_row_q = &quantize_row_q5_0,
            quantize_row_q_reference = &quantize_row_q5_0_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q5_0_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q5_1] = */ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q5_1,
            quantize_row_q = &quantize_row_q5_1,
            quantize_row_q_reference = &quantize_row_q5_1_reference,
            quantize_row_q_dot = &quantize_row_q8_1,
            vec_dot_q = &ggml_vec_dot_q5_1_q8_1,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_1,
        },
        /*[GGML_TYPE_Q8_0] = */ new quantize_fns_t
        {
            dequantize_row_q = &dequantize_row_q8_0,
            quantize_row_q = &quantize_row_q8_0,
            quantize_row_q_reference = &quantize_row_q8_0_reference,
            quantize_row_q_dot = &quantize_row_q8_0,
            vec_dot_q = &ggml_vec_dot_q8_0_q8_0,
            vec_dot_type = ggml_type.GGML_TYPE_Q8_0,
        },
        /*[GGML_TYPE_Q8_1] = */ new quantize_fns_t
        {
            dequantize_row_q = null, // TODO
            quantize_row_q = &quantize_row_q8_1,
            quantize_row_q_reference = &quantize_row_q8_1_reference,
            quantize_row_q_dot = &quantize_row_q8_1,
            vec_dot_q = null, // TODO
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

        byte* pp = stackalloc byte[QK4_0 / 2];

        for (int i = 0; i < nb; i++)
        {
            float amax = 0.0f; // absolute max
            float max = 0.0f;

            for (int l = 0; l < QK4_0; l++)
            {
                float v = x[i * QK4_0 + l];
                if (amax < Math.Abs(v))
                {
                    amax = Math.Abs(v);
                    max = v;
                }
            }

            float d = max / -8;
            float id = d != 0.0f ? 1.0f / d : 0.0f;

            y[i].d = d;

            for (int l = 0; l < QK4_0; l += 2)
            {
                float v0 = x[i * QK4_0 + l + 0] * id;
                float v1 = x[i * QK4_0 + l + 1] * id;

                byte vi0 = (byte)Math.Min(15, Math.Round(v0) + 8);
                byte vi1 = (byte)Math.Min(15, Math.Round(v1) + 8);

                Debug.Assert(vi0 < 16);
                Debug.Assert(vi1 < 16);

                pp[l / 2] = (byte)(vi0 | (vi1 << 4));
            }

            Buffer.MemoryCopy(pp, y[i].qs, QK4_0 / 2, QK4_0 / 2);
        }
    }

    [UnmanagedCallersOnly]
    private static void quantize_row_q4_0(float* x, void* vy, int k)
    {
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
                float v0 = x[i * QK8_1 + l] * id;
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
    private unsafe static void dequantize_row_q4_0(void* vx, float* y, int k)
    {
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
                    var vf = new Vector256<float>[4]
                    {
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
                    var vf = new Vector256<float>[4]
                    {
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
    static void dequantize_row_q4_2(void* vx, float* y, int k)
    {
        Debug.Assert(k % QK4_2 == 0);
        int nb = k / QK4_2;

        block_q4_2* x = (block_q4_2*)vx;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)(Half)(x[i].d);

            byte* pp = x[i].qs;

            for (int l = 0; l < QK4_2; l += 2)
            {
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
    static void dequantize_row_q5_0(void* vx, float* y, int k)
    {
        Debug.Assert(k % QK5_0 == 0);
        int nb = k / QK5_0;

        block_q5_0* x = (block_q5_0*)vx;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)(Half)(x[i].d);

            byte* pp = x[i].qs;

            uint qh = *(uint*)x[i].qh;

            for (int l = 0; l < QK5_0; l += 2)
            {
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

        for (int i = 0; i < nb; i++)
        {
            float d = (float)(Half)(x[i].d);
            float m = (float)(Half)(x[i].m);

            byte* pp = x[i].qs;

            uint qh = *(uint*)x[i].qh;

            for (int l = 0; l < QK5_1; l += 2)
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

                sumi += v0 * v1;
            }

            sumf += (x[i].d * y[i].d) * sumi;
        }

        *s = sumf;
    }

    static void ggml_vec_mad_f32(int n, float* y, float* x, float v)
    {
#if GGML_SIMD
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
        // scalar
        for (int i = 0; i < n; ++i)
        {
            y[i] += x[i] * v;
        }
#endif
    }

    static void ggml_vec_scale_f32(int n, float* y, float v)
    {
#if GGML_SIMD
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
        // scalar
        for (int i = 0; i < n; ++i)
        {
            y[i] *= v;
        }
#endif
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
                    table_gelu_f16[i] = (Half)(ggml_gelu_f32(f));
                    table_silu_f16[i] = (Half)(ggml_silu_f32(f));
                    table_exp_f16[i] = (Half)(MathF.Exp(f));
                }

                long t_end = ggml_time_us();

                GGML_PRINT_DEBUG(
                    $"{nameof(ggml_init)}: GELU, SILU and EXP tables initialized in {(t_end - t_start) / 1000.0f} ms\n");
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
            mem_buffer = @params.mem_buffer is not null
                ? @params.mem_buffer
                : NativeMemory.AlignedAlloc((nuint)mem_size, GGML_MEM_ALIGN),
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

                    GGML_PRINT_DEBUG(
                        $"{nameof(ggml_free)}: context {i} with {ctx->n_objects} objects has been freed. memory used = {ctx->objects_end->offs + ctx->objects_end->size}\n");

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


//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

    static ggml_opt_result ggml_opt_adam(
        ggml_context* ctx,
        ggml_opt_params @params,
        ggml_tensor* f,
        ggml_cgraph* gf,
        ggml_cgraph* gb)
    {
        Debug.Assert(ggml_is_scalar(f));

        gf->n_threads = @params.n_threads;
        gb->n_threads = @params.n_threads;

        // these will store the parameters we want to optimize
        ggml_tensor** ps = stackalloc ggml_tensor*[GGML_MAX_PARAMS];

        int np = 0;
        int nx = 0;
        for (int i = 0; i < gf->n_nodes; ++i)
        {
            var node = (ggml_tensor*)gf->nodes[i];
            if (node->is_param)
            {
                GGML_PRINT_DEBUG("found param {0}: grad->op = {1}\n", np, node->grad->op);

                Debug.Assert(np < GGML_MAX_PARAMS);

                ps[np++] = node;
                nx += (int)ggml_nelements(node);
            }
        }

        // constants
        float alpha = @params.adam.alpha;
        float beta1 = @params.adam.beta1;
        float beta2 = @params.adam.beta2;
        float eps = @params.adam.eps;

        float* x = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // view of the parameters
        float* g1 = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // gradient
        float* g2 = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // gradient squared
        float* m = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // first moment
        float* v = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // second moment
        float* mh = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // first moment hat
        float* vh = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // second moment hat

        float* pf = @params.past > 0
            ? (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, @params.past)->data
            : null; // past function values

        // initialize
        ggml_vec_set_f32(nx, m, 0.0f);
        ggml_vec_set_f32(nx, v, 0.0f);

        // update view
        ggml_opt_get_params(np, ps, x);

        // compute the function value
        ggml_graph_reset(gf);
        ggml_set_f32(f->grad, 1.0f);
        ggml_graph_compute(ctx, gb);

        float fx_prev = ggml_get_f32_1d(f, 0);
        if (pf is not null)
        {
            pf[0] = fx_prev;
        }

        int n_no_improvement = 0;
        float fx_best = fx_prev;

        // run the optimizer
        for (int t = 0; t < @params.adam.n_iter; ++t)
        {
            GGML_PRINT_DEBUG($"=== iter {t} ===\n");

            GGML_PRINT_DEBUG("f      = {0,10:F6}\n", ggml_get_f32_1d(f, 0));
            GGML_PRINT_DEBUG_5("df/dx0 = {0,10:F6}\n", ggml_get_f32_1d(ps[0]->grad, 0));
            GGML_PRINT_DEBUG_5("df/dx1 = {0,10:F6}\n", ggml_get_f32_1d(ps[1]->grad, 0));

            for (int i = 0; i < np; ++i)
            {
                GGML_PRINT_DEBUG("param {0}: {1,10:F6}, g = {2,10:F6}\n", i,
                    ggml_get_f32_1d(ps[i], 0), ggml_get_f32_1d(ps[i]->grad, 0));
            }

            long t_start_wall = ggml_time_us();
            long t_start_cpu = ggml_cycles();

            {
                // update the gradient
                ggml_opt_get_grad(np, ps, g1);

                // m_t = beta1*m_t-1 + (1 - beta1)*g_t
                ggml_vec_scale_f32(nx, m, beta1);
                ggml_vec_mad_f32(nx, m, g1, 1.0f - beta1);

                // g2 = g1^2
                ggml_vec_sqr_f32(nx, g2, g1);

                // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
                ggml_vec_scale_f32(nx, v, beta2);
                ggml_vec_mad_f32(nx, v, g2, 1.0f - beta2);

                // m^hat = m_t / (1 - beta1^t)
                // v^hat = v_t / (1 - beta2^t)
                // x_t = x_t-1 - alpha*m^hat/(sqrt(v^hat) + eps)
                ggml_vec_cpy_f32(nx, mh, m);
                ggml_vec_cpy_f32(nx, vh, v);

                ggml_vec_scale_f32(nx, mh, alpha / (1.0f - MathF.Pow(beta1, t + 1)));
                ggml_vec_scale_f32(nx, vh, 1.0f / (1.0f - MathF.Pow(beta2, t + 1)));

                ggml_vec_sqrt_f32(nx, vh, vh);
                ggml_vec_acc1_f32(nx, vh, eps);

                ggml_vec_div_f32(nx, mh, mh, vh);
                ggml_vec_sub_f32(nx, x, x, mh);

                // update the parameters
                ggml_opt_set_params(np, ps, x);
            }

            ggml_graph_reset(gf);
            ggml_set_f32(f->grad, 1.0f);
            ggml_graph_compute(ctx, gb);

            float fx = ggml_get_f32_1d(f, 0);

            // check convergence
            if (MathF.Abs(fx - fx_prev) / fx < @params.adam.eps_f)
            {
                GGML_PRINT_DEBUG("converged\n");

                return ggml_opt_result.GGML_OPT_OK;
            }

            // delta-based convergence test
            if (pf != null)
            {
                // need at least params.past iterations to start checking for convergence
                if (@params.past <= t)
                {
                    float rate = (pf[t % @params.past] - fx) / fx;

                    if (MathF.Abs(rate) < @params.delta)
                    {
                        return ggml_opt_result.GGML_OPT_OK;
                    }
                }

                pf[t % @params.past] = fx;
            }

            // check for improvement
            if (@params.max_no_improvement > 0)
            {
                if (fx_best > fx)
                {
                    fx_best = fx;
                    n_no_improvement = 0;
                }
                else
                {
                    ++n_no_improvement;

                    if (n_no_improvement >= @params.max_no_improvement)
                    {
                        return ggml_opt_result.GGML_OPT_OK;
                    }
                }
            }

            fx_prev = fx;

            {
                long t_end_cpu = ggml_cycles();
                GGML_PRINT_DEBUG("time iter:      {0,5:F3} s\n", ((float)(t_end_cpu - t_start_cpu)) / CLOCKS_PER_SEC);

                long t_end_wall = ggml_time_us();
                GGML_PRINT_DEBUG("wall time iter: {0,5:F3} s\n", (t_end_wall - t_start_wall) / 1e6);
            }
        }

        return ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE;
    }

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

    struct ggml_lbfgs_iteration_data
    {
        public float alpha;
        public float ys;
        public float* s;
        public float* y;
    };

    static ggml_opt_result linesearch_backtracking(
        ggml_context* ctx,
        ggml_opt_params* @params,
        int nx,
        float* x,
        float* fx,
        float* g,
        float* d,
        float* step,
        float* xp,
        ggml_tensor* f,
        ggml_cgraph* gf,
        ggml_cgraph* gb,
        int np,
        ggml_tensor** ps)
    {
        int count = 0;

        float width = 0.0f;
        float dg = 0.0f;
        float finit = 0.0f;
        float dginit = 0.0f;
        float dgtest = 0.0f;

        const float dec = 0.5f;
        const float inc = 2.1f;

        if (*step <= 0.0f)
        {
            return ggml_opt_result.GGML_LINESEARCH_INVALID_PARAMETERS;
        }

        // compute the initial gradient in the search direction
        ggml_vec_dot_f32(nx, &dginit, g, d);

        // make sure that d points to a descent direction
        if (0 < dginit)
        {
            return ggml_opt_result.GGML_LINESEARCH_FAIL;
        }

        // initialize local variables
        finit = *fx;
        dgtest = @params->lbfgs.ftol * dginit;

        while (true)
        {
            ggml_vec_cpy_f32(nx, x, xp);
            ggml_vec_mad_f32(nx, x, d, *step);

            // evaluate the function and gradient values
            {
                ggml_opt_set_params(np, ps, x);

                ggml_graph_reset(gf);
                ggml_set_f32(f->grad, 1.0f);
                ggml_graph_compute(ctx, gb);

                ggml_opt_get_grad(np, ps, g);

                *fx = ggml_get_f32_1d(f, 0);
            }

            ++count;

            if (*fx > finit + (*step) * dgtest)
            {
                width = dec;
            }
            else
            {
                // Armijo condition is satisfied
                if (@params->lbfgs.linesearch == ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_ARMIJO)
                {
                    return (ggml_opt_result)count;
                }

                ggml_vec_dot_f32(nx, &dg, g, d);

                // check the Wolfe condition
                if (dg < @params->lbfgs.wolfe * dginit)
                {
                    width = inc;
                }
                else
                {
                    if (@params->lbfgs.linesearch == ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_WOLFE)
                    {
                        // regular Wolfe conditions
                        return (ggml_opt_result)count;
                    }

                    if (dg > -@params->lbfgs.wolfe * dginit)
                    {
                        width = dec;
                    }
                    else
                    {
                        // strong Wolfe condition (GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                        return (ggml_opt_result)count;
                    }

                    return (ggml_opt_result)count;
                }
            }

            if (*step < @params->lbfgs.min_step)
            {
                return ggml_opt_result.GGML_LINESEARCH_MINIMUM_STEP;
            }

            if (*step > @params->lbfgs.max_step)
            {
                return ggml_opt_result.GGML_LINESEARCH_MAXIMUM_STEP;
            }

            if (@params->lbfgs.max_linesearch <= count)
            {
                return ggml_opt_result.GGML_LINESEARCH_MAXIMUM_ITERATIONS;
            }

            (*step) *= width;
        }

        return ggml_opt_result.GGML_LINESEARCH_FAIL;
    }

    static ggml_opt_result ggml_opt_lbfgs(
        ggml_context* ctx,
        ggml_opt_params @params,
        ggml_tensor* f,
        ggml_cgraph* gf,
        ggml_cgraph* gb)
    {
        if (@params.lbfgs.linesearch == ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_WOLFE ||
            @params.lbfgs.linesearch == ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
        {
            if (@params.lbfgs.wolfe <= @params.lbfgs.ftol || 1.0f <= @params.lbfgs.wolfe)
            {
                return ggml_opt_result.GGML_OPT_INVALID_WOLFE;
            }
        }

        gf->n_threads = @params.n_threads;
        gb->n_threads = @params.n_threads;

        int m = @params.lbfgs.m;

        // these will store the parameters we want to optimize
        ggml_tensor** ps = stackalloc ggml_tensor*[GGML_MAX_PARAMS];

        int np = 0;
        int nx = 0;
        for (int i = 0; i < gf->n_nodes; ++i)
        {
            var node = ggml_cgraph.get_node(gf, i);
            if (node->is_param)
            {
                GGML_PRINT_DEBUG("found param {0}: grad->op = {1}\n", np, node->grad->op);

                Debug.Assert(np < GGML_MAX_PARAMS);

                ps[np++] = node;
                nx += (int)ggml_nelements(node);
            }
        }

        float* x = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // current parameters
        float* xp = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // previous parameters
        float* g = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // current gradient
        float* gp = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // previous gradient
        float* d = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data; // search direction

        float* pf = @params.past > 0
            ? (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, @params.past)->data
            : null; // past function values

        float fx = 0.0f; // cost function value
        float xnorm = 0.0f; // ||x||
        float gnorm = 0.0f; // ||g||
        float step = 0.0f;

        // initialize x from the graph nodes
        ggml_opt_get_params(np, ps, x);

        // the L-BFGS memory
        ggml_lbfgs_iteration_data* lm = stackalloc ggml_lbfgs_iteration_data[m];

        for (int i = 0; i < m; ++i)
        {
            lm[i].alpha = 0.0f;
            lm[i].ys = 0.0f;
            lm[i].s = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data;
            lm[i].y = (float*)ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, nx)->data;
        }

        // evaluate the function value and its gradient
        {
            ggml_opt_set_params(np, ps, x);

            ggml_graph_reset(gf);
            ggml_set_f32(f->grad, 1.0f);
            ggml_graph_compute(ctx, gb);

            ggml_opt_get_grad(np, ps, g);

            fx = ggml_get_f32_1d(f, 0);
        }

        if (pf is not null)
        {
            pf[0] = fx;
        }

        float fx_best = fx;

        // search direction = -gradient
        ggml_vec_neg_f32(nx, d, g);

        // ||x||, ||g||
        ggml_vec_norm_f32(nx, &xnorm, x);
        ggml_vec_norm_f32(nx, &gnorm, g);

        if (xnorm < 1.0f)
        {
            xnorm = 1.0f;
        }

        // already optimized
        if (gnorm / xnorm <= @params.lbfgs.eps)
        {
            return ggml_opt_result.GGML_OPT_OK;
        }

        // initial step
        ggml_vec_norm_inv_f32(nx, &step, d);
        

        int j = 0;
        int k = 1;
        int ls = 0;
        int end = 0;
        int bound = 0;
        int n_no_improvement = 0;

        float ys = 0.0f;
        float yy = 0.0f;
        float beta = 0.0f;

        while (true)
        {
            // store the current position and gradient vectors
            ggml_vec_cpy_f32(nx, xp, x);
            ggml_vec_cpy_f32(nx, gp, g);

            ls = (int)linesearch_backtracking(ctx, &@params, nx, x, &fx, g, d, &step, xp, f, gf, gb, np, ps);

            if (ls < 0)
            {
                // linesearch failed - go back to the previous point and return
                ggml_vec_cpy_f32(nx, x, xp);
                ggml_vec_cpy_f32(nx, g, gp);

                return (ggml_opt_result)ls;
            }

            ggml_vec_norm_f32(nx, &xnorm, x);
            ggml_vec_norm_f32(nx, &gnorm, g);

            GGML_PRINT_DEBUG("f = {0,10:F6}\n", ggml_get_f32_1d(f, 0));

            if (xnorm < 1.0f)
            {
                xnorm = 1.0f;
            }

            if (gnorm / xnorm <= @params.lbfgs.eps)
            {
                // converged
                return ggml_opt_result.GGML_OPT_OK;
            }

            // delta-based convergence test
            if (pf != null)
            {
                // need at least params.past iterations to start checking for convergence
                if (@params.past <= k)
                {
                    float rate = (pf[k % @params.past] - fx) / fx;

                    if (MathF.Abs(rate) < @params.delta)
                    {
                        return ggml_opt_result.GGML_OPT_OK;
                    }
                }

                pf[k % @params.past] = fx;
            }

            // check for improvement
            if (@params.max_no_improvement > 0)
            {
                if (fx < fx_best)
                {
                    fx_best = fx;
                    n_no_improvement = 0;
                }
                else
                {
                    n_no_improvement++;

                    if (n_no_improvement >= @params.max_no_improvement)
                    {
                        return ggml_opt_result.GGML_OPT_OK;
                    }
                }
            }

            if (@params.lbfgs.n_iter != 0 && @params.lbfgs.n_iter < k + 1)
            {
                // reached the maximum number of iterations
                return ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE;
            }

            // update vectors s and y:
            //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            //   y_{k+1} = g_{k+1} - g_{k}.
            //
            ggml_vec_sub_f32(nx, lm[end].s, x, xp);
            ggml_vec_sub_f32(nx, lm[end].y, g, gp);

            // compute scalars ys and yy:
            //     ys = y^t \cdot s    -> 1 / \rho.
            //     yy = y^t \cdot y.
            //
            ggml_vec_dot_f32(nx, &ys, lm[end].y, lm[end].s);
            ggml_vec_dot_f32(nx, &yy, lm[end].y, lm[end].y);

            lm[end].ys = ys;

            // find new search direction
            //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

            bound = (m <= k) ? m : k;
            k++;
            end = (end + 1) % m;

            // initialize search direction with -g
            ggml_vec_neg_f32(nx, d, g);

            j = end;
            for (int i = 0; i < bound; ++i)
            {
                j = (j + m - 1) % m;
                // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
                ggml_vec_dot_f32(nx, &lm[j].alpha, lm[j].s, d);
                lm[j].alpha /= lm[j].ys;
                // q_{i} = q_{i+1} - \alpha_{i} y_{i}
                ggml_vec_mad_f32(nx, d, lm[j].y, -lm[j].alpha);
            }

            ggml_vec_scale_f32(nx, d, ys / yy);

            for (int i = 0; i < bound; ++i)
            {
                // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
                ggml_vec_dot_f32(nx, &beta, lm[j].y, d);
                beta /= lm[j].ys;
                // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
                ggml_vec_mad_f32(nx, d, lm[j].s, lm[j].alpha - beta);
                j = (j + 1) % m;
            }

            step = 1.0f;
        }

        return ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE;
    }

    public static ggml_opt_params ggml_opt_default_params(ggml_opt_type type)
    {
        ggml_opt_params result = default;

        switch (type)
        {
            case ggml_opt_type.GGML_OPT_ADAM:
            {
                result = new ggml_opt_params
                {
                    type = ggml_opt_type.GGML_OPT_ADAM,
                    n_threads = 1,
                    past = 0,
                    delta = 1e-5f,

                    max_no_improvement = 100,

                    print_forward_graph = true,
                    print_backward_graph = true,

                    adam = new()
                    {
                        n_iter = 10000,
                        alpha = 0.001f,
                        beta1 = 0.9f,
                        beta2 = 0.999f,
                        eps = 1e-8f,
                        eps_f = 1e-5f,
                        eps_g = 1e-3f,
                    },
                };
            }
                break;
            case ggml_opt_type.GGML_OPT_LBFGS:
            {
                result = new ggml_opt_params
                {
                    type = ggml_opt_type.GGML_OPT_LBFGS,
                    n_threads = 1,
                    past = 0,
                    delta = 1e-5f,

                    max_no_improvement = 0,

                    print_forward_graph = true,
                    print_backward_graph = true,

                    lbfgs = new()
                    {
                        m = 6,
                        n_iter = 100,
                        max_linesearch = 20,

                        eps = 1e-5f,
                        ftol = 1e-4f,
                        wolfe = 0.9f,
                        min_step = 1e-20f,
                        max_step = 1e+20f,

                        linesearch = ggml_linesearch.GGML_LINESEARCH_DEFAULT,
                    },
                };
            }
                break;
            default:
                Debug.Assert(false);
                break;
        }

        return result;
    }

    public static ggml_opt_result ggml_opt(
        ggml_context* ctx,
        ggml_opt_params @params,
        ggml_tensor* f)
    {
        bool free_ctx = false;
        if (ctx == null)
        {
            ggml_init_params params_ctx = new ggml_init_params
            {
                mem_size = 16 * 1024 * 1024,
                mem_buffer = null,
                no_alloc = false,
            };

            ctx = ggml_init(params_ctx);
            if (ctx == null)
            {
                return ggml_opt_result.GGML_OPT_NO_CONTEXT;
            }

            free_ctx = true;
        }

        ggml_opt_result result = ggml_opt_result.GGML_OPT_OK;

        // build forward + backward compute graphs
        ggml_cgraph gf = ggml_build_forward(f);
        ggml_cgraph gb = ggml_build_backward(ctx, &gf, false);

        switch (@params.type)
        {
            case ggml_opt_type.GGML_OPT_ADAM:
            {
                result = ggml_opt_adam(ctx, @params, f, &gf, &gb);
            }
                break;
            case ggml_opt_type.GGML_OPT_LBFGS:
            {
                result = ggml_opt_lbfgs(ctx, @params, f, &gf, &gb);
            }
                break;
        }

        if (@params.print_forward_graph)
        {
            ggml_graph_print(&gf);
            ggml_graph_dump_dot(&gf, null, "opt-forward.dot");
        }

        if (@params.print_backward_graph)
        {
            ggml_graph_print(&gb);
            ggml_graph_dump_dot(&gb, &gf, "opt-backward.dot");
        }

        if (free_ctx)
        {
            ggml_free(ctx);
        }

        return result;
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

        ggml_tensor* result = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_F32, 1);

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

    public static void ggml_vec_set_i8(int n, byte* x, byte v)
    {
        for (int i = 0; i < n; ++i) x[i] = v;
    }

    public static void ggml_vec_set_i16(int n, short* x, short v)
    {
        for (int i = 0; i < n; ++i) x[i] = v;
    }

    public static void ggml_vec_set_i32(int n, int* x, int v)
    {
        for (int i = 0; i < n; ++i) x[i] = v;
    }

    public static void ggml_vec_set_f16(int n, Half* x, int v)
    {
        for (int i = 0; i < n; ++i) x[i] = (Half)v;
    }

    static void ggml_vec_add_f32(int n, float* z, float* x, float* y)
    {
        for (int i = 0; i < n; ++i) z[i] = x[i] + y[i];
    }

    static void ggml_vec_acc_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] += x[i];
    }

    static void ggml_vec_acc1_f32(int n, float* y, float v)
    {
        for (int i = 0; i < n; ++i) y[i] += v;
    }

    static void ggml_vec_sub_f32(int n, float* z, float* x, float* y)
    {
        for (int i = 0; i < n; ++i) z[i] = x[i] - y[i];
    }

    static void ggml_vec_set_f32(int n, float* y, float v)
    {
        for (int i = 0; i < n; ++i) y[i] = v;
    }

    static void ggml_vec_cpy_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = x[i];
    }

    static void ggml_vec_neg_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = -x[i];
    }

    static void ggml_vec_mul_f32(int n, float* z, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) z[i] = x[i] * y[i];
    }

    static void ggml_vec_div_f32(int n, float* z, float* x, float* y)
    {
        for (int i = 0; i < n; ++i) z[i] = x[i] / y[i];
    }

    static void ggml_vec_dot_f32(int n, float* s, float* x, float* y)
    {
        double sumf = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sumf += (x[i] * y[i]);
        }

        *s = (float)sumf;
    }

    static void ggml_vec_dot_f16(int n, float* s, Half* x, Half* y)
    {
        double sumf = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sumf += (float)x[i] * (float)y[i];
        }

        *s = (float)sumf;
    }

    static void ggml_vec_norm_f32(int n, float* s, float* x)
    {
        ggml_vec_dot_f32(n, s, x, x);
        *s = MathF.Sqrt(*s);
    }

    static void ggml_vec_sqr_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = x[i] * x[i];
    }

    static void ggml_vec_sqrt_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = MathF.Sqrt(x[i]);
    }

    static void ggml_vec_abs_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = Math.Abs(x[i]);
    }

    static void ggml_vec_sgn_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.0f) ? 1.0f : ((x[i] < 0.0f) ? -1.0f : 0.0f);
    }

    static void ggml_vec_step_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
    }

    static void ggml_vec_relu_f32(int n, float* y, float* x)
    {
        for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }

    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    static float ggml_gelu_f32(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }

    static void ggml_vec_gelu_f16(int n, Half* x, Half* y)
    {
        short* i16 = (short*)x;
        for (int i = 0; i < n; ++i)
        {
            y[i] = table_gelu_f16[i16[i]];
        }
    }
#if GGML_GELU_FP16
    static void ggml_vec_gelu_f32(int n, float* x, float* y)
    {
        ushort t;
        for (int i = 0; i < n; ++i)
        {
            Half fp16 = (Half)(x[i]);
            NativeMemory.Copy(&fp16, &t, sizeof(ushort));
            y[i] = (float)(table_gelu_f16[t]);
        }
    }
#else
    static void ggml_vec_gelu_f32(int n, float* x, float* y) {
        for (int i = 0; i < n; ++i) {
            y[i] = ggml_gelu_f32(x[i]);
        }
    }
#endif
    static float ggml_silu_f32(float x)
    {
        return x / (1.0f + MathF.Exp(-x));
    }

    static void ggml_vec_silu_f16(int n, Half* x, Half* y)
    {
        short* i16 = (short*)x;
        for (int i = 0; i < n; ++i)
        {
            y[i] = table_silu_f16[i16[i]];
        }
    }
#if GGML_SILU_FP16
    static void ggml_vec_silu_f32(int n, float* x, float* y)
    {
        ushort t;
        for (int i = 0; i < n; ++i)
        {
            Half fp16 = (Half)(x[i]);
            NativeMemory.Copy(&fp16, &t, sizeof(ushort));
            y[i] = (float)(table_silu_f16[t]);
        }
    }
#else
    static void ggml_vec_silu_f32(int n, float* x, float* y) {
        for (int i = 0; i < n; ++i) {
            y[i] = ggml_silu_f32(x[i]);
        }
    }
#endif

    static void ggml_vec_sum_f32(int n, float* s, float* x)
    {
#if !GGML_USE_ACCELERATE
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sum += (double)x[i];
        }

        *s = (float)sum;
#else
        vDSP_sve(x, 1, s, n);
#endif
    }

    static void ggml_vec_sum_ggf(int n, double* s, float* x)
    {
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sum += (double)x[i];
        }

        *s = sum;
    }

    static void ggml_vec_max_f32(int n, float* s, float* x)
    {
#if !GGML_USE_ACCELERATE
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; ++i)
        {
            max = MathF.Max(max, x[i]);
        }

        *s = max;
#else
        vDSP_maxv(x, 1, s, n);
#endif
    }

    static void ggml_vec_norm_inv_f32(int n, float* s, float* x)
    {
        ggml_vec_norm_f32(n, s, x);
        *s = 1.0f / (*s);
    }

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

    static void ggml_set_f32_1d(ggml_tensor* tensor, int i, float value)
    {
        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
            {
                Debug.Assert(tensor->nb[0] == sizeof(byte));
                ((byte*)(tensor->data))[i] = (byte)value;
            }
                break;
            case ggml_type.GGML_TYPE_I16:
            {
                Debug.Assert(tensor->nb[0] == sizeof(short));
                ((short*)(tensor->data))[i] = (short)value;
            }
                break;
            case ggml_type.GGML_TYPE_I32:
            {
                Debug.Assert(tensor->nb[0] == sizeof(int));
                ((int*)(tensor->data))[i] = (int)value;
            }
                break;
            case ggml_type.GGML_TYPE_F16:
            {
                Debug.Assert((int)tensor->nb[0] == sizeof(Half));
                ((Half*)(tensor->data))[i] = (Half)(value);
            }
                break;
            case ggml_type.GGML_TYPE_F32:
            {
                Debug.Assert(tensor->nb[0] == sizeof(float));
                ((float*)(tensor->data))[i] = value;
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_graph_print(ggml_cgraph* cgraph)
    {
        long[] perf_total_per_op_us = new long[(int)ggml_op.GGML_OP_COUNT];

        GGML_PRINT("=== GRAPH ===\n");

        GGML_PRINT_DEBUG("n_threads       = {0}\n", cgraph->n_threads);
        GGML_PRINT_DEBUG("total work size = {0} bytes\n", cgraph->work_size);

        GGML_PRINT("n_nodes = {0}\n", cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            ggml_tensor* node = ggml_cgraph.get_node(cgraph, i);

            perf_total_per_op_us[(int)node->op] += Math.Max(1, node->perf_time_us);

            GGML_PRINT(
                " - {0,3}: [{1,5:X}, {2,5:X}, {3,5:X}] {4,16} {5} ({6,3}) cpu = {7,7:F3} / {8,7:F3} ms, wall = {9,7:F3} / {10,7:F3} ms\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                GGML_OP_LABEL[(int)node->op], node->is_param ? "x" : node->grad is not null ? "g" : " ",
                node->perf_runs,
                (double)node->perf_cycles / (double)ggml_cycles_per_ms(),
                (double)node->perf_cycles / (double)ggml_cycles_per_ms() / (double)node->perf_runs,
                (double)node->perf_time_us / 1000.0,
                (double)node->perf_time_us / 1000.0 / node->perf_runs);
        }

        GGML_PRINT("n_leafs = {0}\n", cgraph->n_leafs);
        for (int i = 0; i < cgraph->n_leafs; i++)
        {
            ggml_tensor* node = ggml_cgraph.get_leaf(cgraph, i);

            GGML_PRINT($" - {i,3}: [ {node->ne[0]}, {node->ne[1]}] {GGML_OP_LABEL[(int)node->op],8}\n");
        }

        for (int i = 0; i < (int)ggml_op.GGML_OP_COUNT; i++)
        {
            if (perf_total_per_op_us[i] == 0)
            {
                continue;
            }

            GGML_PRINT(
                $"perf_total_per_op_us[{GGML_OP_LABEL[i],-16}] = {(perf_total_per_op_us[i] / 1000.0),7:F3} ms\n");
        }

        GGML_PRINT("========================================\n");
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
            ggml_tensor* current_node = ggml_cgraph.get_node(cgraph, i);
            if (current_node == node)
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
            ggml_tensor* parent = ggml_cgraph.get_node(cgraph, i);

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

    private static void ggml_lock_init(int* spin)
    {
    }

    private static void ggml_lock_destroy(int* spin)
    {
    }

    private static void ggml_lock_lock(int* spin)
    {
    }

    private static void ggml_lock_unlock(int* spin)
    {
    }

    public static void ggml_graph_dump_dot(ggml_cgraph* gb, ggml_cgraph* gf, string filename)
    {
        string color;

        using var fp = new StreamWriter(File.OpenWrite(filename));

        fp.WriteLine("digraph G {");
        fp.WriteLine("  newrank = true;");
        fp.WriteLine("  rankdir = LR;");

        for (int i = 0; i < gb->n_nodes; i++)
        {
            ggml_tensor* node = ggml_cgraph.get_node(gb, i);

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

            fp.WriteLine(
                $"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"{i} [{node->ne[0]}, {node->ne[1]} | <x>{GGML_OP_SYMBOL[(int)node->op]}");

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
            ggml_tensor* node = ggml_cgraph.get_leaf(gb, i);

            color = "pink";

            if (ggml_nelements(node) == 1)
            {
                fp.WriteLine(
                    $"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>{(double)ggml_get_f32_1d(node, 0)}\"; ]");
            }
            else
            {
                fp.WriteLine(
                    $"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>CONST {i} [{node->ne[0]}, {node->ne[1]}]\"; ]");
            }
        }

        for (int i = 0; i < gb->n_nodes; i++)
        {
            ggml_tensor* node = ggml_cgraph.get_node(gb, i);

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
            ggml_tensor* node = ggml_cgraph.get_leaf(gb, i);

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

    ////////////////////////////////////////////////////////////////////////////////

    static void ggml_opt_set_params(int np, ggml_tensor** ps, float* x)
    {
        int i = 0;
        for (int p = 0; p < np; ++p)
        {
            long ne = ggml_nelements(ps[p]);
            // TODO: add function to set tensor from array
            for (long j = 0; j < ne; ++j)
            {
                ggml_set_f32_1d(ps[p], (int)j, x[i++]);
            }
        }
    }

    static void ggml_opt_get_params(int np, ggml_tensor** ps, float* x)
    {
        int i = 0;
        for (int p = 0; p < np; ++p)
        {
            long ne = ggml_nelements(ps[p]);
            // TODO: add function to get all elements at once
            for (long j = 0; j < ne; ++j)
            {
                x[i++] = ggml_get_f32_1d(ps[p], (int)j);
            }
        }
    }

    static void ggml_opt_get_grad(int np, ggml_tensor** ps, float* g)
    {
        int i = 0;
        for (int p = 0; p < np; ++p)
        {
            long ne = ggml_nelements(ps[p]);
            // TODO: add function to get all elements at once
            for (long j = 0; j < ne; ++j)
            {
                g[i++] = ggml_get_f32_1d(ps[p]->grad, (int)j);
            }
        }
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

        ggml_compute_state_shared state_shared = new ggml_compute_state_shared
        {
            spin = GGML_LOCK_INITIALIZER,
            n_threads = n_threads,
            n_ready = 0,
            has_work = 0,
            stop = 0,
        };
        //ggml_compute_state* workers = n_threads > 1 ? stackalloc ggml_compute_state[(n_threads - 1)] : null;
        ggml_compute_state[] workers = n_threads > 1 ? new ggml_compute_state[n_threads - 1] : null!;

        // create thread pool
        if (n_threads > 1)
        {
            ggml_lock_init(&state_shared.spin);

            atomic_store(ref state_shared.has_work, 1);

            for (int j = 0; j < n_threads - 1; j++)
            {
                workers[j] = new ggml_compute_state
                {
                    thrd = null!,
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

                workers[j].thrd = new Thread(ggml_graph_compute_thread);
                workers[j].thrd.Start(workers[j]);
                //int rc = ggml_thread_create(&workers[j].thrd, null, &ggml_graph_compute_thread, &workers[j]);
                //Debug.Assert(rc == 0);
            }
        }

        // initialize tasks + work buffer
        {
            nuint work_size = 0;

            // thread scheduling for the different operations
            for (int i = 0; i < cgraph->n_nodes; i++)
            {
                ggml_tensor* node = ggml_cgraph.get_node(cgraph, i);

                switch (node->op)
                {
                    case ggml_op.GGML_OP_CPY:
                    case ggml_op.GGML_OP_DUP:
                    {
                        node->n_tasks = n_threads;

                        nuint cur = 0;
                        if (ggml_is_quantized(node->type))
                        {
                            cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F32] * (nuint)node->ne[0] *
                                  (nuint)n_threads;
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
                            cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F32] * (nuint)node->src0->ne[0] *
                                  (nuint)n_threads;
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
                            cur = (nuint)GGML_TYPE_SIZE[(int)ggml_type.GGML_TYPE_F16] *
                                  (nuint)ggml_nelements(node->src1);
#endif
                        }
                        else if (node->src0->type == ggml_type.GGML_TYPE_F32 &&
                                 node->src1->type == ggml_type.GGML_TYPE_F32)
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
                                cur = (nuint)GGML_TYPE_SIZE[(int)type_q] * (nuint)ggml_nelements(node->src1) /
                                      (nuint)GGML_BLCK_SIZE[(int)type_q];
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
                            cur = (nuint)(sizeof(float) * node->src1->ne[1] *
                                          node->n_tasks); // TODO: this can become (n_tasks-1)
                            cur += (nuint)(sizeof(float) * node->src1->ne[1] *
                                           node->n_tasks); // this is overestimated by x2
                        }

                        if (node->src1->type == ggml_type.GGML_TYPE_F16)
                        {
                            cur = (nuint)(sizeof(float) * node->src1->ne[1] *
                                          node->n_tasks); // TODO: this can become (n_tasks-1)
                            cur += (nuint)(sizeof(float) * node->src1->ne[1] *
                                           node->n_tasks); // this is overestimated by x2
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

                GGML_PRINT_DEBUG(
                    $"{nameof(ggml_graph_compute)}: allocating work buffer for graph ({cgraph->work_size} bytes)\n");
                cgraph->work = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I8, (long)cgraph->work_size);
            }
        }

        long perf_start_cycles = ggml_perf_cycles();
        long perf_start_time_us = ggml_perf_time_us();

        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            GGML_PRINT_DEBUG_5($"{nameof(ggml_graph_compute)}: {i}/{cgraph->n_nodes}\n");

            ggml_tensor* node = ggml_cgraph.get_node(cgraph, i);

            // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
            //if (node->grad == NULL && node->perf_runs > 0) {
            //    continue;
            //}

            long perf_node_start_cycles = ggml_perf_cycles();
            long perf_node_start_time_us = ggml_perf_time_us();

            // INIT
            ggml_compute_params @params = new ggml_compute_params
            {
                type = ggml_task_type.GGML_TASK_INIT,
                ith = 0,
                nth = node->n_tasks,
                wsize = cgraph->work is not null ? ggml_nbytes(cgraph->work) : 0,
                wdata = cgraph->work is not null ? cgraph->work->data : null,
            };

            ggml_compute_forward(&@params, node);

            // COMPUTE
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
                    workers[j].@params = new ggml_compute_params
                    {
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
                workers[j].thrd.Join();
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

            GGML_PRINT_DEBUG(
                $"{nameof(ggml_graph_compute)}: perf ({cgraph->perf_runs}) - cpu = {0:F3} / {1:F3} ms, wall = {2:F3} / {3:F3} ms\n",
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
            ggml_tensor* grad = ggml_cgraph.get_grad(cgraph, i);

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

        return ((nuint)ggml_nelements(tensor) * (nuint)GGML_TYPE_SIZE[(int)tensor->type]) /
               (nuint)GGML_BLCK_SIZE[(int)tensor->type];
    }

    public static void ggml_set_param(
        ggml_context* ctx,
        ggml_tensor* tensor)
    {
        tensor->is_param = true;

        Debug.Assert(tensor->grad == null);
        tensor->grad = ggml_dup_tensor(ctx, tensor);
    }

    private static void ggml_compute_forward_dup_f16(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_nelements(dst) == ggml_nelements(src0));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

        long ne0 = dst->ne[0];
        long ne1 = dst->ne[1];
        long ne2 = dst->ne[2];
        long ne3 = dst->ne[3];

        ulong nb00 = src0->nb[0];
        ulong nb01 = src0->nb[1];
        ulong nb02 = src0->nb[2];
        ulong nb03 = src0->nb[3];

        ulong nb0 = dst->nb[0];
        ulong nb1 = dst->nb[1];
        ulong nb2 = dst->nb[2];
        ulong nb3 = dst->nb[3];

        int ith = @params->ith; // thread index
        int nth = @params->nth; // number of threads
        int dr;
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type)
        {
            // parallelize by elements
            long ne = ggml_nelements(dst);
            dr = (int)((ne + nth - 1) / nth);
            int ie0 = dr * ith;
            int ie1 = Math.Min(ie0 + dr, (int)ne);

            NativeMemory.Copy(
                ((byte*)dst->data + ie0 * (int)nb0),
                ((byte*)src0->data + ie0 * (int)nb00),
                (nuint)(ie1 - ie0) * (nuint)GGML_TYPE_SIZE[(int)src0->type]);

            return;
        }

        // parallelize by rows
        int nr = (int)ne01;
        // number of rows per thread
        dr = (nr + nth - 1) / nth;
        // row range for this thread
        int ir0 = dr * ith;
        int ir1 = Math.Min(ir0 + dr, (int)nr);

        if (src0->type == dst->type &&
            ne00 == ne0 &&
            nb00 == GGML_TYPE_SIZE[(int)src0->type] && nb0 == GGML_TYPE_SIZE[(int)dst->type])
        {
            // copy by rows
            nuint rs = (nuint)ne00 * (nuint)nb00;
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        NativeMemory.Copy(
                            ((byte*)dst->data + (nuint)i01 * (nuint)nb1 + (nuint)i02 * (nuint)nb2 +
                             (nuint)i03 * (nuint)nb3),
                            ((byte*)src0->data + (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                             (nuint)i03 * (nuint)nb03),
                            rs);
                    }
                }
            }

            return;
        }

        // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

        if (ggml_is_contiguous(dst))
        {
            if (nb00 == (ulong)sizeof(Half))
            {
                if (dst->type == ggml_type.GGML_TYPE_F16)
                {
                    nuint id = 0;
                    nuint rs = (nuint)ne00 * (nuint)nb00;
                    byte* dst_ptr = (byte*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)rs * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                byte* src0_ptr = (byte*)src0->data + (nuint)i01 * (nuint)nb01 +
                                                 (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03;
                                NativeMemory.Copy(dst_ptr + id, src0_ptr, rs);
                                id += rs;
                            }

                            id += rs * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (dst->type == ggml_type.GGML_TYPE_F32)
                {
                    nuint id = 0;
                    float* dst_ptr = (float*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                Half* src0_ptr = (Half*)((byte*)src0->data + (nuint)i01 * (nuint)nb01 +
                                                         (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    dst_ptr[id] = (float)src0_ptr[i00];
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (ggml_is_quantized(dst->type))
                {
                    var quantize_row_q = quantize_fns[(int)dst->type].quantize_row_q;
                    float* src0_f32 = (float*)@params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                    nuint id = 0;
                    nuint rs = (nuint)nb0 * (nuint)(ne00 / GGML_BLCK_SIZE[(int)dst->type]);
                    char* dst_ptr = (char*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += rs * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                Half* src0_ptr = (Half*)((byte*)src0->data + (nuint)i01 * (nuint)nb01 +
                                                         (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);

                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    src0_f32[i00] = (float)src0_ptr[i00];
                                }

                                quantize_row_q(src0_f32, dst_ptr + id, (int)ne00);
                                id += rs;
                            }

                            id += rs * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else
                {
                    Debug.Assert(false); // TODO: implement
                }
            }
            else
            {
                //printf("%s: this is not optimal - fix me\n", __func__);

                if (dst->type == ggml_type.GGML_TYPE_F32)
                {
                    nuint id = 0;
                    float* dst_ptr = (float*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    Half* src0_ptr = (Half*)((char*)src0->data + (nuint)i00 * (nuint)nb00 +
                                                             (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                                                             (nuint)i03 * (nuint)nb03);

                                    dst_ptr[id] = (float)(*src0_ptr);
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (dst->type == ggml_type.GGML_TYPE_F16)
                {
                    nuint id = 0;
                    Half* dst_ptr = (Half*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    Half* src0_ptr = (Half*)((byte*)src0->data + (nuint)i00 * (nuint)nb00 +
                                                             (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                                                             (nuint)i03 * (nuint)nb03);

                                    dst_ptr[id] = *src0_ptr;
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else
                {
                    Debug.Assert(false); // TODO: implement
                }
            }

            return;
        }

        // dst counters
        long i10 = 0;
        long i11 = 0;
        long i12 = 0;
        long i13 = 0;

        if (dst->type == ggml_type.GGML_TYPE_F16)
        {
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    i10 += ne00 * ir0;
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }

                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        for (long i00 = 0; i00 < ne00; i00++)
                        {
                            byte* src0_ptr = ((byte*)src0->data + (nuint)i00 * (nuint)nb00 + (nuint)i01 * (nuint)nb01 +
                                              (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                            byte* dst_ptr = ((byte*)dst->data + (nuint)i10 * (nuint)nb0 + (nuint)i11 * (nuint)nb1 +
                                             (nuint)i12 * (nuint)nb2 + (nuint)i13 * (nuint)nb3);

                            NativeMemory.Copy(dst_ptr, src0_ptr, (nuint)sizeof(Half));

                            if (++i10 == ne00)
                            {
                                i10 = 0;
                                if (++i11 == ne01)
                                {
                                    i11 = 0;
                                    if (++i12 == ne02)
                                    {
                                        i12 = 0;
                                        if (++i13 == ne03)
                                        {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    i10 += ne00 * (ne01 - ir1);
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (dst->type == ggml_type.GGML_TYPE_F32)
        {
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    i10 += ne00 * ir0;
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }

                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        for (long i00 = 0; i00 < ne00; i00++)
                        {
                            byte* src0_ptr = ((byte*)src0->data + (nuint)i00 * (nuint)nb00 + (nuint)i01 * (nuint)nb01 +
                                              (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                            byte* dst_ptr = ((byte*)dst->data + (nuint)i10 * (nuint)nb0 + (nuint)i11 * (nuint)nb1 +
                                             (nuint)i12 * (nuint)nb2 + (nuint)i13 * (nuint)nb3);

                            *(float*)dst_ptr = (float)(*(Half*)src0_ptr);

                            if (++i10 == ne0)
                            {
                                i10 = 0;
                                if (++i11 == ne1)
                                {
                                    i11 = 0;
                                    if (++i12 == ne2)
                                    {
                                        i12 = 0;
                                        if (++i13 == ne3)
                                        {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    i10 += ne00 * (ne01 - ir1);
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Debug.Assert(false); // TODO: implement
        }
    }

    private static void ggml_compute_forward_dup_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_nelements(dst) == ggml_nelements(src0));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

        long ne0 = dst->ne[0];
        long ne1 = dst->ne[1];
        long ne2 = dst->ne[2];
        long ne3 = dst->ne[3];

        ulong nb00 = src0->nb[0];
        ulong nb01 = src0->nb[1];
        ulong nb02 = src0->nb[2];
        ulong nb03 = src0->nb[3];

        ulong nb0 = dst->nb[0];
        ulong nb1 = dst->nb[1];
        ulong nb2 = dst->nb[2];
        ulong nb3 = dst->nb[3];

        int ith = @params->ith; // thread index
        int nth = @params->nth; // number of threads
        int dr;

        if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type)
        {
            // parallelize by elements
            int ne = (int)ggml_nelements(dst);
            dr = (ne + nth - 1) / nth;
            int ie0 = dr * ith;
            int ie1 = Math.Min(ie0 + dr, ne);

            NativeMemory.Copy(
                ((byte*)dst->data + (nuint)ie0 * (nuint)nb0),
                ((byte*)src0->data + (nuint)ie0 * (nuint)nb00),
                (nuint)(ie1 - ie0) * (nuint)GGML_TYPE_SIZE[(int)src0->type]);

            return;
        }

        // parallelize by rows
        int nr = (int)ne01;
        // number of rows per thread
        dr = ((nr + nth - 1) / nth);
        // row range for this thread
        int ir0 = dr * ith;
        int ir1 = Math.Min(ir0 + dr, nr);

        if (src0->type == dst->type &&
            ne00 == ne0 &&
            nb00 == GGML_TYPE_SIZE[(int)src0->type] && nb0 == GGML_TYPE_SIZE[(int)dst->type])
        {
            // copy by rows
            nuint rs = (nuint)ne00 * (nuint)nb00;
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        NativeMemory.Copy(
                            ((byte*)dst->data + (nuint)i01 * (nuint)nb1 + (nuint)i02 * (nuint)nb2 +
                             (nuint)i03 * (nuint)nb3),
                            ((byte*)src0->data + (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                             (nuint)i03 * (nuint)nb03),
                            rs);
                    }
                }
            }

            return;
        }

        if (ggml_is_contiguous(dst))
        {
            // TODO: simplify
            if (nb00 == sizeof(float))
            {
                if (dst->type == ggml_type.GGML_TYPE_F32)
                {
                    nuint id = 0;
                    nuint rs = (nuint)ne00 * (nuint)nb00;
                    byte* dst_ptr = (byte*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += rs * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                byte* src0_ptr = (byte*)src0->data + (nuint)i01 * (nuint)nb01 +
                                                 (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03;
                                NativeMemory.Copy(dst_ptr + id, src0_ptr, rs);
                                id += rs;
                            }

                            id += rs * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (dst->type == ggml_type.GGML_TYPE_F16)
                {
                    nuint id = 0;
                    Half* dst_ptr = (Half*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    float* src0_ptr = (float*)((byte*)src0->data + (nuint)i00 * (nuint)nb00 +
                                                               (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                                                               (nuint)i03 * (nuint)nb03);

                                    dst_ptr[id] = (Half)(*src0_ptr);
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (ggml_is_quantized(dst->type))
                {
                    var quantize_row_q = quantize_fns[(int)dst->type].quantize_row_q;

                    nuint id = 0;
                    nuint rs = (nuint)nb0 * (nuint)(ne00 / GGML_BLCK_SIZE[(int)dst->type]);
                    byte* dst_ptr = (byte*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += rs * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                float* src0_ptr = (float*)((byte*)src0->data + (nuint)i01 * (nuint)nb01 +
                                                           (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                                quantize_row_q(src0_ptr, dst_ptr + id, (int)ne00);
                                id += rs;
                            }

                            id += rs * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else
                {
                    Debug.Assert(false); // TODO: implement
                }
            }
            else
            {
                //printf("%s: this is not optimal - fix me\n", __func__);

                if (dst->type == ggml_type.GGML_TYPE_F32)
                {
                    nuint id = 0;
                    float* dst_ptr = (float*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    float* src0_ptr = (float*)((byte*)src0->data + (nuint)i00 * (nuint)nb00 +
                                                               (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                                                               (nuint)i03 * (nuint)nb03);

                                    dst_ptr[id] = *src0_ptr;
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else if (dst->type == ggml_type.GGML_TYPE_F16)
                {
                    nuint id = 0;
                    Half* dst_ptr = (Half*)dst->data;

                    for (int i03 = 0; i03 < ne03; i03++)
                    {
                        for (int i02 = 0; i02 < ne02; i02++)
                        {
                            id += (nuint)ne00 * (nuint)ir0;
                            for (int i01 = ir0; i01 < ir1; i01++)
                            {
                                for (int i00 = 0; i00 < ne00; i00++)
                                {
                                    float* src0_ptr = (float*)((byte*)src0->data + (nuint)i00 * (nuint)nb00 +
                                                               (nuint)i01 * (nuint)nb01 + (nuint)i02 * (nuint)nb02 +
                                                               (nuint)i03 * (nuint)nb03);

                                    dst_ptr[id] = (Half)(*src0_ptr);
                                    id++;
                                }
                            }

                            id += (nuint)ne00 * (nuint)(ne01 - ir1);
                        }
                    }
                }
                else
                {
                    Debug.Assert(false); // TODO: implement
                }
            }

            return;
        }

        // dst counters

        long i10 = 0;
        long i11 = 0;
        long i12 = 0;
        long i13 = 0;

        if (dst->type == ggml_type.GGML_TYPE_F32)
        {
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    i10 += ne00 * ir0;
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }

                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        for (long i00 = 0; i00 < ne00; i00++)
                        {
                            byte* src0_ptr = ((byte*)src0->data + (nuint)i00 * (nuint)nb00 + (nuint)i01 * (nuint)nb01 +
                                              (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                            byte* dst_ptr = ((byte*)dst->data + (nuint)i10 * (nuint)nb0 + (nuint)i11 * (nuint)nb1 +
                                             (nuint)i12 * (nuint)nb2 + (nuint)i13 * (nuint)nb3);

                            NativeMemory.Copy(dst_ptr, src0_ptr, sizeof(float));

                            if (++i10 == ne0)
                            {
                                i10 = 0;
                                if (++i11 == ne1)
                                {
                                    i11 = 0;
                                    if (++i12 == ne2)
                                    {
                                        i12 = 0;
                                        if (++i13 == ne3)
                                        {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    i10 += ne00 * (ne01 - ir1);
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (dst->type == ggml_type.GGML_TYPE_F16)
        {
            for (long i03 = 0; i03 < ne03; i03++)
            {
                for (long i02 = 0; i02 < ne02; i02++)
                {
                    i10 += ne00 * ir0;
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }

                    for (long i01 = ir0; i01 < ir1; i01++)
                    {
                        for (long i00 = 0; i00 < ne00; i00++)
                        {
                            byte* src0_ptr = ((byte*)src0->data + (nuint)i00 * (nuint)nb00 + (nuint)i01 * (nuint)nb01 +
                                              (nuint)i02 * (nuint)nb02 + (nuint)i03 * (nuint)nb03);
                            byte* dst_ptr = ((byte*)dst->data + (nuint)i10 * (nuint)nb0 + (nuint)i11 * (nuint)nb1 +
                                             (nuint)i12 * (nuint)nb2 + (nuint)i13 * (nuint)nb3);

                            *(Half*)dst_ptr = (Half)(*(float*)src0_ptr);

                            if (++i10 == ne0)
                            {
                                i10 = 0;
                                if (++i11 == ne1)
                                {
                                    i11 = 0;
                                    if (++i12 == ne2)
                                    {
                                        i12 = 0;
                                        if (++i13 == ne3)
                                        {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    i10 += ne00 * (ne01 - ir1);
                    while (i10 >= ne0)
                    {
                        i10 -= ne0;
                        if (++i11 == ne1)
                        {
                            i11 = 0;
                            if (++i12 == ne2)
                            {
                                i12 = 0;
                                if (++i13 == ne3)
                                {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Debug.Assert(false); // TODO: implement
        }
    }

    static void ggml_compute_forward_dup(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F16:
            {
                ggml_compute_forward_dup_f16(@params, src0, dst);
            }
                break;
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_dup_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_add_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int ith = @params->ith; // thread index
        int nth = @params->nth; // number of threads

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];

        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb00 == sizeof(float));

        if (nb10 == sizeof(float))
        {
            for (int j = ith; j < n; j += nth)
            {
#if GGML_USE_ACCELERATE
                vDSP_vadd(
                    (float *) ((char *) src0->data + j*nb01), 1,
                    (float *) ((char *) src1->data + j*nb11), 1,
                    (float *) ((char *) dst->data  + j*nb1),  1, nc);
#else
                ggml_vec_add_f32(nc,
                    (float*)((byte*)dst->data + (nuint)j * nb1),
                    (float*)((byte*)src0->data + (nuint)j * nb01),
                    (float*)((byte*)src1->data + (nuint)j * nb11));
#endif
            }
        }
        else
        {
            // src1 is not contiguous
            for (int j = ith; j < n; j += nth)
            {
                float* dst_ptr = (float*)((byte*)dst->data + (nuint)j * nb1);
                float* src0_ptr = (float*)((byte*)src0->data + (nuint)j * nb01);
                for (int i = 0; i < nc; i++)
                {
                    float* src1_ptr = (float*)((byte*)src1->data + (nuint)j * nb11 + (nuint)i * nb10);

                    dst_ptr[i] = src0_ptr[i] + *src1_ptr;
                }
            }
        }
    }

    static void ggml_compute_forward_add_f16_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int ith = @params->ith; // thread index
        int nth = @params->nth; // number of threads

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];

        Debug.Assert(src0->type == ggml_type.GGML_TYPE_F16);
        Debug.Assert(src1->type == ggml_type.GGML_TYPE_F32);
        Debug.Assert(dst->type == ggml_type.GGML_TYPE_F16);

        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb00 == sizeof(float));

        if (nb10 == sizeof(float))
        {
            for (int j = ith; j < n; j += nth)
            {
                Half* dst_ptr = (Half*)((byte*)dst->data + (nuint)j * nb1);
                Half* src0_ptr = (Half*)((byte*)src0->data + (nuint)j * nb01);
                for (int i = 0; i < nc; i++)
                {
                    float* src1_ptr = (float*)((byte*)src1->data + (nuint)j * nb11 + (nuint)i * nb10);
                    dst_ptr[i] = (Half)((float)(src0_ptr[i]) + *src1_ptr);
                }
            }
        }
        else
        {
            // src1 is not contiguous
            Debug.Assert(false);
        }
    }

    static void ggml_compute_forward_add_f16_f16(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int ith = @params->ith; // thread index
        int nth = @params->nth; // number of threads

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];

        Debug.Assert(src0->type == ggml_type.GGML_TYPE_F16);
        Debug.Assert(src1->type == ggml_type.GGML_TYPE_F16);
        Debug.Assert(dst->type == ggml_type.GGML_TYPE_F16);

        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb00 == sizeof(float));

        if (nb10 == (nuint)sizeof(Half))
        {
            for (int j = ith; j < n; j += nth)
            {
                Half* dst_ptr = (Half*)((byte*)dst->data + (nuint)j * nb1);
                Half* src0_ptr = (Half*)((byte*)src0->data + (nuint)j * nb01);
                for (int i = 0; i < nc; i++)
                {
                    Half* src1_ptr = (Half*)((byte*)src1->data + (nuint)j * nb11 + (nuint)i * nb10);
                    dst_ptr[i] = (Half)((float)(src0_ptr[i]) + (float)(*src1_ptr));
                }
            }
        }
        else
        {
            // src1 is not contiguous
            Debug.Assert(false);
        }
    }

    static void ggml_compute_forward_add_q_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

        //long ne10 = src1->ne[0];
        //long ne11 = src1->ne[1];
        long ne12 = src1->ne[2];
        long ne13 = src1->ne[3];

        //long ne0  = dst->ne[0];
        //long ne1  = dst->ne[1];
        long ne2 = dst->ne[2];
        long ne3 = dst->ne[3];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];
        nuint nb12 = (nuint)src1->nb[2];
        nuint nb13 = (nuint)src1->nb[3];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        int ith = @params->ith;
        int nth = @params->nth;

        Debug.Assert(ne02 == ne12);
        Debug.Assert(ne03 == ne13);
        Debug.Assert(ne2 == ne12);
        Debug.Assert(ne3 == ne13);

        ggml_type type = src0->type;
        var dequantize_row_q = quantize_fns[(int)type].dequantize_row_q;
        var quantize_row_q = quantize_fns[(int)type].quantize_row_q;

        // we don't support permuted src0 or src1
        Debug.Assert(nb00 == (nuint)GGML_TYPE_SIZE[(int)type]);
        Debug.Assert(nb10 == sizeof(float));

        // dst cannot be transposed or permuted
        Debug.Assert(nb0 <= nb1);
        Debug.Assert(nb1 <= nb2);
        Debug.Assert(nb2 <= nb3);

        Debug.Assert(ggml_is_quantized(src0->type));
        Debug.Assert(dst->type == src0->type);
        Debug.Assert(src1->type == ggml_type.GGML_TYPE_F32);

        // total rows in src0
        long nr = ne01 * ne02 * ne03;

        // rows per thread
        int dr = (int)((nr + nth - 1) / nth);

        // row range for this thread
        int ir0 = dr * ith;
        int ir1 = Math.Min(ir0 + dr, (int)nr);

        float* wdata = (float*)@params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        for (int ir = ir0; ir < ir1; ++ir)
        {
            // src0 indices
            nuint i03 = (nuint)ir / (nuint)(ne02 * ne01);
            nuint i02 = ((nuint)ir - i03 * (nuint)ne02 * (nuint)ne01) / (nuint)ne01;
            nuint i01 = ((nuint)ir - i03 * (nuint)ne02 * (nuint)ne01 - i02 * (nuint)ne01);

            // src1 and dst are same shape as src0 => same indices
            nuint i13 = i03;
            nuint i12 = i02;
            nuint i11 = i01;

            nuint i3 = i03;
            nuint i2 = i02;
            nuint i1 = i01;

            void* src0_row = (void*)((byte*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
            float* src1_row = (float*)((byte*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
            void* dst_row = (void*)((byte*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

            Debug.Assert(ne00 % 32 == 0);

            // unquantize row from src0 to temp buffer
            dequantize_row_q(src0_row, wdata, (int)ne00);
            // add src1
            ggml_vec_acc_f32((int)ne00, wdata, src1_row);
            // quantize row to dst
            quantize_row_q(wdata, dst_row, (int)ne00);
        }
    }

    static void ggml_compute_forward_add(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_add_f32(@params, src0, src1, dst);
            }
                break;
            case ggml_type.GGML_TYPE_F16:
            {
                if (src1->type == ggml_type.GGML_TYPE_F16)
                {
                    ggml_compute_forward_add_f16_f16(@params, src0, src1, dst);
                }
                else if (src1->type == ggml_type.GGML_TYPE_F32)
                {
                    ggml_compute_forward_add_f16_f32(@params, src0, src1, dst);
                }
                else
                {
                    Debug.Assert(false);
                }
            }
                break;
            case ggml_type.GGML_TYPE_Q4_0:
            case ggml_type.GGML_TYPE_Q4_1:
            case ggml_type.GGML_TYPE_Q4_2:
            case ggml_type.GGML_TYPE_Q4_3:
            case ggml_type.GGML_TYPE_Q5_0:
            case ggml_type.GGML_TYPE_Q5_1:
            case ggml_type.GGML_TYPE_Q8_0:
            {
                ggml_compute_forward_add_q_f32(@params, src0, src1, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_sub_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        Debug.Assert(dst->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src0->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src1->nb[0] == (nuint)sizeof(float));

        for (int i = 0; i < n; i++)
        {
            ggml_vec_sub_f32(nc,
                (float*)((byte*)dst->data + (ulong)i * (dst->nb[1])),
                (float*)((byte*)src0->data + (ulong)i * (src0->nb[1])),
                (float*)((byte*)src1->data + (ulong)i * (src1->nb[1])));
        }
    }

    static void ggml_compute_forward_sub(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_sub_f32(@params, src0, src1, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_mul_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        Debug.Assert(dst->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src0->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src1->nb[0] == (nuint)sizeof(float));

        for (int i = 0; i < n; i++)
        {
            ggml_vec_mul_f32(nc,
                (float*)((byte*)dst->data + (ulong)i * (dst->nb[1])),
                (float*)((byte*)src0->data + (ulong)i * (src0->nb[1])),
                (float*)((byte*)src1->data + (ulong)i * (src1->nb[1])));
        }
    }

    static void ggml_compute_forward_mul(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_mul_f32(@params, src0, src1, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_div_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        Debug.Assert(dst->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src0->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src1->nb[0] == (nuint)sizeof(float));

        for (int i = 0; i < n; i++)
        {
            ggml_vec_div_f32(nc,
                (float*)((byte*)dst->data + (ulong)i * (dst->nb[1])),
                (float*)((byte*)src0->data + (ulong)i * (src0->nb[1])),
                (float*)((byte*)src1->data + (ulong)i * (src1->nb[1])));
        }
    }

    static void ggml_compute_forward_div(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_div_f32(@params, src0, src1, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_sqr_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        Debug.Assert(dst->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src0->nb[0] == (nuint)sizeof(float));

        for (int i = 0; i < n; i++)
        {
            ggml_vec_sqr_f32(nc,
                (float*)((byte*)dst->data + (ulong)i * (dst->nb[1])),
                (float*)((byte*)src0->data + (ulong)i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_sqr(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_sqr_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_sqrt_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int n = ggml_nrows(src0);
        int nc = (int)src0->ne[0];

        Debug.Assert(dst->nb[0] == (nuint)sizeof(float));
        Debug.Assert(src0->nb[0] == (nuint)sizeof(float));

        for (int i = 0; i < n; i++)
        {
            ggml_vec_sqrt_f32(nc,
                (float*)((byte*)dst->data + (ulong)i * (dst->nb[1])),
                (float*)((byte*)src0->data + (ulong)i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_sqrt(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_sqrt_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_sum_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_is_scalar(dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        Debug.Assert(src0->nb[0] == sizeof(float));

        nuint ne00 = (nuint)src0->ne[0];
        nuint ne01 = (nuint)src0->ne[1];
        nuint ne02 = (nuint)src0->ne[2];
        nuint ne03 = (nuint)src0->ne[3];

        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        double sum = 0;
        double row_sum = 0;

        for (nuint i03 = 0; i03 < ne03; i03++)
        {
            for (nuint i02 = 0; i02 < ne02; i02++)
            {
                for (nuint i01 = 0; i01 < ne01; i01++)
                {
                    ggml_vec_sum_ggf((int)ne00,
                        &row_sum,
                        (float*)((byte*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                    sum += row_sum;
                }
            }
        }

        ((float*)dst->data)[0] = (float)sum;
    }

    static void ggml_compute_forward_sum(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_mean_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        Debug.Assert(src0->nb[0] == sizeof(float));

        nuint ne00 = (nuint)src0->ne[0];
        nuint ne01 = (nuint)src0->ne[1];
        nuint ne02 = (nuint)src0->ne[2];
        nuint ne03 = (nuint)src0->ne[3];

        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint ne0 = (nuint)dst->ne[0];
        nuint ne1 = (nuint)dst->ne[1];
        nuint ne2 = (nuint)dst->ne[2];
        nuint ne3 = (nuint)dst->ne[3];

        Debug.Assert(ne0 == 1);
        Debug.Assert(ne1 == ne01);
        Debug.Assert(ne2 == ne02);
        Debug.Assert(ne3 == ne03);

        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        for (nuint i03 = 0; i03 < ne03; i03++)
        {
            for (nuint i02 = 0; i02 < ne02; i02++)
            {
                for (nuint i01 = 0; i01 < ne01; i01++)
                {
                    ggml_vec_sum_f32((int)ne00,
                        (float*)((byte*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                        (float*)((byte*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));

                    *(float*)((byte*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3) /= (float)ne00;
                }
            }
        }
    }

    static void ggml_compute_forward_mean(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_repeat_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_can_repeat(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        // TODO: implement support for rank > 2 tensors
        Debug.Assert(src0->ne[2] == 1);
        Debug.Assert(src0->ne[3] == 1);
        Debug.Assert(dst->ne[2] == 1);
        Debug.Assert(dst->ne[3] == 1);

        nuint nc = (nuint)dst->ne[0];
        nuint nr = (nuint)dst->ne[1];
        nuint nc0 = (nuint)src0->ne[0];
        nuint nr0 = (nuint)src0->ne[1];
        nuint ncr = nc / nc0; // guaranteed to be an integer due to the check in ggml_can_repeat
        nuint nrr = nr / nr0; // guaranteed to be an integer due to the check in ggml_can_repeat

        // TODO: support for transposed / permuted tensors
        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        // TODO: maybe this is not optimal?
        for (nuint i = 0; i < nrr; i++)
        {
            for (nuint j = 0; j < ncr; j++)
            {
                for (nuint k = 0; k < nr0; k++)
                {
                    ggml_vec_cpy_f32((int)nc0,
                        (float*)((byte*)dst->data + (i * nr0 + k) * (dst->nb[1]) + j * nc0 * (dst->nb[0])),
                        (float*)((byte*)src0->data + (k) * (src0->nb[1])));
                }
            }
        }
    }

    static void ggml_compute_forward_repeat(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_abs_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        nuint n = (nuint)ggml_nrows(src0);
        nuint nc = (nuint)src0->ne[0];

        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        for (nuint i = 0; i < n; i++)
        {
            ggml_vec_abs_f32((int)nc,
                (float*)((byte*)dst->data + i * (dst->nb[1])),
                (float*)((byte*)src0->data + i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_abs(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_abs_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_sgn_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        nuint n = (nuint)ggml_nrows(src0);
        nuint nc = (nuint)src0->ne[0];

        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        for (nuint i = 0; i < n; i++)
        {
            ggml_vec_sgn_f32((int)nc,
                (float*)((byte*)dst->data + i * (dst->nb[1])),
                (float*)((byte*)src0->data + i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_sgn(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_sgn_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_neg_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        nuint n = (nuint)ggml_nrows(src0);
        nuint nc = (nuint)src0->ne[0];

        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        for (nuint i = 0; i < n; i++)
        {
            ggml_vec_neg_f32((int)nc,
                (float*)((byte*)dst->data + i * (dst->nb[1])),
                (float*)((byte*)src0->data + i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_neg(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_neg_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_step_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        nuint n = (nuint)ggml_nrows(src0);
        nuint nc = (nuint)src0->ne[0];

        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        for (nuint i = 0; i < n; i++)
        {
            ggml_vec_step_f32((int)nc,
                (float*)((byte*)dst->data + i * (dst->nb[1])),
                (float*)((byte*)src0->data + i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_step(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_step_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_relu_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(@params->ith == 0);
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        nuint n = (nuint)ggml_nrows(src0);
        nuint nc = (nuint)src0->ne[0];

        Debug.Assert(dst->nb[0] == sizeof(float));
        Debug.Assert(src0->nb[0] == sizeof(float));

        for (nuint i = 0; i < n; i++)
        {
            ggml_vec_relu_f32((int)nc,
                (float*)((byte*)dst->data + i * (dst->nb[1])),
                (float*)((byte*)src0->data + i * (src0->nb[1])));
        }
    }

    static void ggml_compute_forward_relu(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_relu_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_gelu_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_is_contiguous(src0));
        Debug.Assert(ggml_is_contiguous(dst));
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int ith = @params->ith;
        int nth = @params->nth;

        int nc = (int)src0->ne[0];
        int nr = ggml_nrows(src0);

        // rows per thread
        int dr = (nr + nth - 1) / nth;

        // row range for this thread
        int ir0 = dr * ith;
        nuint ir1 = (nuint)Math.Min(ir0 + dr, nr);

        for (nuint i1 = (nuint)ir0; i1 < ir1; i1++)
        {
            ggml_vec_gelu_f32((int)nc,
                (float*)((byte*)dst->data + i1 * (dst->nb[1])),
                (float*)((byte*)src0->data + i1 * (src0->nb[1])));

#if !NDEBUG
            for (int k = 0; k < nc; k++)
            {
                float x = ((float*)((byte*)dst->data + i1 * (dst->nb[1])))[k];

                Debug.Assert(!float.IsNaN(x));
                Debug.Assert(!float.IsNaN(x));
            }
#endif
        }
    }

    static void ggml_compute_forward_gelu(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_silu_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_is_contiguous(src0));
        Debug.Assert(ggml_is_contiguous(dst));
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        int ith = @params->ith;
        int nth = @params->nth;

        int nc = (int)src0->ne[0];
        int nr = ggml_nrows(src0);

        // rows per thread
        int dr = (nr + nth - 1) / nth;

        // row range for this thread
        int ir0 = dr * ith;
        nuint ir1 = (nuint)Math.Min(ir0 + dr, nr);

        for (nuint i1 = (nuint)ir0; i1 < ir1; i1++)
        {
            ggml_vec_silu_f32((int)nc,
                (float*)((byte*)dst->data + i1 * (dst->nb[1])),
                (float*)((byte*)src0->data + i1 * (src0->nb[1])));

#if !NDEBUG
            for (int k = 0; k < nc; k++)
            {
                float x = ((float*)((byte*)dst->data + i1 * (dst->nb[1])))[k];

                Debug.Assert(!float.IsNaN(x));
                Debug.Assert(!float.IsNaN(x));
            }
#endif
        }
    }

    static void ggml_compute_forward_silu(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_norm_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        Debug.Assert(src0->nb[0] == sizeof(float));

        int ith = @params->ith;
        int nth = @params->nth;

        nuint ne00 = (nuint)src0->ne[0];
        nuint ne01 = (nuint)src0->ne[1];
        nuint ne02 = (nuint)src0->ne[2];
        nuint ne03 = (nuint)src0->ne[3];

        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        float eps = 1e-5f; // TODO: make this a parameter

        // TODO: optimize
        for (nuint i03 = 0; i03 < ne03; i03++)
        {
            for (nuint i02 = 0; i02 < ne02; i02++)
            {
                for (nuint i01 = (nuint)ith; i01 < ne01; i01 += (nuint)nth)
                {
                    float* x = (float*)((byte*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    double sum = 0.0;
                    for (nuint i00 = 0; i00 < ne00; i00++)
                    {
                        sum += x[i00];
                    }

                    float mean = (float)(sum / ne00);

                    float* y = (float*)((byte*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    double sum2 = 0.0;
                    for (nuint i00 = 0; i00 < ne00; i00++)
                    {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sum2 += (v * v);
                    }

                    float variance = (float)(sum2 / ne00);
                    float scale = 1.0f / MathF.Sqrt(variance + eps);

                    ggml_vec_scale_f32((int)ne00, y, scale);
                }
            }
        }
    }

    static void ggml_compute_forward_norm(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_rms_norm_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_are_same_shape(src0, dst));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        Debug.Assert(src0->nb[0] == sizeof(float));

        int ith = @params->ith;
        int nth = @params->nth;

        nuint ne00 = (nuint)src0->ne[0];
        nuint ne01 = (nuint)src0->ne[1];
        nuint ne02 = (nuint)src0->ne[2];
        nuint ne03 = (nuint)src0->ne[3];

        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        float eps = 1e-6f; // TODO: make this a parameter

        // TODO: optimize
        for (nuint i03 = 0; i03 < ne03; i03++)
        {
            for (nuint i02 = 0; i02 < ne02; i02++)
            {
                for (nuint i01 = (nuint)ith; i01 < ne01; i01 += (nuint)nth)
                {
                    float* x = (float*)((byte*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    double sum = 0.0;
                    for (nuint i00 = 0; i00 < ne00; i00++)
                    {
                        sum += x[i00] * x[i00];
                    }

                    float mean = (float)(sum / ne00);

                    float* y = (float*)((byte*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    NativeMemory.Copy(x, y, ne00 * sizeof(float));
                    // for (int i00 = 0; i00 < ne00; i00++) {
                    //     y[i00] = x[i00];
                    // }

                    float scale = 1.0f / MathF.Sqrt(mean + eps);

                    ggml_vec_scale_f32((int)ne00, y, scale);
                }
            }
        }
    }

    static void ggml_compute_forward_rms_norm(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32(@params, src0, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(
        ggml_tensor * src0,
        ggml_tensor * src1,
        ggml_tensor * dst) {
    //long ne00 = src0->ne[0];
    //long ne01 = src0->ne[1];

    long ne10 = src1->ne[0];

    long ne0 = dst->ne[0];
    long ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) && ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32))) {

        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    return false;
}
#endif

    static void ggml_compute_forward_mul_mat_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        long t0 = ggml_perf_time_us();

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
        long ne10 = src1->ne[0];
#endif
        long ne11 = src1->ne[1];
#if !NDEBUG
        long ne12 = src1->ne[2];
        long ne13 = src1->ne[3];

        nuint ne0 = (nuint)dst->ne[0];
        nuint ne1 = (nuint)dst->ne[1];
        nuint ne2 = (nuint)dst->ne[2];
        nuint ne3 = (nuint)dst->ne[3];

        nuint nb00 = (nuint)src0->nb[0];
#endif
        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

#if !NDEBUG
        nuint nb10 = (nuint)src1->nb[0];
#endif
        nuint nb11 = (nuint)src1->nb[1];
        nuint nb12 = (nuint)src1->nb[2];
        nuint nb13 = (nuint)src1->nb[3];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        int ith = @params->ith;
        int nth = @params->nth;

        Debug.Assert(ne02 == ne12);
        Debug.Assert(ne03 == ne13);
        Debug.Assert(ne2 == (nuint)ne12);
        Debug.Assert(ne3 == (nuint)ne13);

        // we don't support permuted src0 or src1
        Debug.Assert(nb00 == sizeof(float));
        Debug.Assert(nb10 == sizeof(float));

        // dst cannot be transposed or permuted
        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb0 <= nb1);
        Debug.Assert(nb1 <= nb2);
        Debug.Assert(nb2 <= nb3);

        Debug.Assert(ne0 == (nuint)ne01);
        Debug.Assert(ne1 == (nuint)ne11);
        Debug.Assert(ne2 == (nuint)ne02);
        Debug.Assert(ne3 == (nuint)ne03);

        // nb01 >= nb00 - src0 is not transposed
        //   compute by src0 rows

#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if GGML_USE_CUBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        size_t x_size, y_size, d_size;
        float *d_X = ggml_cuda_pool_malloc(sizeof(float) * x_ne, &x_size);
        float *d_Y = ggml_cuda_pool_malloc(sizeof(float) * y_ne, &y_size);
        float *d_D = ggml_cuda_pool_malloc(sizeof(float) * d_ne, &d_size);
#endif

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

#if GGML_USE_CUBLAS
                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_X, x, sizeof(float) * x_ne, cudaMemcpyHostToDevice, g_cudaStream));
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(float) * y_ne, cudaMemcpyHostToDevice, g_cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, ne00,
                                    d_Y, ne10,
                            &beta,  d_D, ne01));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, g_cudaStream));
#elif GGML_USE_CLBLAST
                // zT = y * xT
                ggml_cl_sgemm_wrapper(GGML_BLAS_ORDER_ROW_MAJOR, GGML_BLAS_OP_N, GGML_BLAS_OP_T,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne10,
                        0.0f,    d, ne01,
                        GGML_TYPE_F32);
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }
#if GGML_USE_CUBLAS
        CUDA_CHECK(cudaStreamSynchronize(g_cudaStream));
        ggml_cuda_pool_free(d_X, x_size);
        ggml_cuda_pool_free(d_Y, y_size);
        ggml_cuda_pool_free(d_D, d_size);
#endif
        //printf("CBLAS F32 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

        if (@params->type == ggml_task_type.GGML_TASK_INIT)
        {
            return;
        }

        if (@params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        // parallelize by src0 rows using ggml_vec_dot_f32

        // total rows in src0
        nuint nr = (nuint)(ne01 * ne02 * ne03);

        // rows per thread
        nuint dr = (nr + (nuint)nth - 1) / (nuint)nth;

        // row range for this thread
        nuint ir0 = dr * (nuint)ith;
        nuint ir1 = Math.Min(ir0 + dr, nr);

        for (nuint ir = ir0; ir < ir1; ++ir)
        {
            // src0 indices
            nuint i03 = (nuint)(ir / (nuint)(ne02 * ne01));
            nuint i02 = (nuint)((ir - i03 * (nuint)ne02 * (nuint)ne01) / (nuint)ne01);
            nuint i01 = (nuint)((ir - i03 * (nuint)ne02 * (nuint)ne01 - i02 * (nuint)ne01));

            for (long ic = 0; ic < ne11; ++ic)
            {
                // src1 indices
                nuint i13 = (nuint)i03;
                nuint i12 = (nuint)i02;
                nuint i11 = (nuint)ic;

                // dst indices
                nuint i0 = (nuint)i01;
                nuint i1 = (nuint)i11;
                nuint i2 = (nuint)i02;
                nuint i3 = (nuint)i03;

                ggml_vec_dot_f32((int)ne00,
                    (float*)((byte*)dst->data + (i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                    (float*)((byte*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03)),
                    (float*)((byte*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13)));
            }
        }

        //long t1 = ggml_perf_time_us();
        //static int64_t acc = 0;
        //acc += t1 - t0;
        //if (t1 - t0 > 10) {
        //    printf("\n");
        //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
        //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
        //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
        //    printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

        //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
        //}
    }

    static void ggml_compute_forward_mul_mat_f16_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        long t0 = ggml_perf_time_us();

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

        long ne10 = src1->ne[0];
        long ne11 = src1->ne[1];
        long ne12 = src1->ne[2];
        long ne13 = src1->ne[3];

        nuint ne0 = (nuint)dst->ne[0];
        nuint ne1 = (nuint)dst->ne[1];
        nuint ne2 = (nuint)dst->ne[2];
        nuint ne3 = (nuint)dst->ne[3];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];
        nuint nb12 = (nuint)src1->nb[2];
        nuint nb13 = (nuint)src1->nb[3];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        int ith = @params->ith;
        int nth = @params->nth;

        Debug.Assert(ne02 == ne12);
        Debug.Assert(ne03 == ne13);
        Debug.Assert(ne2 == (nuint)ne12);
        Debug.Assert(ne3 == (nuint)ne13);

        // TODO: we don't support permuted src0
        Debug.Assert((int)nb00 == sizeof(Half));

        // dst cannot be transposed or permuted
        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb0 <= nb1);
        Debug.Assert(nb1 <= nb2);
        Debug.Assert(nb2 <= nb3);

        Debug.Assert(ne0 == (nuint)ne01);
        Debug.Assert(ne1 == (nuint)ne11);
        Debug.Assert(ne2 == (nuint)ne02);
        Debug.Assert(ne3 == (nuint)ne03);

        // nb01 >= nb00 - src0 is not transposed
        //   compute by src0 rows

#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        GGML_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if GGML_USE_CUBLAS
        ggml_fp16_t * const wdata = params->wdata;

        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        size_t x_size, y_size, d_size;
        float *d_X = ggml_cuda_pool_malloc(sizeof(float) * x_ne, &x_size);
        float *d_Y = ggml_cuda_pool_malloc(sizeof(float) * y_ne, &y_size);
        float *d_D = ggml_cuda_pool_malloc(sizeof(float) * d_ne, &d_size);
#else
        float * const wdata = params->wdata;
#endif
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
#if GGML_USE_CUBLAS
                // with cuBlAS, instead of converting src0 to fp32, we convert src1 to fp16
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne11; ++i01) {
                        for (int64_t i00 = 0; i00 < ne10; ++i00) {
                            wdata[id++] = GGML_FP32_TO_FP16(*(float *) ((char *) src1->data + i03*nb13 + i02*nb12 + i01*nb11 + i00*nb10));
                        }
                    }
                }
#else
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne01; ++i01) {
                        for (int64_t i00 = 0; i00 < ne00; ++i00) {
                            wdata[id++] = GGML_FP16_TO_FP32(*(ggml_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        }
                    }
                }
#endif

#if GGML_USE_CUBLAS
                const ggml_fp16_t * x = (ggml_fp16_t *) ((char *) src0->data + i02*nb02 + i03*nb03);
                const ggml_fp16_t * y = (ggml_fp16_t *) wdata;

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_X, x, sizeof(ggml_fp16_t) * x_ne, cudaMemcpyHostToDevice, g_cudaStream));
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(ggml_fp16_t) * y_ne, cudaMemcpyHostToDevice, g_cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasGemmEx(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, CUDA_R_16F, ne00,
                                    d_Y, CUDA_R_16F, ne10,
                            &beta,  d_D, CUDA_R_32F, ne01,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, g_cudaStream));
#elif GGML_USE_CLBLAST
                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // zT = y * xT
                ggml_cl_sgemm_wrapper(GGML_BLAS_ORDER_ROW_MAJOR, GGML_BLAS_OP_N, GGML_BLAS_OP_T,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne10,
                        0.0f,    d, ne01,
                        GGML_TYPE_F32);
#else
                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // zT = y * xT
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }

#if GGML_USE_CUBLAS
        CUDA_CHECK(cudaStreamSynchronize(g_cudaStream));
        ggml_cuda_pool_free(d_X, x_size);
        ggml_cuda_pool_free(d_Y, y_size);
        ggml_cuda_pool_free(d_D, d_size);
#endif
        /*printf("CBLAS F16 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);*/

        return;
    }
#endif

        Half* wdata = (Half*)@params->wdata;
        if (@params->type == ggml_task_type.GGML_TASK_INIT)
        {
            // Half* wdata = (Half*)@params->wdata;
            nuint id = 0;
            for (nuint i13 = 0; i13 < (nuint)ne13; ++i13) {
                for (nuint i12 = 0; i12 < (nuint)ne12; ++i12) {
                    for (nuint i11 = 0; i11 < (nuint)ne11; ++i11) {
                        for (nuint i10 = 0; i10 < (nuint)ne10; ++i10) {
                            wdata[id++] = (Half)(*(float *)((byte *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        }
                    }
                }
            }

            Debug.Assert(id*(nuint)sizeof(Half) <= @params->wsize);

            return;
        }

        if (@params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

    // fp16 -> half the size, so divide by 2
    // TODO: do not support transposed src1
    Debug.Assert(nb10/2 == (nuint)sizeof(Half));

    // parallelize by src0 rows using ggml_vec_dot_f16

    // total rows in src0
    nuint nr = (nuint)(ne01*ne02*ne03);

    // rows per thread
    nuint dr = (nuint)(nr + (nuint)nth - 1)/(nuint)nth;

    // row range for this thread
    nuint ir0 = dr*(nuint)ith;
    nuint ir1 = Math.Min(ir0 + dr, nr);

    //Half * wdata = @params->wdata;

    for (nuint ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        nuint i03 = (nuint)ir/(nuint)(ne02*ne01);
        nuint i02 = (ir - i03*(nuint)ne02*(nuint)ne01)/(nuint)ne01;
        nuint i01 = (ir - i03*(nuint)ne02*(nuint)ne01 - i02*(nuint)ne01);

        nuint i13 = i03;
        nuint i12 = i02;

        nuint i0 = i01;
        nuint i2 = i02;
        nuint i3 = i03;

        Half * src0_row = (Half *) ((byte *) src0->data + (i01*nb01 + i02*nb02        + i03*nb03));
        Half * src1_col =                         wdata + (       0 + i12*(nuint)ne11 + i13*(nuint)ne12*(nuint)ne11)*(nuint)ne00;

        float * dst_col = (float *) ((byte *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

        for (nuint ic = 0; ic < (nuint)ne11; ++ic) {
            ggml_vec_dot_f16((int)ne00, &dst_col[ic*ne0], src0_row, src1_col + ic*(nuint)ne00);
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
    }

    static void ggml_compute_forward_mul_mat_q_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        long t0 = ggml_perf_time_us();

        long ne00 = src0->ne[0];
        long ne01 = src0->ne[1];
        long ne02 = src0->ne[2];
        long ne03 = src0->ne[3];

        long ne10 = src1->ne[0];
        long ne11 = src1->ne[1];
        long ne12 = src1->ne[2];
        long ne13 = src1->ne[3];

        nuint ne0 = (nuint)dst->ne[0];
        nuint ne1 = (nuint)dst->ne[1];
        nuint ne2 = (nuint)dst->ne[2];
        nuint ne3 = (nuint)dst->ne[3];

        nuint nb00 = (nuint)src0->nb[0];
        nuint nb01 = (nuint)src0->nb[1];
        nuint nb02 = (nuint)src0->nb[2];
        nuint nb03 = (nuint)src0->nb[3];

        nuint nb10 = (nuint)src1->nb[0];
        nuint nb11 = (nuint)src1->nb[1];
        nuint nb12 = (nuint)src1->nb[2];
        nuint nb13 = (nuint)src1->nb[3];

        nuint nb0 = (nuint)dst->nb[0];
        nuint nb1 = (nuint)dst->nb[1];
        nuint nb2 = (nuint)dst->nb[2];
        nuint nb3 = (nuint)dst->nb[3];

        int ith = @params->ith;
        int nth = @params->nth;

        Debug.Assert(ne02 == ne12);
        Debug.Assert(ne03 == ne13);
        Debug.Assert(ne2 == (nuint)ne12);
        Debug.Assert(ne3 == (nuint)ne13);

        ggml_type type = src0->type;
        var quantize_row_q_dot = quantize_fns[(int)type].quantize_row_q_dot;
        var vec_dot_q          = quantize_fns[(int)type].vec_dot_q;
        ggml_type vec_dot_type       = quantize_fns[(int)type].vec_dot_type;

        // we don't support permuted src0 or src1
        Debug.Assert((int) nb00 == (int) GGML_TYPE_SIZE[(int) type]);
        Debug.Assert(nb10 == sizeof(float));

        // dst cannot be transposed or permuted
        Debug.Assert(nb0 == sizeof(float));
        Debug.Assert(nb0 <= nb1);
        Debug.Assert(nb1 <= nb2);
        Debug.Assert(nb2 <= nb3);

        Debug.Assert(ne0 == (nuint)ne01);
        Debug.Assert(ne1 == (nuint)ne11);
        Debug.Assert(ne2 == (nuint)ne02);
        Debug.Assert(ne3 == (nuint)ne03);

        // nb01 >= nb00 - src0 is not transposed
        //   compute by src0 rows

#if GGML_USE_ACCELERATE || GGML_USE_OPENBLAS || GGML_USE_CUBLAS || GGML_USE_CLBLAST
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if GGML_USE_CUBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        size_t x_size, y_size, d_size, q_size;
        float *d_X = ggml_cuda_pool_malloc(sizeof(float) * x_ne, &x_size);
        float *d_Y = ggml_cuda_pool_malloc(sizeof(float) * y_ne, &y_size);
        float *d_D = ggml_cuda_pool_malloc(sizeof(float) * d_ne, &d_size);
        float *d_Q = ggml_cuda_pool_malloc(GGML_TYPE_SIZE[type] * x_ne / GGML_BLCK_SIZE[type], &q_size);

        void (*dequantize_row_q_cuda)(const void * x, float * y, int k, cudaStream_t stream)  = NULL;
        if (type == GGML_TYPE_Q4_0) {
            dequantize_row_q_cuda = dequantize_row_q4_0_cuda;
        }
        else if (type == GGML_TYPE_Q4_1) {
            dequantize_row_q_cuda = dequantize_row_q4_1_cuda;
        }
        else if (type == GGML_TYPE_Q4_2) {
            dequantize_row_q_cuda = dequantize_row_q4_2_cuda;
        }
        else if (type == GGML_TYPE_Q4_3) {
            dequantize_row_q_cuda = dequantize_row_q4_3_cuda;
        }
        else if (type == GGML_TYPE_Q5_0) {
            dequantize_row_q_cuda = dequantize_row_q5_0_cuda;
        }
        else if (type == GGML_TYPE_Q5_1) {
            dequantize_row_q_cuda = dequantize_row_q5_1_cuda;
        }
        else if (type == GGML_TYPE_Q8_0) {
            dequantize_row_q_cuda = dequantize_row_q8_0_cuda;
        }
        else {
            GGML_ASSERT(false);
        }
#elif !GGML_USE_CLBLAST
        float * const wdata = params->wdata;
        dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;
#endif

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

#if GGML_USE_CUBLAS
                // copy and dequantize on device
                CUDA_CHECK(
                    cudaMemcpyAsync(d_Q, (char *) src0->data + i03*nb03 + i02*nb02,
                        GGML_TYPE_SIZE[type] * x_ne / GGML_BLCK_SIZE[type], cudaMemcpyHostToDevice, g_cudaStream));

                dequantize_row_q_cuda(d_Q, d_X, ne01 * ne00, g_cudaStream);
                CUDA_CHECK(cudaGetLastError());
#elif GGML_USE_CLBLAST
                const void* x = (char *) src0->data + i03*nb03 + i02*nb02;
#else
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne01; ++i01) {
                        dequantize_row_q((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }
                const float * x = wdata;
#endif


#if GGML_USE_CUBLAS
                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(float) * y_ne, cudaMemcpyHostToDevice, g_cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, ne00,
                                    d_Y, ne10,
                            &beta,  d_D, ne01));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, g_cudaStream));
#elif GGML_USE_CLBLAST
                // zT = y * xT
                ggml_cl_sgemm_wrapper(GGML_BLAS_ORDER_ROW_MAJOR, GGML_BLAS_OP_N, GGML_BLAS_OP_T,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne10,
                        0.0f,    d, ne01,
                        type);
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }

#if GGML_USE_CUBLAS
        CUDA_CHECK(cudaStreamSynchronize(g_cudaStream));
        ggml_cuda_pool_free(d_X, x_size);
        ggml_cuda_pool_free(d_Y, y_size);
        ggml_cuda_pool_free(d_D, d_size);
        ggml_cuda_pool_free(d_Q, q_size);
#endif
        //printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

        byte* wdata = (byte*)@params->wdata;
        nuint row_size;
        if (@params->type == ggml_task_type.GGML_TASK_INIT)
        {
            // byte* wdata = (byte*)@params->wdata;
            row_size = (nuint)ne10*(nuint)GGML_TYPE_SIZE[(int)vec_dot_type]/(nuint)GGML_BLCK_SIZE[(int)vec_dot_type];
            for (nuint i13 = 0; i13 < (nuint)ne13; ++i13) {
                for (nuint i12 = 0; i12 < (nuint)ne12; ++i12) {
                    for (nuint i11 = 0; i11 < (nuint)ne11; ++i11) {
                        quantize_row_q_dot((float *)((byte *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, (int)ne10);
                        wdata += row_size;
                    }
                }
            }

            return;
        }

        if (@params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        // parallelize by src0 rows using ggml_vec_dot_q

    // total rows in src0
    nuint nr = (nuint)(ne01*ne02*ne03);

    // rows per thread
    nuint dr = (nuint)(nr + (nuint)nth - 1)/(nuint)nth;

    // row range for this thread
    nuint ir0 = dr*(nuint)ith;
    nuint ir1 = Math.Min(ir0 + dr, nr);

    //Half * wdata = @params->wdata;
    row_size = (nuint)ne00*(nuint)GGML_TYPE_SIZE[(int)vec_dot_type]/(nuint)GGML_BLCK_SIZE[(int)vec_dot_type];
    for (nuint ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        nuint i03 = (nuint)ir/(nuint)(ne02*ne01);
        nuint i02 = (ir - i03*(nuint)ne02*(nuint)ne01)/(nuint)ne01;
        nuint i01 = (ir - i03*(nuint)ne02*(nuint)ne01 - i02*(nuint)ne01);

        nuint i13 = i03;
        nuint i12 = i02;

        nuint i0 = i01;
        nuint i2 = i02;
        nuint i3 = i03;

        void * src0_row = (void *) ((byte *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        byte * src1_col =          ((byte *)      wdata + (      (0 + i12*(nuint)ne11 + i13*(nuint)ne12*(nuint)ne11)*row_size));

        float * dst_col = (float *) ((byte *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

        Debug.Assert(ne00 % 32 == 0);

        for (nuint ic = 0; ic < (nuint)ne11; ++ic) {
            vec_dot_q((int)ne00, &dst_col[ic*ne0], src0_row, (void *) (src1_col + ic*row_size));
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
    }

    static void ggml_compute_forward_mul_mat(
        ggml_compute_params * @params,
        ggml_tensor * src0,
        ggml_tensor * src1,
        ggml_tensor * dst) {
        switch (src0->type) {
            case ggml_type.GGML_TYPE_Q4_0:
            case ggml_type.GGML_TYPE_Q4_1:
            case ggml_type.GGML_TYPE_Q4_2:
            case ggml_type.GGML_TYPE_Q4_3:
            case ggml_type.GGML_TYPE_Q5_0:
            case ggml_type.GGML_TYPE_Q5_1:
            case ggml_type.GGML_TYPE_Q8_0:
            case ggml_type.GGML_TYPE_Q8_1:
            {
                ggml_compute_forward_mul_mat_q_f32(@params, src0, src1, dst);
            } break;
            case ggml_type.GGML_TYPE_F16:
            {
                ggml_compute_forward_mul_mat_f16_f32(@params, src0, src1, dst);
            } break;
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_mul_mat_f32(@params, src0, src1, dst);
            } break;
            default:
            {
                Debug.Assert(false);
            } break;
        }
    }

    static void ggml_compute_forward_scale_f32(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        Debug.Assert(ggml_is_contiguous(src0));
        Debug.Assert(ggml_is_contiguous(dst));
        Debug.Assert(ggml_are_same_shape(src0, dst));
        Debug.Assert(ggml_is_scalar(src1));

        if (@params->type == ggml_task_type.GGML_TASK_INIT || @params->type == ggml_task_type.GGML_TASK_FINALIZE)
        {
            return;
        }

        // scale factor
        float v = *(float *) src1->data;

        int ith = @params->ith;
        int nth = @params->nth;

        int nc = (int)src0->ne[0];
        int nr = ggml_nrows(src0);

        // rows per thread
        int dr = (nr + nth - 1) / nth;

        // row range for this thread
        int ir0 = dr * ith;
        nuint ir1 = (nuint)Math.Min(ir0 + dr, nr);

        for (nuint i1 = (nuint)ir0; i1 < ir1; i1++)
        {
            ggml_vec_scale_f32((int)nc, (float*)((byte*)dst->data + i1 * (dst->nb[1])), v);
        }
    }

    static void ggml_compute_forward_scale(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* src1,
        ggml_tensor* dst)
    {
        switch (src0->type)
        {
            case ggml_type.GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(@params, src0, src1, dst);
            }
                break;
            default:
            {
                Debug.Assert(false);
            }
                break;
        }
    }

    static void ggml_compute_forward_cpy(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        ggml_compute_forward_dup(@params, src0, dst);
    }

    static void ggml_compute_forward_cont(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
        ggml_compute_forward_dup(@params, src0, dst);
    }

    static void ggml_compute_forward_reshape(
        ggml_compute_params* @params,
        ggml_tensor* src0,
        ggml_tensor* dst)
    {
    }

    static void ggml_compute_forward_view(
        ggml_compute_params* @params,
        ggml_tensor* src0)
    {
    }

    static void ggml_compute_forward_permute(
        ggml_compute_params* @params,
        ggml_tensor* src0)
    {
    }

    static void ggml_compute_forward_transpose(
        ggml_compute_params* @params,
        ggml_tensor* src0)
    {
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
            ggml_tensor* current_node = ggml_cgraph.get_node(cgraph, i);
            if (current_node == node)
            {
                return;
            }
        }

        for (int i = 0; i < cgraph->n_leafs; i++)
        {
            if (ggml_cgraph.get_leaf(cgraph, i) == node)
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

            ggml_cgraph.set_leaf(cgraph, cgraph->n_leafs, node);
            cgraph->n_leafs++;
        }
        else
        {
            Debug.Assert(cgraph->n_nodes < GGML_MAX_NODES);

            ggml_cgraph.set_node(cgraph, cgraph->n_nodes, node);
            ggml_cgraph.set_grad(cgraph, cgraph->n_nodes, node->grad);
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
            ggml_tensor* last_node = ggml_cgraph.get_node(cgraph, cgraph->n_nodes - 1);
            Debug.Assert(last_node == tensor);
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
                ggml_tensor* node = ggml_cgraph.get_node(gf, i);

                if (node->grad is not null)
                {
                    node->grad = ggml_dup_tensor(ctx, node);
                    ggml_cgraph.set_grad(gf, i, node->grad);
                }
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = ggml_cgraph.get_node(gf, i);

            // because we detached the grad nodes from the original graph, we can afford inplace operations
            if (node->grad is not null)
            {
                ggml_compute_backward(ctx, node, keep);
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = ggml_cgraph.get_node(gf, i);

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
                GGML_PRINT(
                    $"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + size_needed + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
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
                GGML_PRINT(
                    $"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
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

    static bool ggml_is_quantized(ggml_type type)
    {
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

    static System.Span<TElement> InlineArrayAsSpan<TBuffer, TElement>(ref TBuffer buffer, int size)
        where TBuffer : struct
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

    static void ggml_graph_compute_thread(object? data)
    {
        ggml_compute_state state = (ggml_compute_state)data!;
        int n_threads = state.shared->n_threads;
        ggml_compute_state_shared* shared = state.shared;
        while (true)
        {
            if (atomic_fetch_add(ref state.shared->n_ready, 1) == n_threads - 1)
            {
                atomic_store(ref state.shared->has_work, 0);
            }
            else
            {
                while (atomic_load(ref state.shared->has_work) != 0)
                {
                    if (atomic_load(ref state.shared->stop) != 0)
                    {
                        return;
                    }

                    ggml_lock_lock(&shared->spin);
                    ggml_lock_unlock(&shared->spin);
                }
            }

            atomic_fetch_sub(ref state.shared->n_ready, 1);

            // wait for work
            while (atomic_load(ref state.shared->has_work) == 0)
            {
                if (atomic_load(ref state.shared->stop) != 0)
                {
                    return;
                }

                ggml_lock_lock(&state.shared->spin);
                ggml_lock_unlock(&state.shared->spin);
            }

            // check if we should stop
            if (atomic_load(ref state.shared->stop) != 0)
            {
                break;
            }

            if (state.node is not null)
            {
                if (state.@params.ith < state.@params.nth)
                {
                    ggml_compute_forward(ref state.@params, state.node);
                }

                state.node = null;
            }
            else
            {
                break;
            }
        }
    }

    static void ggml_compute_forward(ggml_compute_params* @params, ggml_tensor* tensor)
    {
        Debug.Assert(@params is not null);

        switch (tensor->op)
        {
            case ggml_op.GGML_OP_DUP:
            {
                ggml_compute_forward_dup(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_ADD:
            {
                ggml_compute_forward_add(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_SUB:
            {
                ggml_compute_forward_sub(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_MUL:
            {
                ggml_compute_forward_mul(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_DIV:
            {
                ggml_compute_forward_div(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_SUM:
            {
                ggml_compute_forward_sum(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_ABS:
            {
                ggml_compute_forward_abs(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_SGN:
            {
                ggml_compute_forward_sgn(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_NEG:
            {
                ggml_compute_forward_neg(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_STEP:
            {
                ggml_compute_forward_step(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_RELU:
            {
                ggml_compute_forward_relu(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_GELU:
            {
                ggml_compute_forward_gelu(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_SILU:
            {
                ggml_compute_forward_silu(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_NORM:
            {
                ggml_compute_forward_norm(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(@params, tensor->src0, tensor->src1, tensor);
            }
                break;
            case ggml_op.GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_CONT:
            {
                ggml_compute_forward_cont(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(@params, tensor->src0, tensor);
            }
                break;
            case ggml_op.GGML_OP_VIEW:
            {
                ggml_compute_forward_view(@params, tensor->src0);
            }
                break;
            case ggml_op.GGML_OP_PERMUTE:
            {
                ggml_compute_forward_permute(@params, tensor->src0);
            }
                break;
            case ggml_op.GGML_OP_TRANSPOSE:
            {
                ggml_compute_forward_transpose(@params, tensor->src0);
            }
                break;
            // case ggml_op.GGML_OP_GET_ROWS:
            //     {
            //         ggml_compute_forward_get_rows(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_DIAG_MASK_INF:
            //     {
            //         ggml_compute_forward_diag_mask_inf(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_SOFT_MAX:
            //     {
            //         ggml_compute_forward_soft_max(@params, tensor->src0, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_ROPE:
            //     {
            //         ggml_compute_forward_rope(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_ALIBI:
            //     {
            //         ggml_compute_forward_alibi(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_CONV_1D_1S:
            //     {
            //         ggml_compute_forward_conv_1d_1s(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_CONV_1D_2S:
            //     {
            //         ggml_compute_forward_conv_1d_2s(@params, tensor->src0, tensor->src1, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_FLASH_ATTN:
            //     {
            //         int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
            //         Debug.Assert(t == 0 || t == 1);
            //         bool masked = t != 0;
            //         ggml_compute_forward_flash_attn(@params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_FLASH_FF:
            //     {
            //         ggml_compute_forward_flash_ff(@params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
            //     }
            //     break;
            // case ggml_op.GGML_OP_MAP_UNARY:
            //     {
            //         ggml_unary_op_f32_t fun = *((ggml_unary_op_f32_t*)tensor->opt[0]->data);
            //         ggml_compute_forward_map_unary(@params, tensor->src0, tensor, fun);
            //     }
            //     break;
            // case ggml_op.GGML_OP_MAP_BINARY:
            //     {
            //         ggml_binary_op_f32_t fun = *((ggml_binary_op_f32_t*)tensor->opt[0]->data);
            //         ggml_compute_forward_map_binary(@params, tensor->src0, tensor->src1, tensor, fun);
            //     }
            //     break;
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

    static void ggml_compute_forward(ref ggml_compute_params @params, ggml_tensor* tensor)
    {
        fixed (ggml_compute_params* p = &@params)
            ggml_compute_forward(p, tensor);
    }

    static void print_tensor_data(void* data, int NF)
    {
        // print results
        for (int i = 0; i < 16; i++)
        {
            Console.WriteLine($"x[{i,3}] = {((float*)data)[i]}");
        }

        Console.WriteLine("...");
        for (int i = NF - 16; i < NF; i++)
        {
            Console.WriteLine($"x[{i,3}] = {((float*)data)[i]}");
        }

        Console.WriteLine();
    }
}