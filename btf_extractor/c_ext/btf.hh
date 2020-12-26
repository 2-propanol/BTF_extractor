/*   The MIT License
*
*   Free BTF Library 
*   Copyright (c) 2016 Zdravko Velinov
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in
*   all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*   THE SOFTWARE.
*/

#ifndef _FREE_BTF_HH_
#define _FREE_BTF_HH_

#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <assert.h>

#define BTF_MAX_CHANNEL_COUNT 3

struct BTF;
struct BTFExtra;

struct Vector2
{
    float x, y;
};

struct Vector3
{
    float x, y, z;
};

struct Matrix3
{
    Vector3 tangent, binormal, normal;
};

typedef Vector3 Spectrum;

inline Spectrum BTFFetchSpectrum(const BTF *btf, uint32_t light_vert, uint32_t view_vert, uint32_t x, uint32_t y);
inline Spectrum BTFFetchSpectrum(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx);
void BTFFetchChannelsSingleFloat(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **multi_chan_result);
void BTFFetchChannelsSIMD(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **multi_chan_result);
void BTFEvaluateMatrixHalfFloatToFloat(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **result);

BTF *LoadBTF(const char *file_path, BTFExtra *extra = nullptr);
void DestroyBTF(BTF *btf);

struct BTFCommonHeader
{
    uint32_t Size;
    uint32_t Version;
    char MeasurementSetup[80];
    char ImageSensor[80];
    char LightSource[80];
    float PPMM;
    Vector3 RGB;
};

struct BTFExtra
{
    BTFCommonHeader Header;
    std::string XMLString;
    std::vector<std::string> Channels;
    std::vector<Matrix3> Rotations;
};

struct Edge
{
    uint32_t Index0,
        Index1;
    uint32_t Triangle;
    float Angle0;
    float Angle1;
};

struct ThreadedBVHNode2;

struct BTF
{
    uint32_t ConsineFlag = false;
    uint32_t ChannelCount = 3;
    uint32_t Width = 0,
             Height = 0;
    uint32_t DynamicRangeReduction = false;
    struct
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
    } HeightMapSize;
    uint16_t *HeightMap = nullptr;

    uint32_t ColorModel = 0;
    Vector3 ColorMean;
    Matrix3 ColorTransform;

    uint32_t RowCount = 0,
             ColumnCount = 0,
             DataSize = 0;

    uint8_t *LeftSingularU = nullptr,
            *RightSingularSxV = nullptr;

    uint64_t LeftSingularUSize = 0,
             RightSingularSxVSize = 0;

    Vector3 *Views = nullptr;
    Vector3 *Lights = nullptr;

    uint32_t ViewCount = 0;
    uint32_t LightCount = 0;

    uint32_t *LightIndices = nullptr;
    uint32_t LightTriangleCount = 0;

    uint32_t UElementStride = 0,
             SxVElementStride = 0;

    uint32_t *Offsets = nullptr;
    uint32_t *ComponentCounts = nullptr;
};

struct BTFDeleter
{
    inline void operator()(BTF *btf) { DestroyBTF(btf); }
};

typedef std::unique_ptr<BTF, BTFDeleter> BTFPtr;

//#	define BTFFetchChannels BTFFetchChannelsSingleFloat
#define BTFFetchChannels BTFFetchChannelsSIMD

inline float Dot(const Vector3 &lhs, const Vector3 &rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vector3 operator*(const Vector3 &lhs, const Vector3 &rhs) { return Vector3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z}; }

inline Vector3 operator+(const Vector3 &lhs, const Vector3 &rhs) { return Vector3{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }

inline Vector3 operator-(const Vector3 &lhs, const Vector3 &rhs) { return Vector3{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}; }

inline Vector3 operator-(const Vector3 &vec, float scalar) { return Vector3{vec.x - scalar, vec.y - scalar, vec.z - scalar}; }

inline float Dot(const Vector2 &lhs, const Vector2 &rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }

inline Vector3 ParabolicToCartesianCoordinates(const Vector2 &coordinates)
{
    float len_sq = Dot(coordinates, coordinates);
    return {2.0f * coordinates.x / (len_sq + 1.0f), 2.0f * coordinates.y / (len_sq + 1.0f), (1 - len_sq) / (1 + len_sq)};
}

template <class T, class TOp>
inline void BTFEvaluateMatrix(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **result);

inline Spectrum BTFFetchSpectrum(const BTF *btf, uint32_t light_vert, uint32_t view_vert, uint32_t x, uint32_t y)
{
    assert(light_vert < btf->LightCount && view_vert < btf->LightCount &&
           x < btf->Width && y < btf->Height && "Invalid index");
    uint32_t xy_idx = y * btf->Width + x;
    uint32_t lv_idx = view_vert * btf->LightCount + light_vert;

    return BTFFetchSpectrum(btf, lv_idx, xy_idx);
}

inline float Clampf(float val, float minval, float maxval)
{
    return std::fmax(std::fmin(val, maxval), minval);
}

inline Vector3 Clamp(const Vector3 &val, float smin, float smax)
{
    return Vector3{Clampf(val.x, smin, smax),
                   Clampf(val.y, smin, smax),
                   Clampf(val.z, smin, smax)};
}

inline Vector3 YUVToRGB(const Vector3 &color)
{
    Vector3 vx{1.0f, 0.0f, 1.13983f};
    Vector3 vy{1.0f, -0.39465f, -0.5806f};
    Vector3 vz{1.0f, 2.03211f, 0.0f};

    return Clamp(Vector3{Dot(vx, color), Dot(vy, color), Dot(vz, color)}, 0.0f, 1.0f);
}

inline Vector3 Exp(const Vector3 &val)
{
    return Vector3{expf(val.x), expf(val.y), expf(val.z)};
}

inline Spectrum BTFFetchSpectrum(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx)
{
    uint32_t chan_count = btf->ChannelCount;

    if (chan_count > BTF_MAX_CHANNEL_COUNT)
        return {};

    float multi_chan_result[BTF_MAX_CHANNEL_COUNT];
    memset(multi_chan_result, 0, BTF_MAX_CHANNEL_COUNT * sizeof(float));

    float *res_ptr = multi_chan_result;
    BTFFetchChannels(btf, lv_idx, xy_idx, &res_ptr);

    Spectrum result{};
    float drr_eps = 1e-5f;

    switch (chan_count)
    {
    case 3:
    {
        Vector3 interm_color = *reinterpret_cast<Vector3 *>(multi_chan_result);

        switch (btf->ColorModel)
        {
        case 0:
            break;
        case 11:
        {
            Vector3 yuv;
            yuv.x = expf(interm_color.x) - drr_eps;
            yuv.y = interm_color.y * yuv.x + drr_eps;
            yuv.z = interm_color.z * yuv.x + drr_eps;
            interm_color = YUVToRGB(yuv);
        }
        break;
        default:
        {
            assert(false && "Unsupported color model");
        }
        }

        if (btf->DynamicRangeReduction)
        {
            interm_color = Exp(interm_color) - drr_eps;
        }
        result = interm_color;
    }
    break;
    default:
    {
        assert(false && "Conversion is unsupported");
    }
    break;
    }

    if (btf->ConsineFlag)
    {
        assert(false && "Stub");
        // TODO
    }

    return result;
}

inline void BTFFetchChannelsSingleFloat(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **multi_chan_result)
{
    switch (btf->DataSize)
    {
        //    case 2:
        //    {
        //        BTFEvaluateMatrix<half, float>(btf, lv_idx, xy_idx, multi_chan_result);
        //    } break;
    case 4:
    {
        BTFEvaluateMatrix<float, float>(btf, lv_idx, xy_idx, multi_chan_result);
    }
    break;
    case 8:
    {
        BTFEvaluateMatrix<double, double>(btf, lv_idx, xy_idx, multi_chan_result);
    }
    break;
    default:
    {
        assert(!"Unsupported");
    }
    }
}

inline void BTFFetchChannelsSIMD(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **multi_chan_result)
{
    switch (btf->DataSize)
    {
    case 2:
    {
        BTFEvaluateMatrixHalfFloatToFloat(btf, lv_idx, xy_idx, multi_chan_result);
    }
    break;
    case 4:
    {
        BTFEvaluateMatrix<float, float>(btf, lv_idx, xy_idx, multi_chan_result); // TODO: SIMD
    }
    break;
    case 8:
    {
        BTFEvaluateMatrix<double, double>(btf, lv_idx, xy_idx, multi_chan_result); // TODO: SIMD
    }
    break;
    default:
    {
        assert(!"Unsupported");
    }
    }
}

#endif // _FREE_BTF_HH_

#ifdef BTF_IMPLEMENTATION
#include <xmmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <algorithm>

template <class T, class TOp>
inline void BTFEvaluateMatrix(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **result)
{
    uint32_t chan_count = btf->ChannelCount;
    uint32_t *u_offsets = btf->Offsets;
    uint32_t *sxv_offsets = btf->Offsets + btf->ChannelCount;
    for (uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
    {
        auto component_count = btf->ComponentCounts[chan_idx];
        uint32_t SxV_elem_offset = xy_idx * component_count * btf->SxVElementStride;
        uint32_t SxV_offset = sxv_offsets[chan_idx];
        T *SxVslice = reinterpret_cast<T *>(btf->RightSingularSxV + SxV_offset + SxV_elem_offset);

        uint32_t u_elem_offset = lv_idx * component_count * btf->UElementStride;
        uint32_t u_offset = u_offsets[chan_idx];
        T *Uslice = reinterpret_cast<T *>(btf->LeftSingularU + u_offset + u_elem_offset);
        TOp sum{};
        for (uint32_t comp_idx = 0; comp_idx < component_count; ++comp_idx)
        {
            auto u = static_cast<TOp>(Uslice[comp_idx]);
            auto v = static_cast<TOp>(SxVslice[comp_idx]);
            assert(std::isfinite(u) && std::isfinite(v) && "invalid component");
            sum += u * v;
        }
        (*result)[chan_idx] = static_cast<float>(sum);
    }
}

inline void BTFEvaluateMatrixHalfFloatToFloat(const BTF *btf, uint32_t lv_idx, uint32_t xy_idx, float **result)
{
    uint32_t chan_count = btf->ChannelCount;
    uint32_t *u_offsets = btf->Offsets;
    uint32_t *sxv_offsets = btf->Offsets + btf->ChannelCount;
    for (uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
    {
        auto component_count = btf->ComponentCounts[chan_idx];
        uint32_t SxV_idx = xy_idx * component_count * btf->SxVElementStride;
        uint32_t SxV_offset = sxv_offsets[chan_idx];
        auto SxVslice = btf->RightSingularSxV + SxV_idx + SxV_offset;

        uint32_t u_idx = lv_idx * component_count * btf->UElementStride;
        uint32_t u_offset = u_offsets[chan_idx];
        auto Uslice = btf->LeftSingularU + u_idx + u_offset;

        __m256 sum = _mm256_setzero_ps();
        const uint32_t m128_fp16_count = sizeof(__m128) / sizeof(uint16_t);
        uint32_t steps = component_count / m128_fp16_count;
        for (uint32_t step_idx = 0; step_idx < steps; ++step_idx)
        {
            uint32_t offset = step_idx * sizeof(__m128);
            __m128i U_vec_half = _mm_loadu_si128(reinterpret_cast<__m128i *>(Uslice + offset));
            __m128i SxV_vec_half = _mm_loadu_si128(reinterpret_cast<__m128i *>(SxVslice + offset));

            __m256 U_vec = _mm256_cvtph_ps(U_vec_half);
            __m256 V_vec = _mm256_cvtph_ps(SxV_vec_half);

            __m256 USV = _mm256_mul_ps(U_vec, V_vec);

            sum = _mm256_add_ps(sum, USV);
        }

        union partial_m128
        {
            uint16_t value[m128_fp16_count];
            __m128i pack;
        } partial_U_half, partial_SxV_half;

        partial_U_half.pack = _mm_setzero_si128();
        partial_SxV_half.pack = _mm_setzero_si128();

        for (uint32_t comp_idx = steps * m128_fp16_count, part_idx = 0; comp_idx < component_count; ++comp_idx, ++part_idx)
        {
            partial_U_half.value[part_idx] = reinterpret_cast<uint16_t *>(Uslice)[comp_idx];
            partial_SxV_half.value[part_idx] = reinterpret_cast<uint16_t *>(SxVslice)[comp_idx];
        }

        __m256 partial_U = _mm256_cvtph_ps(partial_U_half.pack);
        __m256 partial_SxV = _mm256_cvtph_ps(partial_SxV_half.pack);

        __m256 partial_USV = _mm256_mul_ps(partial_U, partial_SxV);

        sum = _mm256_add_ps(sum, partial_USV);

        __m128 sum_lo = _mm256_extractf128_ps(sum, 0);
        __m128 sum_hi = _mm256_extractf128_ps(sum, 1);

        __m128 sum_reduce = _mm_add_ps(sum_lo, sum_hi);
        sum_reduce = _mm_hadd_ps(sum_reduce, sum_reduce);
        sum_reduce = _mm_hadd_ps(sum_reduce, sum_reduce);

        float final_sum = _mm_cvtss_f32(sum_reduce);

        (*result)[chan_idx] = final_sum;
    }
}

void DestroyBTF(BTF *btf)
{
    delete[] btf->HeightMap;
    btf->HeightMap = nullptr;
    delete[] btf->LeftSingularU;
    btf->LeftSingularU = nullptr;
    delete[] btf->RightSingularSxV;
    btf->RightSingularSxV = nullptr;
    delete[] btf->Views;
    btf->Views = nullptr;
    delete[] btf->Lights;
    btf->Lights = nullptr;
    delete[] btf->Offsets;
    btf->Offsets = nullptr;
    delete[] btf->ComponentCounts;
    btf->ComponentCounts = nullptr;
    free(btf->LightIndices);
    btf->LightIndices = nullptr;
}

#define BTF_MAKE_FOURCC(x, y, z, w) \
    (((x & 0xFFU)) |                \
     ((y & 0xFFU) << 8U) |          \
     ((z & 0xFFU) << 16U) |         \
     ((w & 0xFFU) << 24U))

#define BTF_MAKE_EIGHTCC(c0, c1, c2, c3, c4, c5, c6, c7, c8) \
    (((c0 & 0xFFU))       | \
    ((c1 & 0xFFU) << 8U)  | \
    ((c2 & 0xFFU) << 16U) | \
    ((c3 & 0xFFU) << 24U))| \
    ((c4 & 0xFFU) << 32U))| \
    ((c5 & 0xFFU) << 40U))| \
    ((c6 & 0xFFU) << 48U))| \
    ((c7 & 0xFFU) << 56U))

inline void SinCos_Tan(float omega, float *s, float *c)
{
    float t = tanf(omega * 0.5f);
    float t2 = t * t;
    float div = 1.0f / (1.0f + t2);
    *s = 2.0f * t * div;
    *c = (1.0f - t2) * div;
}

#define FastSinCos SinCos_Tan

inline Vector3 SphereToCartesianCoordinates(const Vector2 &angles)
{
    float cos_theta, sin_theta, cos_phi, sin_phi;
    FastSinCos(angles.x, &sin_theta, &cos_theta);
    FastSinCos(angles.y, &sin_phi, &cos_phi);

    return {cos_phi * sin_theta, sin_phi * sin_theta, cos_theta};
}

BTF *LoadBTF(const char *filename, BTFExtra *out_extra)
{
    BTFExtra extra;
    BTFPtr btf(new BTF, BTFDeleter());
    std::fstream fs(filename, std::ios::binary | std::ios::in);

    if (!fs)
    {
        return nullptr;
    }

    char c = fs.get();
    if (c != '!')
    {
        return nullptr;
    }

    uint32_t signature;
    fs.read(reinterpret_cast<char *>(&signature), sizeof(signature));

    auto fmf1_signature = BTF_MAKE_FOURCC('F', 'M', 'F', '1');

    auto read_common = [&fs, &extra, &btf, filename]() {
        bool rotations_included = false;
        auto c = fs.get();
        if (c != 'R')
        {
            fs.unget();
        }
        else
        {
            rotations_included = true;
        }

        auto start_off = fs.tellg();
        fs.read(reinterpret_cast<char *>(&extra.Header), sizeof(extra.Header));

        if (extra.Header.Version > 1)
        {
            fs.read(reinterpret_cast<char *>(&btf->ConsineFlag), sizeof(btf->ConsineFlag));
        }

        if (extra.Header.Version > 2)
        {
            uint32_t len;
            fs.read(reinterpret_cast<char *>(&len), sizeof(len));
            if (len)
            {
                extra.XMLString.resize(len);
                fs.read(&extra.XMLString.front(), len);
            }
        }

        if (extra.Header.Version > 3)
        {
            fs.read(reinterpret_cast<char *>(&btf->ChannelCount), sizeof(btf->ChannelCount));
            extra.Channels.resize(btf->ChannelCount);

            for (uint32_t chan_idx = 0; chan_idx < btf->ChannelCount; ++chan_idx)
            {
                uint32_t chan_size;
                fs.read(reinterpret_cast<char *>(&chan_size), sizeof(chan_size));
                auto &chan = extra.Channels[chan_idx];
                chan.resize(chan_size);
                fs.read(&chan.front(), chan_size);
            }
        }
        else
        {
            extra.Channels = {"R", "G", "B"};
        }

        btf->Offsets = new uint32_t[2 * btf->ChannelCount];
        btf->ComponentCounts = new uint32_t[btf->ChannelCount];

        auto header_read = fs.tellg() - start_off;
        if (header_read != extra.Header.Size)
        {
            return false;
        }

        uint32_t view_count;
        fs.read(reinterpret_cast<char *>(&view_count), sizeof(view_count));
        std::unique_ptr<Vector3[]> views(new Vector3[view_count]);
        btf->ViewCount = view_count;
        std::unique_ptr<Vector3[]> lights;

        for (uint32_t view_idx = 0; view_idx < view_count; ++view_idx)
        {
            Vector2 angles;
            fs.read(reinterpret_cast<char *>(&angles), sizeof(angles));

            views[view_idx] = SphereToCartesianCoordinates(angles);

            uint32_t num_lights;
            fs.read(reinterpret_cast<char *>(&num_lights), sizeof(num_lights));

            if (btf->LightCount == 0)
            {
                btf->LightCount = num_lights;
                lights = std::unique_ptr<Vector3[]>(new Vector3[btf->LightCount]);
                for (uint32_t light_idx = 0; light_idx < btf->LightCount; ++light_idx)
                {
                    fs.read(reinterpret_cast<char *>(&angles), sizeof(angles));
                    lights[light_idx] = SphereToCartesianCoordinates(angles);
                }
            }
            else if (btf->LightCount == num_lights)
            {
                fs.seekg(sizeof(angles) * btf->LightCount, std::ios::cur);
            }
            else
            {
                return false;
            }
        }

        fs.read(reinterpret_cast<char *>(&btf->Width), sizeof(btf->Width));
        fs.read(reinterpret_cast<char *>(&btf->Height), sizeof(btf->Height));

        if (rotations_included)
        {
            uint32_t num_rotations;
            fs.read(reinterpret_cast<char *>(&num_rotations), sizeof(num_rotations));

            if (num_rotations)
            {
                extra.Rotations.resize(num_rotations);
                fs.read(reinterpret_cast<char *>(&extra.Rotations.front()), extra.Rotations.size() * sizeof(extra.Rotations.front()));
            }
        }

        btf->Views = views.release();
        btf->Lights = lights.release();

        return true;
    };

    switch (signature)
    {
    case BTF_MAKE_FOURCC('D', 'F', 'M', 'F'):
    {
        bool rotations_included = false, ext_header = false;

        const char *expected = "08FC";
        for (uint32_t i = 0; expected[i]; ++i)
            if (expected[i] != (c = fs.get()))
            {
                return nullptr;
            }

        if (!read_common())
            return nullptr;

        uint32_t num_components;
        fs.read(reinterpret_cast<char *>(&num_components), sizeof(num_components));

        fs.read(reinterpret_cast<char *>(&btf->ColorModel), sizeof(btf->ColorModel));
        fs.read(reinterpret_cast<char *>(&btf->ColorMean), sizeof(btf->ColorMean));
        static_assert(sizeof(Matrix3) == 3 * 3 * sizeof(float), "Invalid matrix size");
        fs.read(reinterpret_cast<char *>(&btf->ColorTransform), sizeof(btf->ColorTransform));

        btf->DataSize = 0;

        auto light_count = btf->LightCount;
        auto expected_row_count = light_count * light_count;
        auto expected_column_count = btf->Width * btf->Height;

        btf->RowCount = expected_row_count;
        btf->ColumnCount = expected_column_count;

        uint32_t u_plane_offset = 0, sxv_plane_offset = 0;

        struct DeleteSubelements
        {
            uint32_t SubDataCount;
            DeleteSubelements(uint32_t count)
                : SubDataCount(count) {}
            void operator()(uint8_t **data)
            {
                for (uint32_t i = 0; i < SubDataCount; ++i)
                {
                    delete[] data[i];
                }
                delete[] data;
            }
        };

        std::unique_ptr<uint8_t *[], DeleteSubelements> matrix_planes(new uint8_t *[2 * btf->ChannelCount], DeleteSubelements(2 * btf->ChannelCount));

        for (uint32_t chan_idx = 0, chan_idx_end = btf->ChannelCount; chan_idx < chan_idx_end; ++chan_idx)
        {
            uint8_t scalar_size = fs.get();

            if (btf->DataSize == 0)
            {
                btf->DataSize = scalar_size;
            }
            else if (btf->DataSize != scalar_size)
            {
                assert(!"Stub");
                return nullptr;
            }

            fs.read(reinterpret_cast<char *>(&num_components), sizeof(num_components));

            auto num_component = std::max(1u, std::min(num_components, num_components));

            btf->ComponentCounts[chan_idx] = num_component;

            uint32_t slice_size = num_component * btf->ChannelCount * btf->DataSize;

            auto u_plane = matrix_planes[chan_idx] = new uint8_t[expected_row_count * slice_size];
            auto SxV_plane = matrix_planes[btf->ChannelCount + chan_idx] = new uint8_t[expected_column_count * slice_size];

            uint32_t num_row, num_column;
            fs.read(reinterpret_cast<char *>(&num_row), sizeof(num_row));
            fs.read(reinterpret_cast<char *>(&num_column), sizeof(num_column));

            if (num_row != expected_row_count)
            {
                return nullptr;
            }

            if (num_column != expected_column_count)
            {
                return nullptr;
            }

            size_t scalars_size = num_components * btf->DataSize;
            std::unique_ptr<uint8_t[]> scalars(new uint8_t[scalars_size]);
            fs.read(reinterpret_cast<char *>(scalars.get()), scalars_size);

            if (!fs)
            {
                return nullptr;
            }

            uint32_t left_singular_size = btf->RowCount * num_components * btf->DataSize;
            fs.read(reinterpret_cast<char *>(u_plane), left_singular_size);

            if (!fs)
            {
                return nullptr;
            }

            uint32_t right_singular_size = btf->ColumnCount * num_components * btf->DataSize;
            fs.read(reinterpret_cast<char *>(SxV_plane), right_singular_size);

            if (!fs)
            {
                return nullptr;
            }

            btf->Offsets[chan_idx] = u_plane_offset;
            u_plane_offset += left_singular_size;
            btf->Offsets[btf->ChannelCount + chan_idx] = sxv_plane_offset;
            sxv_plane_offset += right_singular_size;
        }

        btf->LeftSingularUSize = u_plane_offset;
        btf->RightSingularSxVSize = sxv_plane_offset;

        uint32_t plane_idx = 0;
        {
            btf->LeftSingularU = new uint8_t[u_plane_offset];
            uint32_t offset = 0;
            for (uint32_t end_idx = btf->ChannelCount; plane_idx < end_idx; ++plane_idx)
            {
                auto size = btf->RowCount * btf->ComponentCounts[plane_idx] * btf->DataSize;
                memcpy(btf->LeftSingularU + offset, matrix_planes[plane_idx], size);
                offset += size;
            }
            assert(offset == u_plane_offset && "Invalid data offset");
        }

        {
            btf->RightSingularSxV = new uint8_t[sxv_plane_offset];
            uint32_t offset = 0;
            for (uint32_t end_idx = 2 * btf->ChannelCount, comp_idx = 0; plane_idx < end_idx; ++plane_idx, ++comp_idx)
            {
                auto size = btf->ColumnCount * btf->ComponentCounts[comp_idx] * btf->DataSize;
                memcpy(btf->RightSingularSxV + offset, matrix_planes[plane_idx], size);
                offset += size;
            }
            assert(offset == sxv_plane_offset && "Invalid data offset");
        }

        auto end_pos = fs.tellg();
        fs.seekg(0, std::ios::end);
        auto actual_end_pos = fs.tellg();
        if (end_pos != actual_end_pos)
        {
            return nullptr;
        }

        btf->UElementStride = btf->DataSize;
        btf->SxVElementStride = btf->DataSize;
    }
    break;
    case BTF_MAKE_FOURCC('F', 'M', 'F', '0'):
    case BTF_MAKE_FOURCC('F', 'M', 'F', '1'):
    {
        bool rotations_included = false, ext_header = false;
        const char *expected;
        if (signature == BTF_MAKE_FOURCC('F', 'M', 'F', '1'))
        {
            ext_header = true;
            expected = "2FCE";
        }
        else
            expected = "6FC";

        for (uint32_t i = 0; expected[i]; ++i)
            if (expected[i] != (c = fs.get()))
            {
                return nullptr;
            }

        if (!read_common())
            return nullptr;

        if (ext_header)
        {
            char header_version = fs.get();
            if (header_version >= 1)
            {
                fs.read(reinterpret_cast<char *>(&btf->DynamicRangeReduction), sizeof(btf->DynamicRangeReduction));
            }

            if (header_version >= 2)
            {
                fs.read(reinterpret_cast<char *>(&btf->HeightMapSize), sizeof(btf->HeightMapSize));
                if (btf->HeightMapSize.Width && btf->HeightMapSize.Height)
                {
                    uint32_t tex_area = btf->HeightMapSize.Width * btf->HeightMapSize.Height;
                    btf->HeightMap = new uint16_t[tex_area];

                    fs.read(reinterpret_cast<char *>(btf->HeightMap), sizeof(btf->HeightMap[0]) * tex_area);
                }
            }

            if (header_version >= 3)
            {
                return nullptr;
            }
        }

        uint32_t num_components0, num_components1;
        fs.read(reinterpret_cast<char *>(&num_components0), sizeof(num_components0));

        btf->DataSize = fs.get();

        fs.read(reinterpret_cast<char *>(&num_components1), sizeof(num_components1));

        auto num_components = std::max(1u, std::min(num_components0, num_components1));
        for (uint32_t i = 0; i < btf->ChannelCount; ++i)
        {
            btf->ComponentCounts[i] = num_components;
        }
        fs.read(reinterpret_cast<char *>(&btf->RowCount), sizeof(btf->RowCount));
        fs.read(reinterpret_cast<char *>(&btf->ColumnCount), sizeof(btf->ColumnCount));

        if (!fs)
        {
            return nullptr;
        }

        size_t scalars_size = num_components * btf->DataSize;
        std::unique_ptr<uint8_t[]> scalars(new uint8_t[scalars_size]);
        fs.read(reinterpret_cast<char *>(scalars.get()), scalars_size);

        if (!fs)
        {
            return nullptr;
        }

        auto light_count = btf->LightCount;
        if (btf->RowCount != light_count * light_count * btf->ChannelCount)
        {
            return nullptr;
        }

        if (btf->ColumnCount != btf->Width * btf->Height)
        {
            return nullptr;
        }

        size_t left_singular_size = btf->RowCount * num_components * btf->DataSize;
        btf->LeftSingularU = new uint8_t[left_singular_size];
        btf->LeftSingularUSize = left_singular_size;
        fs.read(reinterpret_cast<char *>(btf->LeftSingularU), left_singular_size);

        if (!fs)
        {
            return nullptr;
        }

        size_t right_singular_size = btf->ColumnCount * num_components * btf->DataSize;
        btf->RightSingularSxVSize = right_singular_size;
        btf->RightSingularSxV = new uint8_t[right_singular_size];
        fs.read(reinterpret_cast<char *>(btf->RightSingularSxV), right_singular_size);

        if (!fs)
        {
            return nullptr;
        }

        for (uint32_t idx = 0; idx < btf->ChannelCount; ++idx)
        {
            btf->Offsets[idx] = idx * num_components * btf->DataSize;
        }

        uint32_t offset = btf->ChannelCount;
        for (uint32_t idx = 0; idx < btf->ChannelCount; ++idx)
        {
            btf->Offsets[offset + idx] = 0;
        }

        btf->UElementStride = btf->ChannelCount * btf->DataSize;
        btf->SxVElementStride = btf->DataSize;

        auto end_pos = fs.tellg();
        fs.seekg(0, std::ios::end);
        auto actual_end_pos = fs.tellg();
        if (end_pos != actual_end_pos)
        {
            return nullptr;
        }
    }
    break;
    case BTF_MAKE_FOURCC('P', 'V', 'F', '0'):
    {
        assert(!"Stub");
        return nullptr;
    }
    break;
    case BTF_MAKE_FOURCC('B', 'D', 'I', 'F'):
    {
        assert(!"Stub");
        return nullptr;
    }
    break;
    }

    // Flip origin
    uint32_t chan_count = btf->ChannelCount;
    uint32_t *sxv_offsets = btf->Offsets + btf->ChannelCount;
    for (uint32_t y = 0, btf_height = btf->Height, y_end = btf_height / 2; y < y_end; ++y)
    {
        for (uint32_t x = 0, btf_width = btf->Width; x < btf_width; ++x)
        {
            uint32_t src_xy_idx = y * btf_width + x;
            uint32_t dst_xy_idx = (btf_height - 1 - y) * btf_width + x;
            for (uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
            {
                auto component_count = btf->ComponentCounts[chan_idx];
                uint32_t src_SxV_elem_offset = src_xy_idx * component_count * btf->SxVElementStride;
                uint32_t dst_SxV_elem_offset = dst_xy_idx * component_count * btf->SxVElementStride;
                uint32_t SxV_offset = sxv_offsets[chan_idx];
                auto src_SxVslice = btf->RightSingularSxV + SxV_offset + src_SxV_elem_offset;
                auto dst_SxVslice = btf->RightSingularSxV + SxV_offset + dst_SxV_elem_offset;

                std::swap_ranges(src_SxVslice, src_SxVslice + component_count * btf->DataSize, dst_SxVslice);
            }
        }
    }

    if (out_extra)
    {
        *out_extra = extra;
    }

    return btf.release();
}
#endif
