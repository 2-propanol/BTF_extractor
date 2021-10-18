ctypedef unsigned int uint32_t

cdef extern from "btf.hh" nogil:
    struct Vector3:
        float x, y, z

    ctypedef Vector3 Spectrum

    struct BTF:
        uint32_t ChannelCount
        uint32_t Width, Height

        Vector3 *Lights, *Views
        uint32_t LightCount, ViewCount

    struct BTFExtra:
        pass

    cdef Spectrum BTFFetchSpectrum(
        const BTF *btf,
        uint32_t light_vert, uint32_t view_vert,
        uint32_t x, uint32_t y
    )
    cdef BTF *LoadBTF(const char *file_path, BTFExtra *extra)
    cdef void DestroyBTF(BTF *btf)
