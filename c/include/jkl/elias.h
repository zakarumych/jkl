#ifndef JKL_ELIAS_H
#define JKL_ELIAS_H

#include <stdint.h>

#include "bit_reader.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int jkl_elias_gamma_decode_(JklBitReader *reader, uint8_t bits, uint64_t *out_value);

    static inline int jkl_elias_gamma_decode_u32(JklBitReader *reader, uint32_t *out_value)
    {
        uint64_t v;
        JKL_RETURN_IF_ERROR(jkl_elias_gamma_decode_(reader, 32, &v));
        *out_value = (uint32_t)v;
        return JKL_OK;
    }

    static inline int jkl_elias_gamma_decode_u64(JklBitReader *reader, uint64_t *out_value)
    {
        return jkl_elias_gamma_decode_(reader, 64, out_value);
    }

    int jkl_elias_delta_decode_(JklBitReader *reader, uint8_t bits, uint64_t *out_value);

    static inline int jkl_elias_delta_decode_u32(JklBitReader *reader, uint32_t *out_value)
    {
        uint64_t v;
        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode_(reader, 32, &v));
        *out_value = (uint32_t)v;
        return JKL_OK;
    }

    static inline int jkl_elias_delta_decode_u64(JklBitReader *reader, uint64_t *out_value)
    {
        return jkl_elias_delta_decode_(reader, 64, out_value);
    }

    int jkl_elias_delta_decode_nonzero_(JklBitReader *reader, uint8_t bits, uint64_t *out_value);

    static inline int jkl_elias_delta_decode_nonzero_u32(JklBitReader *reader, uint32_t *out_value)
    {
        uint64_t v;
        JKL_RETURN_IF_ERROR(jkl_elias_delta_decode_nonzero_(reader, 32, &v));
        *out_value = (uint32_t)v;
        return JKL_OK;
    }

    static inline int jkl_elias_delta_decode_nonzero_u64(JklBitReader *reader, uint64_t *out_value)
    {
        return jkl_elias_delta_decode_nonzero_(reader, 64, out_value);
    }

#define jkl_elias_gamma_decode(reader, out_value) _Generic((*out_value), \
    uint32_t: jkl_elias_gamma_decode_u32,                                \
    uint64_t: jkl_elias_gamma_decode_u64)(reader, out_value)

#define jkl_elias_delta_decode(reader, out_value) _Generic((*out_value), \
    uint32_t: jkl_elias_delta_decode_u32,                                \
    uint64_t: jkl_elias_delta_decode_u64)(reader, out_value)

#define jkl_elias_delta_decode_nonzero(reader, out_value) _Generic((*out_value), \
    uint32_t: jkl_elias_delta_decode_nonzero_u32,                                \
    uint64_t: jkl_elias_delta_decode_nonzero_u64)(reader, out_value)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
