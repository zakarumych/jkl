#include "elias.h"

#include <limits.h>

int jkl_elias_gamma_decode_(JklBitReader *reader, uint32_t bits, uint64_t *out_value)
{
    uint32_t msb = 0;
    int bit = 0;
    uint64_t tail = 0;

    for (;;)
    {
        JKL_RETURN_IF_ERROR(jkl_bit_read_bit(reader, &bit));

        if (bit != 0)
        {
            break;
        }

        msb += 1;
        if (msb >= bits)
        {
            JKL_RETURN_IF_ERROR(jkl_bit_discard_bits(reader, msb));
            return JKL_ERR_TOO_LARGE;
        }
    }

    JKL_RETURN_IF_ERROR(jkl_bit_read_bits(reader, msb, &tail));

    *out_value = (1ULL << msb) + tail;
    return JKL_OK;
}

int jkl_elias_delta_decode_(JklBitReader *reader, uint32_t bits, uint64_t *out_value)
{
    uint64_t msb_plus_1 = 0;
    uint64_t tail = 0;
    uint8_t msb;

    JKL_RETURN_IF_ERROR(jkl_elias_gamma_decode_(reader, 8, &msb_plus_1));

    msb = msb_plus_1 - 1;

    if (msb > bits)
    {
        JKL_RETURN_IF_ERROR(jkl_bit_discard_bits(reader, msb));
        return JKL_ERR_TOO_LARGE;
    }

    JKL_RETURN_IF_ERROR(jkl_bit_read_bits(reader, (uint32_t)msb, &tail));

    if (msb == bits)
    {
        if (tail != 0)
        {
            return JKL_ERR_TOO_LARGE;
        }
        *out_value = UINT64_MAX;
        return JKL_OK;
    }

    *out_value = ((1ULL << msb) + tail) - 1ULL;
    return JKL_OK;
}

int jkl_elias_delta_decode_nonzero_(JklBitReader *reader, uint32_t bits, uint64_t *out_value)
{
    uint64_t msb_plus_1 = 0;
    uint64_t tail = 0;
    uint8_t msb;

    JKL_RETURN_IF_ERROR(jkl_elias_gamma_decode_(reader, 8, &msb_plus_1));

    msb = msb_plus_1 - 1;

    if (msb >= bits)
    {
        return JKL_ERR_TOO_LARGE;
    }

    JKL_RETURN_IF_ERROR(jkl_bit_read_bits(reader, msb, &tail));

    *out_value = (1ULL << msb) + tail;
    return JKL_OK;
}
