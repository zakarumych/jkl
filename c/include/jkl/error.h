#ifndef JKL_ERROR_H
#define JKL_ERROR_H

#define JKL_OK 0

#define JKL_ERR_IO -2
#define JKL_ERR_OOM -3
#define JKL_ERR_INVALID_MAGIC -4
#define JKL_ERR_INVALID_COMPRESSION -5
#define JKL_ERR_INVALID_FORMAT -6
#define JKL_ERR_MIP_ZERO -7
#define JKL_ERR_INVALID_DIMENSIONS -8
#define JKL_ERR_INVALID_EXTENT -9
#define JKL_ERR_INVALID_DATA -10
#define JKL_ERR_TOO_LARGE -11
#define JKL_ERR_UNSUPPORTED_FORMAT -12
#define JKL_ERR_EOF -13
#define JKL_ERR_DECODE_INCOMPLETE -14
#define JKL_ERR_OUT_OF_RANGE -15
#define JKL_ERR_NEED_TOKEN -16

/* Converts error code to boolean success value: success => true, error => false. */
#define JKL_SUCCEEDED(err_code) ((err_code) == JKL_OK)

/* Returns early when expression yields an error code. */
#define JKL_RETURN_IF_ERROR(expr)      \
    do                                 \
    {                                  \
        int _jkl_error_code = (expr);  \
        if (_jkl_error_code != JKL_OK) \
        {                              \
            return _jkl_error_code;    \
        }                              \
    } while (0)

#endif
