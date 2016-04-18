#ifndef XGBOOST_UTILS_H
#define XGBOOST_UTILS_H
/*!
 * \file xgboost_utils.h
 * \brief simple utils to support the code
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */

#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define fopen64 fopen
#else

// use 64 bit offset, either to include this header in the beginning, or 
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C"{    
#include <sys/types.h>
};
#include <cstdio>
#endif

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstdarg>

namespace xgboost{
/*! \brief namespace for helper utils of the project */
namespace utils{

/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;
/*!
 * \brief handling of Assert error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleAssertError(const char *msg) {
  fprintf(stderr, "AssertError:%s\n", msg);
  exit(-1);
}
/*!
 * \brief handling of Check error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleCheckError(const char *msg) {
  throw std::runtime_error(msg);
}

inline void Assert(bool exp) {
  if(!exp) {
    fprintf(stderr, "AssertError\n");
    exit(-1);
  }
}
/*! \brief assert an condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleAssertError(msg.c_str());
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  HandleCheckError(msg.c_str());
}

/*!\brief same as assert, but this is intended to be used as message for user*/
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

inline void Warning(const char *msg) {
  fprintf(stderr, "warning:%s\n",msg);
}

/*! \brief replace fopen, report error when the file open fails */
inline FILE *FopenCheck(const char *fname, const char *flag) {
  FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}      
 
}  // namespace utils
}  // namespace xgboost

#endif
