#ifndef XGBOOST_RANDOM_H
#define XGBOOST_RANDOM_H
/*!
 * \file random.h
 * \brief PRNG to support random number generation
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 *
 * Use standard PRNG from stdlib
 */
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int  uint32_t;
#else
#include <inttypes.h>
#endif

/*! namespace of PRNG */
namespace xgboost{
namespace random{
/*! \brief seed the PRNG */
inline void Seed(uint32_t seed) {
  srand(seed);
}

}  // namespace random
}  // namespace xgboost

#endif
