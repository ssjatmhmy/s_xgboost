#ifndef XGBOOST_DATA_H
#define XGBOOST_DATA_H

/*!
 * \file data.h
 * \brief the input data structure for gradient boosting
 */

#include <vector>
#include <cstdio>

namespace xgboost{
/*! \brief interger type used in boost */
typedef int bst_int;
/*! \brief unsigned interger type used in boost */
typedef unsigned bst_uint;
/*! \brief float type used in boost */
typedef float bst_float;
/*! \brief debug option for booster */    
const bool bst_debug = false;    

/**
 * \brief interface of feature matrix, needed for tree construction
 *  this interface defines two way to access features,
 *  row access is defined by iterator of RowBatch
 *  col access is optional, checked by HaveColAccess, and defined by iterator of ColBatch
 */
class IFMatrix {
 public:
  // the following are column meta data, should be able to answer them fast
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const = 0;
  /*! \return number of columns in the FMatrix */
  virtual size_t NumCol(void) const = 0;
  // virtual destructor
  virtual ~IFMatrix(void){}
};
}  // namespace xgboost
#endif
