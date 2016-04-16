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
struct IFMatrix {
 public:
  /*! \brief one entry in a row */
  struct REntry{
    /*! \brief feature index */
    bst_uint  findex;
    /*! \brief feature value */
    bst_float fvalue;
    /*! \brief constructor */
    REntry( void ){}
    /*! \brief constructor */
    REntry( bst_uint findex, bst_float fvalue ) : findex(findex), fvalue(fvalue){}
    inline static bool cmp_fvalue( const REntry &a, const REntry &b ){
        return a.fvalue < b.fvalue;
    }
  };
  /*! \brief row iterator */
  struct RowIter{
    const REntry *dptr_, *end_;
    RowIter( const REntry* dptr, const REntry* end )
        :dptr_(dptr),end_(end){}
    inline bool Next( void ){
        if( dptr_ == end_ ) return false;
        else{
           ++ dptr_; return true;
        }
    }
    inline bst_uint  findex( void ) const{
        return dptr_->findex;
    }
    inline bst_float fvalue( void ) const{
        return dptr_->fvalue;
    }
  };
  /*! \brief column iterator */
  struct ColIter: public RowIter{
    ColIter( const REntry* dptr, const REntry* end )
        :RowIter( dptr, end ){}
    inline bst_uint  rindex( void ) const{
        return this->findex();
    }
  };
 public:
  // the following are column meta data, should be able to answer them fast
  /*! \return whether column access is enabled */
  virtual bool HaveColAccess(void) const = 0;
  /*!  \brief get row iterator*/
  virtual RowIter GetRow(size_t ridx) const = 0;
  /*!
   * \brief get column iterator, the columns must be sorted by feature value
   * \param ridx column index
   * \return column iterator
   */
  virtual ColIter GetSortedCol(size_t ridx) const = 0;
  /*! \return number of columns in the FMatrix */
  virtual size_t NumCol(void) const = 0;
  // virtual destructor
  virtual ~IFMatrix(void) {}
};
}  // namespace xgboost
#endif
