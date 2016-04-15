/*!
 * \file matrix_csr.h
 * \brief this file defines some easy to use STL based class for in memory sparse CSR matrix
 * \author Tianqi Chen: tianqi.tchen@gmail.com
*/
#ifndef XGBOOST_MATRIX_CSR_H
#define XGBOOST_MATRIX_CSR_H
#include <vector>
#include <algorithm>
#include "utils.h"

namespace xgboost{
namespace utils{
/*! 
* \brief a class used to help construct CSR format matrix, 
*        can be used to convert row major CSR to column major CSR
* \tparam IndexType type of index used to store the index position, usually unsigned or size_t
* \tparam whether enabling the usage of aclist, this option must be enabled manually
*/
template<typename IndexType, bool UseAcList = false>
struct SparseCSRMBuilder {
 private:
  /*! \brief dummy variable used in the indicator matrix construction */
  std::vector<size_t> dummy_aclist;
  /*! \brief pointer to each of the row */
  std::vector<size_t> &cptr;
  /*! \brief index of nonzero entries in each row */
  std::vector<IndexType> &findex;
  /*! \brief a list of active rows, used when many rows are empty */
  std::vector<size_t> &aclist;
 public:
  SparseCSRMBuilder(std::vector<size_t> &p_cptr,
                    std::vector<IndexType> &p_findex)
      :cptr(p_cptr), findex(p_findex), aclist(dummy_aclist) {
    Assert(!UseAcList, "enabling bug");
  }  
  /*! \brief use with caution! cptr must be cleaned before use */     
  SparseCSRMBuilder(std::vector<size_t> &p_cptr,
                    std::vector<IndexType> &p_findex,
                    std::vector<size_t> &p_aclist)
      :cptr(p_cptr), findex(p_findex), aclist(p_aclist) {
    Assert(UseAcList, "must manually enable the option use aclist");
  }
 public:
  /*! 
  * \brief step 1: initialize the number of cols in the data, not necessary exact
  */
  inline void InitBudget(size_t ncols = 0) {
    cptr.clear();
    cptr.resize(ncols + 1, 0);
  }
  /*! 
  * \brief step 2: add budget to each cols, this function is called when aclist is used
  * \param col_id the id of the col
  * \param nelem  number of element budget add to this col
  */
  inline void AddBudget(size_t col_id, size_t nelem = 1) {
    // why the size of cptr is set to col_id+2: col_id starts from 0, so we have col_id+1 columns.
    // Plus, we need to add a 0 at the beginning of cptr.
    if (cptr.size() < col_id + 2) {
      cptr.resize(col_id + 2, 0);
    }
    cptr[col_id+1] += nelem;
  }
  /*! \brief step 3: initialize the necessary storage */
  inline void InitStorage(void) {
    // initialize cptr to be beginning of each segment
    size_t start = 0;
    for (size_t i = 1; i < cptr.size(); ++i) {
      size_t rlen = cptr[i];
      cptr[i] = start;
      start += rlen;
    }
    findex.resize(start);
  }
  /*! 
  * \brief step 4: 
  * used in indicator matrix construction, add new 
  * element to each col, the number of calls shall be exactly same as add_budget 
  */
  inline void PushElem(size_t col_id, IndexType entry) { 
    size_t &rp = cptr[col_id+1];
    findex[rp++] = entry;
  }
};
}  // namespace utils
}  // namespace xgboost
#endif
