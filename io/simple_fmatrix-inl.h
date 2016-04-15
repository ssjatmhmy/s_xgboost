#ifndef XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP_
#define XGBOOST_IO_SIMPLE_FMATRIX_INL_HPP_

/*!
 * \file data.h
 * \brief the input data structure for gradient boosting
 */

#include <vector>
#include <climits>
#include "../data.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/matrix_csr.h"

namespace xgboost{
/*! 
 * \brief feature matrix to store training instance, in sparse CSR format
 */        
class FMatrixS: public IFMatrix {
 public:
  /*! \brief one entry in a row */
  struct REntry{
    /*! \brief feature index */
    bst_uint findex;
    /*! \brief feature value */
    bst_float fvalue;
    /*! \brief constructor */
    REntry(void) {}
    /*! \brief constructor */
    REntry(bst_uint findex, bst_float fvalue) : findex(findex), fvalue(fvalue) {}
    inline static bool cmp_fvalue(const REntry &a, const REntry &b) {
      return a.fvalue < b.fvalue;
    }
  };
  /*! \brief row iterator */
  struct RowIter {
    const REntry *dptr_, *end_;
    RowIter(const REntry* dptr, const REntry* end)
        :dptr_(dptr), end_(end) {}
    inline bool Next(void) {
      if (dptr_ == end_) return false;
      else {
        ++ dptr_; return true;
      }
    }
    inline bst_uint findex(void) const {
      return dptr_->findex;
    }
    inline bst_float fvalue(void) const {
      return dptr_->fvalue;
    }
  };
  /*!  \brief get number of rows */
  inline size_t NumRow(void) const {
    return row_ptr_.size() - 1;
  }
  /*! 
   * \brief get number of nonzero entries
   * \return number of nonzero entries
   */
  inline size_t NumEntry(void) const {
    return row_data_.size();
  }
 
  inline size_t AddRow(const std::vector<bst_uint> &findex, 
                       const std::vector<bst_float> &fvalue,
                       unsigned fstart = 0, 
                       unsigned fend = UINT_MAX) {
    unsigned cnt = 0;
    for (size_t i = 0; i < findex.size(); ++i) {
      if (findex[i] < fstart || findex[i] >= fend) continue;
      row_data_.push_back(REntry(findex[i], fvalue[i]));
      cnt ++;
    }
    row_ptr_.push_back(row_ptr_.back() + cnt);
    return row_ptr_.size() - 2;
  }
  /*!  \brief get row iterator*/
  inline RowIter GetRow(size_t ridx) const {
    utils::Assert(!bst_debug || ridx < this->NumRow(), "row id exceed bound");
    return RowIter(&row_data_[row_ptr_[ridx]]-1, &row_data_[row_ptr_[ridx+1]]-1);
  }
 public:
  /*!  \brief get number of colmuns */
  inline size_t NumCol(void) const {
    utils::Assert(this->HaveColAccess(), 
                  " Can not get number of column, do not have access.");
    return col_ptr_.size() - 1;
  }
  /*! \brief clear the storage */
  inline void Clear(void) {
    row_ptr_.clear();
    row_ptr_.push_back(0);
    row_data_.clear();
    col_ptr_.clear();
    col_data_.clear();
  }
  inline void InitData(void) {
    utils::SparseCSRMBuilder<REntry> builder(col_ptr_, col_data_);
    builder.InitBudget(0);
    for (size_t i = 0; i < this->NumRow(); ++i) {
      for (RowIter it = this->GetRow(i); it.Next(); ) {
        builder.AddBudget(it.findex());
      }
    }
    builder.InitStorage();
    for(size_t i = 0; i < this->NumRow(); ++i) {
      for (RowIter it = this->GetRow(i); it.Next(); ) {
        builder.PushElem(it.findex(), REntry((bst_uint)i, it.fvalue()));
      }
    }
    // sort columns
    unsigned ncol = static_cast<unsigned>(this->NumCol());
    for (unsigned i = 0; i < ncol; ++i) {
      std::sort(&col_data_[col_ptr_[i]], &col_data_[col_ptr_[i+1]], REntry::cmp_fvalue);
    }
  }
  /*! \return whether column access is enabled */
  inline bool HaveColAccess(void) const {
    return col_ptr_.size() != 0 && col_data_.size() == row_data_.size();
  }
  /*!
  * \brief save data to binary stream 
  *        note: since we have size_t in ptr, 
  *              the function is not consistent between 64bit and 32bit machine
  * \param fo output stream
  */
  inline void SaveBinary(utils::IStream &fo) const {
    FMatrixS::SaveBinary(fo, row_ptr_, row_data_);
    int col_access = this->HaveColAccess() ? 1 : 0;
    fo.Write(&col_access, sizeof(int));
    if (col_access != 0) {
      FMatrixS::SaveBinary(fo, col_ptr_, col_data_);
    }
  }
  /*!
  * \brief load data from binary stream 
  *        note: since we have size_t in ptr, 
  *              the function is not consistent between 64bit and 32bit machin
  * \param fi input stream
  */
  inline void LoadBinary(utils::IStream &fi) {
    FMatrixS::LoadBinary(fi, row_ptr_, row_data_);
    int col_access;                
    fi.Read(&col_access, sizeof(int));
    if (col_access != 0) {
      FMatrixS::LoadBinary(fi, col_ptr_, col_data_);
    }
  }
 private:
  /*!
  * \brief save data to binary stream 
  * \param fo output stream
  * \param ptr pointer data
  * \param data data content
  */
  inline static void SaveBinary(utils::IStream &fo, 
                                const std::vector<size_t> &ptr, 
                                const std::vector<REntry> &data) {
    size_t nrow = ptr.size() - 1;
    fo.Write(&nrow, sizeof(size_t));
    fo.Write(&ptr[0], ptr.size()*sizeof(size_t));
    if (data.size() != 0) {
      fo.Write(&data[0], data.size()*sizeof(REntry));
    }
  }
  /*!
  * \brief load data from binary stream 
  * \param fi input stream
  * \param ptr pointer data
  * \param data data content
  */
  inline static void LoadBinary(utils::IStream &fi,
                                std::vector<size_t> &ptr, 
                                std::vector<REntry> &data) {
    size_t nrow;
    utils::Assert(fi.Read(&nrow, sizeof(size_t)) != 0, "Load FMatrixS");
    ptr.resize(nrow + 1);
    utils::Assert(fi.Read(&ptr[0], ptr.size()*sizeof(size_t)), "Load FMatrixS");

    data.resize(ptr.back()); 
    if (data.size() != 0) {
      utils::Assert(fi.Read(&data[0], data.size()*sizeof(REntry)), "Load FMatrixS");
    }
  }  
 protected:
  /*! \brief row pointer of CSR sparse storage */
  std::vector<size_t> row_ptr_;
  /*! \brief data in the row */
  std::vector<REntry> row_data_;
  /*! \brief column pointer of CSC format */
  std::vector<size_t> col_ptr_;
  /*! \brief column datas */
  std::vector<REntry> col_data_;
};

}  // namespace xgboost
#endif
