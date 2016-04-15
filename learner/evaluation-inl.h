#ifndef XGBOOST_REG_EVAL_INL_H
#define XGBOOST_REG_EVAL_INL_H
/*!
* \file xgboost_reg_eval.h
* \brief evaluation metrics for regression and classification
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include "../utils/utils.h"
#include "../utils/omp.h"

namespace xgboost {
namespace learner {
/*! \brief RMSE */
struct EvalRMSE : public IEvaluator {            
  virtual float Eval(const std::vector<float> &preds, 
                     const std::vector<float> &labels) const {
    const unsigned ndata = static_cast<unsigned>(preds.size());
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (unsigned i = 0; i < ndata; ++i) {
      float diff = preds[i] - labels[i];
      sum += diff * diff;
    } 
    return sqrtf(sum/ndata);
  }
  virtual const char *Name(void) const {
    return "rmse";
  }
};

/*! \brief Error */
struct EvalError : public IEvaluator {            
  virtual float Eval(const std::vector<float> &preds, 
                     const std::vector<float> &labels) const {
    const unsigned ndata = static_cast<unsigned>(preds.size());
    unsigned nerr = 0;
    #pragma omp parallel for reduction(+:nerr) schedule(static)
    for (unsigned i = 0; i < ndata; ++i) {
      if (preds[i] > 0.5f) {
        if (labels[i] < 0.5f) nerr += 1;
      } else {
        if (labels[i] > 0.5f) nerr += 1;
      }
    } 
    return static_cast<float>(nerr) / ndata;
  }
  virtual const char *Name(void) const {
    return "error";
  }
};

/*! \brief Error */
struct EvalLogLoss : public IEvaluator {            
  virtual float Eval(const std::vector<float> &preds, 
                     const std::vector<float> &labels) const {
    const unsigned ndata = static_cast<unsigned>(preds.size());
    unsigned nerr = 0;
    #pragma omp parallel for reduction(+:nerr) schedule(static)
    for (unsigned i = 0; i < ndata; ++ i ) {
        const float y = labels[i];
        const float py = preds[i];
        nerr -= y * std::log(py) + (1.0f-y)*std::log(1-py);
    } 
    return static_cast<float>(nerr) / ndata;
  }
  virtual const char *Name(void) const {
    return "negllik";
  }
};
}  // namespace learner
}  // namespace xgboost
#endif
