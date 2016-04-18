#ifndef XGBOOST_REG_EVAL_H
#define XGBOOST_REG_EVAL_H
/*!
* \file xgboost_reg_eval.h
* \brief evaluation metrics for regression and classification
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include "../utils/utils.h"

namespace xgboost {
namespace learner {
/*! \brief evaluator that evaluates the loss metrics */
struct IEvaluator {
  /*! 
   * \brief evaluate a specific metric 
   * \param preds prediction
   * \param labels label
   */
  virtual float Eval(const std::vector<float> &preds, 
                     const std::vector<float> &labels) const = 0;
  /*! \return name of metric */
  virtual const char *Name(void) const = 0;
};
}  // namespace learner
}  // namespace xgboost

// include implementations of evaluation functions
#include "evaluation-inl.h"
// factory function
namespace xgboost {
namespace learner {
/*! \brief a set of evaluators */
struct EvalSet {
 public:
  inline void AddEval(const char *name) {                
    if (!strcmp(name, "rmse")) evals_.push_back(&rmse_);
    if (!strcmp(name, "error")) evals_.push_back(&error_);
    if (!strcmp(name, "logloss")) evals_.push_back(&logloss_);
  }
  inline void Init(void) {
    std::sort(evals_.begin(), evals_.end());
    evals_.resize(std::unique(evals_.begin(), evals_.end()) - evals_.begin());
  }
  inline void Eval(FILE *fo, const char *evname,
                   const std::vector<float> &preds, 
                   const std::vector<float> &labels) const {
    for (size_t i = 0; i < evals_.size(); ++i) {
      float res = evals_[i]->Eval(preds, labels);
      fprintf(fo, "\t%s-%s:%f", evname, evals_[i]->Name(), res); 
    } 
  }
 private:
  EvalRMSE rmse_;
  EvalError error_;
  EvalLogLoss logloss_;
  std::vector<const IEvaluator*> evals_;  
};
}  // namespace learner
}  // namespace xgboost
#endif
