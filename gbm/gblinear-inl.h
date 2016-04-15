#ifndef XGBOOST_LINEAR_HPP
#define XGBOOST_LINEAR_HPP
/*!
 * \file xgboost_linear.h
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is coordinate descent, require column major format
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <vector>
#include <algorithm>

#include "./gbm.h"
#include "../utils/utils.h"

namespace xgboost {
namespace gbm {
/*! \brief linear model, with L1/L2 regularization */
class LinearBooster : public IGradBooster {
 public:
  LinearBooster(void) {silent = 0;}
  virtual ~LinearBooster(void) {}
 public:
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "silent")) silent = atoi(val);
    if (model.weight.size() == 0) model.param.SetParam(name, val);
    param.SetParam(name, val);
  }
  virtual void LoadModel(utils::IStream &fi) {
    model.LoadModel(fi);
  }
  virtual void SaveModel(utils::IStream &fo) const {
    model.SaveModel(fo);
  }
  virtual void InitModel(void) {
    model.InitModel();
  }
 public:
  virtual void DoBoost(std::vector<float> &grad, 
                       std::vector<float> &hess,
                       const IFMatrix &fmat,
                       const std::vector<unsigned> &root_index) {
    printf("gblinear DoBoost\n");
  }
  inline float Predict(const IFMatrix &fmat, bst_uint ridx, unsigned root_index) {
    float sum = model.bias();
    //for (IFMatrix::RowIter it = fmat.GetRow(ridx); it.Next(); ) { 
    //  sum += model.weight[it.findex()] * it.fvalue();
    //}
    return sum;
  }
 
 protected:
  // training parameter
  struct ParamTrain {
    /*! \brief learning_rate */
    float learning_rate;
    /*! \brief regularization weight for L2 norm */
    float reg_lambda;
    /*! \brief regularization weight for L1 norm */
    float reg_alpha;
     /*! \brief regularization weight for L2 norm  in bias */               
    float reg_lambda_bias;
    
    ParamTrain(void) {
      reg_alpha = 0.0f; reg_lambda = 0.0f; reg_lambda_bias = 0.0f;
      learning_rate = 1.0f;
    }            
    inline void SetParam(const char *name, const char *val) {
      // sync-names
      if (!strcmp("eta", name)) learning_rate = (float)atof(val);
      if (!strcmp("lambda", name)) reg_lambda = (float)atof(val);
      if (!strcmp("alpha", name)) reg_alpha = (float)atof(val);
      if (!strcmp("lambda_bias", name)) reg_lambda_bias = (float)atof(val);
      // real names
      if (!strcmp("learning_rate", name)) learning_rate = (float)atof(val);     
      if (!strcmp("reg_lambda", name)) reg_lambda = (float)atof(val);
      if (!strcmp("reg_alpha", name)) reg_alpha = (float)atof(val);
      if (!strcmp("reg_lambda_bias", name)) reg_lambda_bias = (float)atof(val);
    }
  };  
  // model for linear booster
  class Model {
   public:
    // model parameter
    struct Param {
      // number of feature dimension
      int num_feature;
      // constructor
      Param(void) {
        num_feature = 0;
      }
      inline void SetParam(const char *name, const char *val) {
        if(!strcmp(name, "num_feature")) num_feature = atoi(val);
      }
    };
   public:
    Param param;
    // weight for each of feature, bias is the last one
    std::vector<float> weight;
   public:
    // initialize the model parameter
    inline void InitModel(void) {
      // bias is the last weight
      weight.resize(param.num_feature + 1);
      std::fill(weight.begin(), weight.end(), 0.0f);
    }
    // save the model to file 
    inline void SaveModel(utils::IStream &fo) const {
      fo.Write(&param, sizeof(Param));
      fo.Write(&weight[0], sizeof(float)*weight.size());
    }
    // load model from file
    inline void LoadModel(utils::IStream &fi) {
      utils::Assert(fi.Read(&param, sizeof(Param)) != 0, "Load LinearBooster");
      weight.resize(param.num_feature + 1);
      utils::Assert(fi.Read(&weight[0], sizeof(float)*weight.size()) != 0, "Load LinearBooster");
    }
    // model bias
    inline float &bias(void) {
      return weight.back();
    }
  };
 protected:
  Model model;
  ParamTrain param;
 private:
  int silent;
};
}  // namespace gbm
}  // namespace xgboost
#endif
