#ifndef XGBOOST_TREE_HPP
#define XGBOOST_TREE_HPP
/*!
 * \file xgboost_tree.hpp
 * \brief implementation of regression tree
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
#include "tree_model.h"

namespace xgboost {
namespace gbm {
  const bool rt_debug = false;
  // whether to check bugs
  const bool check_bug = false;

  const float rt_eps = 1e-5f;
  const float rt_2eps = rt_eps * 2.0f;
  
  inline double sqr(double a) {
    return a * a;
  }
};
};
#include "../utils/fmap.h"
#include "svdf_tree.hpp"
//#include "xgboost_col_treemaker.hpp"
//#include "xgboost_row_treemaker.hpp"

namespace xgboost {
namespace gbm {
// regression tree, construction algorithm is seperated from this class
// see RegTreeUpdater
class RegTreeTrainer : public IGradBooster {
 public:
  RegTreeTrainer(void) { 
    silent = 0; tree_maker = 1; 
    // normally we won't have more than 64 OpenMP threads
    threadtemp.resize(64, ThreadEntry());
  }
  virtual ~RegTreeTrainer(void) {}
 public:
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "silent")) silent = atoi(val);
    if (!strcmp(name, "tree_maker")) tree_maker = atoi(val);
    param.SetParam(name, val);
    constrain.SetParam(name, val);
    tree.param.SetParam(name, val);
  }
  virtual void LoadModel(utils::IStream &fi) {
    tree.LoadModel(fi );
  }
  virtual void SaveModel(utils::IStream &fo) const {
    tree.SaveModel(fo);
  }
  virtual void InitModel(void) {
    tree.InitModel();
  }
 public:
  virtual void DoBoost(std::vector<float> &grad, 
                       std::vector<float> &hess,
                       const IFMatrix &smat,
                       const std::vector<unsigned> &root_index) {
    utils::Assert(grad.size() < UINT_MAX, "number of instance exceed what we can handle");
    if (!silent) {
      printf("\nbuild GBRT with %u instances\n", (unsigned)grad.size());
    }
    int num_pruned;
    switch (tree_maker) {
      case 0: {
        utils::Assert(!constrain.HasConstrain(), "tree maker 0 does not support constrain");
        RTreeUpdater updater(param, tree, grad, hess, smat, root_index);
        //tree.param.max_depth = updater.do_boost( num_pruned );
        break;
      }
    }
  }            
  virtual float Predict(const IFMatrix &fmat, bst_uint ridx, unsigned gid = 0) {     
    return 0.0f;  
  }
  virtual float Predict(const std::vector<float> &feat, 
                        const std::vector<bool> &funknown,
                        unsigned gid = 0) {
    return 0.0f; 
  }            

 private:
  // silent 
  int silent;
  RegTree tree;
  TreeParamTrain param;
 private:
  // tree maker
  int tree_maker;
  // feature constrain
  utils::FeatConstrain constrain;  
 private:
  struct ThreadEntry {
    std::vector<float> feat;
    std::vector<bool> funknown;
  };
  std::vector<ThreadEntry> threadtemp;
};
}  // namespace gbm
}  // namespace xgboost
#endif
