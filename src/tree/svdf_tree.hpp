#ifndef XGBOOST_APEX_TREE_HPP
#define XGBOOST_APEX_TREE_HPP
/*!
 * \file xgboost_svdf_tree.hpp
 * \brief implementation of regression tree constructor, with layerwise support
 *        this file is adapted from GBRT implementation in SVDFeature project
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn, tianqi.tchen@gmail.com 
 */
#include <algorithm>
#include "tree_model.h"
#include "../utils/random.h"
#include "../utils/matrix_csr.h"

namespace xgboost {
namespace gbm {                
// updater of rtree, allows the parameters to be stored inside, key solver
class RTreeUpdater {
 private:
  // training parameter
  const TreeParamTrain &param;
  // parameters, reference
  RegTree &tree;
  std::vector<float> &grad;
  std::vector<float> &hess;
  const IFMatrix &smat;
  const std::vector<unsigned> &group_id;
 public:
  RTreeUpdater(const TreeParamTrain &pparam, 
               RegTree &ptree,
               std::vector<float> &pgrad,
               std::vector<float> &phess,
               const IFMatrix &psmat, 
               const std::vector<unsigned> &pgroup_id):
      param(pparam), tree(ptree), grad(pgrad), hess(phess),
      smat(psmat), group_id(pgroup_id) {
  }
};
}  // namespace gbm
}  // namespace xgboost
#endif


