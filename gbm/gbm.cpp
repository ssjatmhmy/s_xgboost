/*!
 * \file xgboost-inl.hpp
 * \brief bootser implementations 
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
// implementation of boosters go to here 

// A good design should have minimum functions defined interface, user should only operate on interface
// I break it a bit, by using template and let user 'see' the implementation
// The user should pretend that they only can use the interface, and we are all cool
// I find this is the only way so far I can think of to make boosters invariant of data structure, 
// while keep everything fast
#include "gbm.h"
#include "../utils/utils.h"
//#include "tree/xgboost_tree.hpp"
#include "./gblinear-inl.h"

namespace xgboost {
namespace gbm {
/*!
 * \brief create a gradient booster, given type of booster
 * \param booster_type type of gradient booster, can be used to specify implements
 * \tparam FMatrix input data type for booster
 * \return the pointer to the gradient booster created
 */
//template<typename FMatrix>
inline IGradBooster *CreateBooster(int booster_type) {
  switch (booster_type) {
    //case 0: return new RegTreeTrainer<FMatrix>();
    case 0: return new LinearBooster();
    default: utils::Error("unknown booster_type"); return NULL;
  }
}
}  // namespace gbm
}  // namespace xgboost

