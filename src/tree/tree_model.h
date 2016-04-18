#ifndef XGBOOST_TREE_MODEL_H
#define XGBOOST_TREE_MODEL_H
/*!
 * \file xgboost_tree_model.h
 * \brief generic definition of model structure used in tree models
 *        used to support learning of boosting tree
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cstring>
#include "../utils/utils.h"
#include "../utils/io.h"

namespace xgboost {
namespace gbm {
/*!
 * \brief template class of TreeModel 
 * \tparam TSplitCond data type to indicate split condition
 * \tparam TNodeStat auxiliary statistics of node to help tree building
 */
template<typename TSplitCond, typename TNodeStat>
class TreeModel {
 public:
  /*! \brief data type to indicate split condition */
  typedef TNodeStat  NodeStat;
  /*! \brief auxiliary statistics of node to help tree building */
  typedef TSplitCond SplitCond;
 public:
  /*! \brief parameters of the tree */
  struct Param {
    /*! \brief number of start root */
    int num_roots;
    /*! \brief total number of nodes */
    int num_nodes;
    /*!\brief number of deleted nodes */
    int num_deleted;
    /*! \brief maximum depth, this is a statistics of the tree */
    int max_depth;
    /*! \brief  number of features used for tree construction */
    int num_feature;
    /*! \brief constructor */
    Param(void) {
      max_depth = 0;
    }
    /*! 
     * \brief set parameters from outside 
     * \param name name of the parameter
     * \param val  value of the parameter
     */
    inline void SetParam(const char *name, const char *val) {
      if (!strcmp("num_roots", name)) num_roots = atoi(val);
      if (!strcmp("num_feature", name)) num_feature = atoi(val);
    }
  };
  /*! \brief tree node */
  class Node {
   private:
    friend class TreeModel<TSplitCond, TNodeStat>;
    /*! 
     * \brief in leaf node, we have weights, in non-leaf nodes, 
     *        we have split condition 
     */
    union Info {
      float leaf_value;
      TSplitCond split_cond;
    };
   private:
    // pointer to parent, highest bit is used to indicate whether it's a left child or not 
    int parent_;
    // pointer to left, right
    int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    unsigned sindex_;            
    // extra info
    Info info_;
   private:
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
   public:
    /*! \brief index of left child */
    inline int cleft(void) const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright(void) const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault(void) const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index(void) const {
      return sindex_ & ((1U<<31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left(void) const {
      return (sindex_ >> 31) != 0;
    } 
    /*! \brief whether current node is leaf node */
    inline bool is_leaf(void) const {
      return cleft_ == -1;
    }
    /*! \brief get leaf value of leaf node */
    inline float leaf_value(void) const {
      return (this->info_).leaf_value;
    }
    /*! \brief get split condition of the node */
    inline TSplitCond split_cond(void) const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    inline int parent(void) const {
      return parent_ & ((1U << 31) - 1);
    } 
    /*! \brief whether current node is left child */
    inline bool is_left_child(void) const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether current node is root */
    inline bool is_root(void) const {
      return parent_ == -1;
    }
    /*! 
     * \brief set the right child 
     * \param nide node id to right child
     */
    inline void set_right_child(int nid) {
      this->cright_ = nid;
    }
    /*! 
     * \brief set split condition of current node 
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    inline void set_split(unsigned split_index, TSplitCond split_cond, bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
    /*! 
     * \brief set the leaf value of the node
     * \param value leaf value
     * \param right right index, could be used to store 
     *        additional information
     */
    inline void set_leaf(float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }
  };
 protected:
  // vector of nodes
  std::vector<Node> nodes;
  // stats of nodes
  std::vector<TNodeStat> stats;
 protected:
  // free node space, used during training process
  std::vector<int> deleted_nodes;
 public:
  /*! \brief model parameter */
  Param param;
 public:
  /*! \brief constructor */
  TreeModel(void) {
    param.num_nodes = 1;
    param.num_roots = 1;
    param.num_deleted = 0;
    nodes.resize(1);
  }
  /*! \brief initialize the model */
  inline void InitModel(void) {
    param.num_nodes = param.num_roots;
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    for (int i = 0; i < param.num_nodes; ++i) {
      nodes[i].set_leaf(0.0f);
      nodes[i].set_parent(-1);
    }
  }
  /*! 
   * \brief save model to stream
   * \param fo output stream
   */
  inline void SaveModel(utils::IStream &fo) const {
    utils::Assert( param.num_nodes == (int)nodes.size() );
    utils::Assert( param.num_nodes == (int)stats.size() );
    fo.Write( &param, sizeof(Param) );
    fo.Write( &nodes[0], sizeof(Node) * nodes.size() );
    fo.Write( &stats[0], sizeof(NodeStat) * nodes.size() );
  }
  /*! 
   * \brief load model from stream
   * \param fi input stream
   */
  inline void LoadModel(utils::IStream &fi) {
    utils::Assert( fi.Read( &param, sizeof(Param) ) > 0, "TreeModel" );
    nodes.resize( param.num_nodes ); stats.resize( param.num_nodes );
    utils::Assert( fi.Read( &nodes[0], sizeof(Node) * nodes.size() ) > 0, "TreeModel::Node" );
    utils::Assert( fi.Read( &stats[0], sizeof(NodeStat) * stats.size() ) > 0, "TreeModel::Node" );

    deleted_nodes.resize( 0 );
    for (int i = param.num_roots; i < param.num_nodes; ++i) {
        if( nodes[i].is_root() ) deleted_nodes.push_back( i );
    }
    utils::Assert( (int)deleted_nodes.size() == param.num_deleted, "number of deleted nodes do not match" );
  }
};



/*! \brief training parameters for regression tree */
struct TreeParamTrain {
  // learning step size for a time
  float learning_rate;
  // minimum loss change required for a split
  float min_split_loss;
  // maximum depth of a tree
  int max_depth;
  //----- the rest parameters are less important ----
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // weight decay parameter used to control leaf fitting
  float reg_lambda;
  // reg method
  int reg_method;
  // default direction choice
  int default_direction;
  // whether we want to do subsample
  float subsample;
  // whether to use layerwise aware regularization
  int use_layerwise;
  // number of threads to be used for tree construction, if OpenMP is enabled, if equals 0, use system default
  int nthread;
  /*! \brief constructor */
  TreeParamTrain(void) {
    learning_rate = 0.3f;
    min_child_weight = 1.0f;
    max_depth = 6;
    reg_lambda = 1.0f;
    reg_method = 2;
    default_direction = 0;
    subsample = 1.0f;
    use_layerwise = 0;
    nthread = 0;
  }
  /*! 
  * \brief set parameters from outside 
  * \param name name of the parameter
  * \param val  value of the parameter
  */            
  inline void SetParam(const char *name, const char *val) {
    // sync-names 
    if( !strcmp( name, "gamma") )  min_split_loss = (float)atof( val );
    if( !strcmp( name, "eta") )    learning_rate  = (float)atof( val );
    if( !strcmp( name, "lambda") ) reg_lambda = (float)atof( val );
    // normal tree prameters
    if( !strcmp( name, "learning_rate") )     learning_rate = (float)atof( val );
    if( !strcmp( name, "min_child_weight") )  min_child_weight = (float)atof( val );
    if( !strcmp( name, "min_split_loss") )    min_split_loss = (float)atof( val );
    if( !strcmp( name, "max_depth") )         max_depth = atoi( val );           
    if( !strcmp( name, "reg_lambda") )        reg_lambda = (float)atof( val );
    if( !strcmp( name, "reg_method") )        reg_method = (float)atof( val );
    if( !strcmp( name, "subsample") )         subsample  = (float)atof( val );
    if( !strcmp( name, "use_layerwise") )     use_layerwise = atoi( val );
    if( !strcmp( name, "nthread") )           nthread = atoi( val );
    if( !strcmp( name, "default_direction") ) {
      if( !strcmp( val, "learn") )  default_direction = 0;
      if( !strcmp( val, "left") )   default_direction = 1;
      if( !strcmp( val, "right") )  default_direction = 2;
    }
  }
};

/*! \brief node statistics used in regression tree */
struct RTreeNodeStat {
  /*! \brief loss chg caused by current split */
  float loss_chg;
  /*! \brief sum of hessian values, used to measure coverage of data */
  float sum_hess;
  /*! \brief weight of current node */
  float base_weight;
  /*! \brief number of child that is leaf node known up to now */
  int leaf_child_cnt;
  /*! \brief print information of current stats to fo */
  inline void Print(FILE *fo, bool is_leaf) const {
    if (!is_leaf) {
      fprintf(fo, "gain=%f,cover=%f", loss_chg, sum_hess);
    } else {
      fprintf(fo, "cover=%f", sum_hess);
    }
  }
};
/*! \brief most comment structure of regression tree */
class RegTree: public TreeModel<bst_float, RTreeNodeStat> {
};
}  // namespace gbm
}  // namespace xgboost
#endif
