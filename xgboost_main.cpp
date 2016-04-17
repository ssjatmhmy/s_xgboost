#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include "./learner/learner-inl.h"
#include "./learner/dmatrix.h"
#include "./utils/fmap.h"
#include "./utils/random.h"
#include "./utils/config.h"

namespace xgboost {
/*!
 * \brief wrapping the training process
 */
class BoostLearnTask { 
typedef learner::DMatrix DMatrix;          
 public:
  inline int Run(int argc, char *argv[]) {
    if (argc < 2) {
      printf("Usage: <config>\n"); 
      return 0;
    }
    utils::ConfigIterator itr(argv[1]);
    while (itr.Next()) {
      this->SetParam(itr.name(), itr.val());
    }
    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        this->SetParam(name, val);
      }
    }
    this->InitData();
    this->InitLearner();
    if (task == "pred") {
      this->TaskPred();
    } else {                  
      this->TaskTrain();
    }
    return 0;
  }
  inline void SetParam(const char *name, const char *val) {
    if (!strcmp("silent", name)) silent = atoi(val);
    if (!strcmp("use_buffer", name)) use_buffer = atoi(val);
    if (!strcmp("seed", name)) random::Seed(atoi(val));
    if (!strcmp("num_round", name)) num_round = atoi(val);
    if (!strcmp("save_period", name)) save_period = atoi(val);
    if (!strcmp("task", name)) task = val;
    if (!strcmp("data", name)) train_path = val;
    if (!strcmp("test:data", name)) test_path = val;
    if (!strcmp("model_in", name)) model_in = val;
    if (!strcmp("model_out", name)) model_out = val;
    if (!strcmp("model_dir", name)) model_dir_path = val;
    if (!strcmp("fmap", name)) name_fmap = val;
    if (!strcmp("name_dump", name)) name_dump = val;
    if( !strcmp("name_dumppath", name)) name_dumppath = val;
    if (!strcmp("name_pred", name)) name_pred = val;
    if (!strcmp("dump_stats", name)) dump_model_stats = atoi(val);
    if (!strncmp("eval[", name, 5)) {
      char evname[256];
      utils::Assert(sscanf(name, "eval[%[^]]", evname) == 1, 
                    "must specify evaluation name for display");
      eval_data_names.push_back(std::string(evname));
      eval_data_paths.push_back(std::string(val));
    }
    learner.SetParam(name, val);
    printf("Set Param %s = %s\n", name, val);
  }
 public:
  BoostLearnTask(void) {
    // default parameters
    silent = 0;
    use_buffer = 1;
    num_round = 10;
    save_period = 0;
    dump_model_stats = 0;
    task = "train";                
    model_in = "NULL";
    model_out = "NULL";
    name_fmap = "NULL";
    name_pred = "pred.txt";
    name_dump = "dump.txt";
    name_dumppath = "dump.path.txt";
    model_dir_path = "./";
  }
  ~BoostLearnTask(void) {
    for (size_t i = 0; i < deval.size(); ++i) {
      delete deval[i];
    }
  }
 private:
  inline void InitData (void) {
    if (name_fmap != "NULL") fmap.LoadText(name_fmap.c_str());
    if (task == "dump") return;
    if (task == "pred" || task == "dumppath") {
      data.CacheLoad(test_path.c_str(), silent!=0, use_buffer!=0);
    } else {
      // training 
      data.CacheLoad(train_path.c_str(), silent!=0, use_buffer!=0);
      utils::Assert(eval_data_names.size() == eval_data_paths.size());
      for (size_t i = 0; i < eval_data_names.size(); ++i) {
        deval.push_back(new DMatrix());
        deval.back()->CacheLoad(eval_data_paths[i].c_str(), silent!=0, use_buffer!=0);
      }
    }
    learner.SetData(&data, deval, eval_data_names);
  }
  inline void InitLearner(void) {
    if (model_in != "NULL") {
      utils::FileStream fi(utils::FopenCheck(model_in.c_str(), "rb"));
      learner.LoadModel(fi);
      fi.Close();
    } else {
      utils::Assert(task == "train", "model_in not specified");
      learner.InitModel();
    }
    learner.InitTrainer();
  }
  inline void TaskTrain(void) {
    const time_t start = time(NULL);
    unsigned long elapsed = 0;
    for (int i = 0; i < num_round; ++i) {
      elapsed = (unsigned long)(time(NULL) - start); 
      if (!silent) printf("boosting round %d, %lu sec elapsed\n", i, elapsed);
      learner.UpdateOneIter(i);
      learner.EvalOneIter(i);
      if (save_period != 0 && (i+1) % save_period == 0) {
        this->SaveModel(i);
      }
      elapsed = (unsigned long)(time(NULL) - start); 
    }
    // always save final round
    if (save_period == 0 || num_round % save_period != 0) {
      if (model_out == "NULL") {
        this->SaveModel(num_round - 1);
      } else {
        this->SaveModel(model_out.c_str());
      }
    }
    if (!silent) {
      printf("\nupdating end, %lu sec in all\n", elapsed);
    }
  }  
  inline void TaskPred(void) {
    std::vector<float> preds;
    if (!silent) printf("start prediction...\n");
    learner.Predict(preds, data);
    if (!silent) printf("writing prediction to %s\n", name_pred.c_str());
    FILE *fo = utils::FopenCheck(name_pred.c_str(), "w");
    for (size_t i = 0; i < preds.size(); ++i) {
      fprintf(fo, "%f\n", preds[i]);
    }
    fclose(fo);                
  }
  inline void SaveModel(const char *fname) const {
    utils::FileStream fo(utils::FopenCheck(fname, "wb"));
    learner.SaveModel(fo);
    fo.Close();
  }
  inline void SaveModel(int i) const {
    char fname[256];
    sprintf(fname ,"%s/%04d.model", model_dir_path.c_str(), i+1);
    this->SaveModel(fname);
  }
 private:
  /* \brief whether silent */
  int silent;
  /* \brief whether use auto binary buffer */
  int use_buffer;
  /* \brief number of boosting iterations */
  int num_round;            
  /* \brief the period to save the model, 0 means only save the final round model */
  int save_period;
  /* \brief the path of training/test data set */
  std::string train_path, test_path;
  /* \brief the path of test model file, or file to restart training */
  std::string model_in;
  /* \brief the path of final model file, to be saved */
  std::string model_out;
  /* \brief the path of directory containing the saved models */
  std::string model_dir_path;
  /* \brief task to perform */
  std::string task;
  /* \brief name of predict file */
  std::string name_pred;
  /* \brief whether dump statistics along with model */
  int dump_model_stats;
  /* \brief name of feature map */
  std::string name_fmap;
  /* \brief name of dump file */
  std::string name_dump;
  /* \brief name of dump path file */
  std::string name_dumppath;
  /* \brief the paths of validation data sets */
  std::vector<std::string> eval_data_paths;            
  /* \brief the names of the evaluation data used in output log */
  std::vector<std::string> eval_data_names;            
 private:
  DMatrix data;
  std::vector<DMatrix*> deval;
  utils::FeatMap fmap;
  learner::BoostLearner learner;
};
}  // namespace xgboost

int main(int argc, char *argv[]) {
  xgboost::BoostLearnTask tsk;
  tsk.SetParam("seed", "0");
  int ret = tsk.Run(argc, argv);
  return ret;
}
