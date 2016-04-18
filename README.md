# s_xgboost (Unfinished)

近期阅读dmlc/xgboost的源码，被里面分布式相关的rabit和dmlc-core带得有点晕。所以着手把这些分布式代码清掉，试图还原成单机版使得代码更易读。过程中参考了xgboost最初版本v0.1。虽然xgboost v0.1也是单机版，但代码组织已经和现在最新的dmlc/xgboost有明显区别了。  

这份代码的组织接近最新版dmlc/xgboost，同时分布式部分的代码也改为了单机代码。另外，这份代码只包含C++实现的train和predict功能，去掉了其它相对不重要的支干代码以及python，R的wrapper等。项目名s_xgboost前面的s是stand-alone的意思。也许它能给想阅读xgboost源码，了解其算法实现，又暂时不想看rabit和dmlc-core的同学一些帮助。同时，对于想用C++写单机机器学习程序的人来说，里面的一些代码也可供参考。  

使用方法：  
进入主目录输入make进行编译。如果想一边调试一边读，可以从demo里的例子开始（参考runexp.sh）。
