#ifndef XGBOOST_UTILS_IO_H_
#define XGBOOST_UTILS_IO_H_

#include <cstdio>
/*!
 * \file xgboost_stream.h
 * \brief general stream interface for serialization
 */
namespace xgboost {
namespace utils {
/*! 
* \brief interface of stream I/O, used to serialize model 
*/
class IStream {
 public:
  /*! 
   * \brief read data from stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   * \return usually is the size of data readed
   */
  virtual size_t Read(void *ptr, size_t size) = 0;        
  /*! 
   * \brief write data to stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}
};

/*! \brief implementation of file i/o stream */
class FileStream: public IStream {
 public:
  explicit FileStream(std::FILE *fp) : fp(fp) {}
  FileStream(void) {
    this->fp = NULL;
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  inline void Close(void) {
    if (fp != NULL) {
      std::fclose(fp); fp = NULL;
    }
  }
  
 private:
  std::FILE *fp;  
};
}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_IO_H_
