#pragma once

#include <string>
#include <utility>
#include <vector>

namespace glinthawk::storage {

enum class OperationResult
{
  OK,
  NotFound,
  Error,
};

class BlobStore
{
public:
  BlobStore() {};
  virtual ~BlobStore() {}

  virtual std::pair<OperationResult, std::string> get( const std::string& key ) = 0;
  virtual OperationResult put( const std::string& key, const std::string& value ) = 0;

  virtual std::vector<std::pair<OperationResult, std::string>> get( const std::vector<std::string>& keys ) = 0;
  virtual std::vector<OperationResult> put( const std::vector<std::pair<std::string, std::string>>& kvs ) = 0;
};

} // namespace glinthawk::storage
