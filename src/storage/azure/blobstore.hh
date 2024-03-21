#pragma once

#include "../blobstore.hh"

#include <functional>
#include <thread>

#include "net/address.hh"
#include "net/http/http_request.hh"
#include "util/uri.hh"

namespace glinthawk::storage::azure {

class BlobStore : public glinthawk::storage::BlobStore
{
private:
  enum class Op
  {
    Get,
    Put,
    Remove
  };

  const size_t MAX_REQUESTS_PER_CONNECTION = 50;
  const size_t MAX_THREADS { 8 };

  const util::ParsedURI container_uri_;
  const std::string sas_token_;

  glinthawk::net::HTTPRequest make_request( const Op operation, const std::string& key, std::string&& payload = {} );

  template<Op operation, typename InputType, typename ResultType>
  void worker_thread( const size_t thread_num,
                      const std::vector<InputType>& requests,
                      std::vector<ResultType>& responses,
                      const net::Address& endpoint );

public:
  BlobStore( const std::string& container_uri, const std::string& sas_token );
  virtual ~BlobStore() {}

  virtual std::pair<OpResult, std::string> get( const std::string& key ) override;
  virtual OpResult put( const std::string& key, const std::string& value ) override;
  virtual OpResult remove( const std::string& key ) override;

  virtual std::vector<std::pair<OpResult, std::string>> get( const std::vector<std::string>& keys ) override;
  virtual std::vector<OpResult> put( const std::vector<std::pair<std::string, std::string>>& kvs ) override;
  virtual std::vector<OpResult> remove( const std::vector<std::string>& keys ) override;

  virtual std::string to_string() const override;
};

}
