#include "../blobstore.hh"

#include <filesystem>
#include <string>
#include <vector>

namespace glinthawk::storage::local {

class BlobStore : public glinthawk::storage::BlobStore
{
private:
  const std::filesystem::path root_;

public:
  BlobStore( const std::filesystem::path& root );
  virtual ~BlobStore() {}

  virtual std::pair<OpResult, std::string> get( const std::string& key ) override;
  virtual OpResult put( const std::string& key, const std::string& value ) override;
  virtual OpResult remove( const std::string& key ) override;

  virtual std::vector<std::pair<OpResult, std::string>> get( const std::vector<std::string>& keys ) override;
  virtual std::vector<OpResult> put( const std::vector<std::pair<std::string, std::string>>& kvs ) override;
  virtual std::vector<OpResult> remove( const std::vector<std::string>& keys ) override;

  virtual std::string to_string() const override;
};

} // namespace glinthawk::storage::local
