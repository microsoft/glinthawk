#include "llama2.hh"

#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "util/timer.hh"

using namespace std;
using namespace glinthawk;

Llama2::Llama2( const filesystem::path& weights_path ) { init_weights( weights_path ); }

void Llama2::init_weights( const filesystem::path& weights_path )
{
  GlobalScopeTimer<Timer::Category::LoadingWeights> _;

  ifstream weights_file { weights_path, ios::binary };

  if ( not weights_file.good() ) {
    throw runtime_error( "Could not open weights file: " + weights_path.string() );
  }

  // get weights_file size
  weights_file.seekg( 0, ios::end );
  const auto weights_file_size = weights_file.tellg();

  LOG( INFO ) << "Weights file size: " << weights_file_size << " bytes";

  // allocate the buffer
  weights_buffer_ = make_unique<float[]>( weights_file_size / sizeof( float ) );

  LOG( INFO ) << "Allocated " << weights_file_size / sizeof( float ) << " floats for weights buffer.";

  // read the weights
  weights_file.seekg( 0, ios::beg );

  if ( weights_file.read( reinterpret_cast<char*>( weights_buffer_.get() ), weights_file_size ) ) {
    LOG( INFO ) << "Read weights file successfully.";
  } else {
    throw runtime_error( "Failed to read the weights file." );
  }
}
