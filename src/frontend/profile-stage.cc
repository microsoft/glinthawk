#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>

#include <glog/logging.h>

#include "models/llama2/model.hh"

#include "util/timer.hh"

#define OOF_IMPL
#include "oof/oof.hh"

#include "platform_macros.hh"

using namespace std;
using namespace glinthawk;

static void signal_handler( int )
{
  cerr << endl << global_timer().summary() << endl;
  exit( EXIT_FAILURE );
}

void usage( const char* argv0 )
{
  cout << "Usage: " << argv0 << " <model_root> <stage=(pre|att|post)> <batch_size>" << endl;
}

int main( int argc, char* argv[] )
{
  if ( argc <= 0 ) {
    abort();
  }

  if ( argc != 4 ) {
    usage( argv[0] );
    return EXIT_FAILURE;
  }

  signal( SIGINT, signal_handler );

  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging( argv[0] );

  try {
    const filesystem::path model_dir { argv[1] };
    // const string model_name { argv[2] }; // XXX let's fix the model to Llama2_70B_Chat for now
    const string stage { argv[2] };
    const uint64_t batch_size = atoi( argv[3] );
    const int start_layer = 1;
    const int end_layer = 1;

    using ModelType = models::llama2::_GLINTHAWK_ARCH_NS_::Llama2_70B_Chat<_GLINTHAWK_DTYPE_>;
    ModelType model { model_dir, start_layer, end_layer, batch_size, batch_size, true /* random params */ };

    using ContextType = ModelType::ContextType;
    vector<shared_ptr<ContextType>> contexts( batch_size );
    vector<models::InferenceState> states( batch_size );

    models::common::_GLINTHAWK_ARCH_NS_::Operations<_GLINTHAWK_DTYPE_> ops;

    for(size_t i = 0; i < batch_size; i++) {
      contexts[i] = make_shared<ContextType>( model.settings() );
      for ( size_t i = 0; i < )
    }



    cerr << endl << global_timer().summary() << endl;
  } catch ( const exception& e ) {
    cerr << "Error: " << e.what() << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
