#pragma once

#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace glinthawk {

enum class StatType
{
  Counter,
  IntDistribution,
  FloatDistribution,
  Ratio,
};

enum class Counters
{
  PromptsStarted,
  PromptsCompleted,
  TokensProcessed,
  TokensGenerated,
  StatesSent,
  StatesReceived,
  StatesProcessed,
  StatesGenerated,

  _Count,
};

enum class IntDistributions
{
  PromptLength,
  PromptLatency,
  KernelForwardTime,
  KernelPreAttentionForwardTime,
  KernelAttentionForwardTime,
  KernelPostAttentionForwardTime,
  KernelClassificationForwardTime,

  PreInference2WorkerTimeBatch,
  PreWorker2SerializeTimeBatch,
  PreSerialize2AttWorkerTimeBatch,
  AttWorker2KernelIncomingTimeBatch,
  AttKernelIncoming2ContextTimeBatch,
  AttKernelContext2BatchingTimeBatch,

  AttInference2WorkerTimeBatch,
  AttWorker2SerializeTimeBatch,
  AttSerialize2PostWorkerTimeBatch,
  PostWorker2KernelIncomingTimeBatch,
  PostKernelIncoming2BatchingTimeBatch,

  PostInference2WorkerTimeBatch,
  PostWorker2SerializeTimeBatch,

  ClsInference2WorkerTimeBatch,
  ClsWorker2SerializeTimeBatch,

  PreWorker2KernelIncomingTime,
  PreKernelIncoming2BatchingTime,
  PreInference2WorkerTime,
  PreWorker2SerializeTime,

  PreSerialize2AttWorkerTime,

  AttWorker2KernelIncomingTime,
  AttKernelIncoming2ContextTime,
  AttKernelContext2BatchingTime,
  AttInference2WorkerTime,
  AttWorker2SerializeTime,

  AttSerialize2PostWorkerTime,

  PostWorker2KernelIncomingTime,
  PostKernelIncoming2BatchingTime,
  PostInference2WorkerTime,
  PostWorker2SerializeTime,

  ClsWorker2KernelIncomingTime,
  ClsKernelIncoming2BatchingTime,
  ClsInference2WorkerTime,
  ClsWorker2SerializeTime,

  PreSerialize2AttWorkerVarTime,
  AttSerialize2PostWorkerVarTime,
  PostSerialize2ClsWorkerVarTime,
  ClsSerialize2PreWorkerVarTime,

  //  OutgoingKernelQueueingTime,
  //  OutgoingWorkerQueueingTime,
  //  NetworkTime,
  //  IncomingKernelQueueingTime,
  //  ContextAdmissionTime,
  //  AttentionQueueingTime,

  IncomingQueue,
  WaitingQueue,
  OutgoingQueue,

  ProcessingPreAttentionQueue,
  ProcessingAttentionQueue,
  ProcessingPostAttentionQueue,
  ProcessingClassificationQueue,

  AllocatedContexts,
  FreeContexts,
  EmptyContexts,

  _Count
};

enum class FloatDistributions
{
  _Count,
};

enum class Ratios
{
  _Count
};

namespace {

constexpr std::array<std::string_view, static_cast<size_t>( Counters::_Count )> counter_keys {
  "prompts_started", "prompts_completed", "tokens_processed", "tokens_generated",
  "states_sent",     "states_received",   "states_processed", "states_generated",
};

constexpr std::array<std::string_view, static_cast<size_t>( IntDistributions::_Count )> int_dist_keys {
  "prompt_length",
  "prompt_latency",
  "kernel_forward_time",
  "kernel_pre_attention_forward_time",
  "kernel_attention_forward_time",
  "kernel_post_attention_forward_time",
  "kernel_classification_forward_time",

  "pre_inference_to_worker_time_batch",
  "pre_worker_to_serialize_time_batch",
  "pre_serialize_to_att_worker_time_batch",
  "att_worker_to_kernel_incoming_time_batch",
  "att_kernel_incoming_to_context_time_batch",
  "att_context_to_batching_time_batch",

  "att_inference_to_worker_time_batch",
  "att_worker_to_serialize_time_batch",
  "att_serialize_to_post_worker_time_batch",
  "post_worker_to_kernel_incoming_time_batch",
  "post_kernel_incoming_to_batching_time_batch",

  "post_inference_to_worker_time_batch",
  "post_worker_to_serialize_time_batch",

  "classification_inference_to_worker_time_batch",
  "classification_worker_to_serialize_time_batch",

  "pre_worker_to_kernel_incoming_time",
  "pre_kernel_incoming_to_batching_time",
  "pre_inference_to_worker_time",
  "pre_worker_to_serialize_time",

  "pre_serialize_to_att_worker_time",

  "att_worker_to_kernel_incoming_time",
  "att_kernel_incoming_to_context_time",
  "att_context_to_batching_time",
  "att_inference_to_worker_time",
  "att_worker_to_serialize_time",

  "att_serialize_to_post_worker_time",

  "post_worker_to_kernel_incoming_time",
  "post_kernel_incoming_to_batching_time",
  "post_inference_to_worker_time",
  "post_worker_to_serialize_time",

  "classification_worker_to_kernel_incoming_time",
  "classification_kernel_incoming_to_batching_time",
  "classification_inference_to_worker_time",
  "classification_worker_to_serialize_time",

  "pre_serialize_to_att_worker_var_time",
  "att_serialize_to_post_worker_var_time",
  "post_serialize_to_cls_worker_var_time",
  "cls_serialize_to_pre_worker_var_time",

  //  "outgoing_kernel_queueing_time",
  //  "outgoing_worker_queueing_time",
  //  "network_time",
  //  "incoming_kernel_queueing_time",
  //  "context_admission_time",
  //  "attention_queueing_time",

  "incoming_queue",
  "waiting_queue",
  "outgoing_queue",
  "processing_pre_attention_queue",
  "processing_attention_queue",
  "processing_post_attention_queue",
  "processing_classification_queue",

  "allocated_contexts",
  "free_contexts",
  "empty_contexts",
};

constexpr std::array<std::string_view, static_cast<size_t>( FloatDistributions::_Count )> float_dist_keys {};

constexpr std::array<std::string_view, static_cast<size_t>( Ratios::_Count )> ratio_keys {};

} // namespace

class Measurement
{
private:
  template<class T>
  struct Distribution
  {
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    T sum {};
    T sum_of_squares {};
    uint64_t count {};
  };

  struct Ratio
  {
    uint64_t numerator {};
    uint64_t denominator {};
  };

  std::string name_;
  std::unordered_map<std::string, std::string> tags_ {};

  std::array<uint64_t, static_cast<size_t>( Counters::_Count )> fields_counters_ {};
  std::array<Distribution<uint64_t>, static_cast<size_t>( IntDistributions::_Count )> fields_int_distribution_ {};
  std::array<Distribution<double>, static_cast<size_t>( FloatDistributions::_Count )> fields_float_distribution_ {};
  std::array<Ratio, static_cast<size_t>( Ratios::_Count )> fields_ratio_ {};

public:
  Measurement( const std::string& name )
    : name_( name )
  {
  }

  void tag( const std::string& key, const std::string& value ) { tags_[key] = value; }

  template<Counters counter>
  void increment( const uint64_t value = 1 )
  {
    fields_counters_[static_cast<size_t>( counter )] += value;
  }

  template<Counters counter>
  size_t get()
  {
    return fields_counters_[static_cast<size_t>( counter )];
  }

  template<IntDistributions distribution>
  void add_point( const uint64_t value )
  {
    auto& dist = fields_int_distribution_[static_cast<size_t>( distribution )];
    dist.min = std::min( dist.min, value );
    dist.max = std::max( dist.max, value );
    dist.sum += value;
    dist.sum_of_squares += value * value;
    dist.count++;
  }

  template<FloatDistributions distribution>
  void add_point( const double value )
  {
    auto& dist = fields_float_distribution_[static_cast<size_t>( distribution )];
    dist.min = std::min( dist.min, value );
    dist.max = std::max( dist.max, value );
    dist.sum += value;
    dist.sum_of_squares += value * value;
    dist.count++;
  }

  template<Ratios ratio>
  void add_point( const uint64_t numerator, const uint64_t denominator )
  {
    auto& r = fields_ratio_[static_cast<size_t>( ratio )];
    r.numerator += numerator;
    r.denominator += denominator;
  }

  void zero_out()
  {
    for ( auto& value : fields_counters_ ) {
      value = {};
    }

    for ( auto& dist : fields_int_distribution_ ) {
      dist = {};
    }

    for ( auto& dist : fields_float_distribution_ ) {
      dist = {};
    }

    for ( auto& r : fields_ratio_ ) {
      r = {};
    }
  }

  std::string to_string()
  {
    std::string result = name_;
    {
      for ( const auto& [key, value] : tags_ ) {
        if ( value.empty() ) {
          continue;
        }

        result += "," + key + "=" + value;
      }
    }

    result += " ";

    size_t i = 0;
    for ( auto& value : fields_counters_ ) {
      result += std::string { counter_keys[i++] } + "=" + std::to_string( value ) + "u,";
    }

    i = 0;
    for ( const auto& dist : fields_int_distribution_ ) {
      if ( dist.count == 0 ) {
        i++;
        continue;
      }

      const double avg = dist.sum / static_cast<float>( dist.count );
      const double var = dist.sum_of_squares / static_cast<float>( dist.count ) - avg * avg;

      result += std::string { int_dist_keys[i] } + "_count=" + std::to_string( dist.count ) + "u,";
      result += std::string { int_dist_keys[i] } + "_min=" + std::to_string( dist.min ) + "u,";
      result += std::string { int_dist_keys[i] } + "_max=" + std::to_string( dist.max ) + "u,";
      result += std::string { int_dist_keys[i] } + "_avg=" + std::to_string( avg ) + ",";
      result += std::string { int_dist_keys[i] } + "_var=" + std::to_string( var ) + ",";

      i++;
    }

    i = 0;
    for ( const auto& dist : fields_float_distribution_ ) {
      if ( dist.count == 0 ) {
        i++;
        continue;
      }

      const double avg = dist.sum / static_cast<float>( dist.count );
      const double var = dist.sum_of_squares / static_cast<float>( dist.count ) - avg * avg;

      result += std::string { float_dist_keys[i] } + "_count=" + std::to_string( dist.count ) + ",";
      result += std::string { float_dist_keys[i] } + "_min=" + std::to_string( dist.min ) + ",";
      result += std::string { float_dist_keys[i] } + "_max=" + std::to_string( dist.max ) + ",";
      result += std::string { float_dist_keys[i] } + "_avg=" + std::to_string( avg ) + ",";
      result += std::string { float_dist_keys[i] } + "_var=" + std::to_string( var ) + ",";

      i++;
    }

    i = 0;
    for ( const auto& r : fields_ratio_ ) {
      if ( r.denominator == 0 or r.numerator == 0 ) {
        // XXX revisit this
        result += std::string { ratio_keys[i] } + "=0,";
        i++;
        continue;
      }

      const auto ratio = static_cast<double>( r.numerator ) / static_cast<double>( r.denominator );
      result += std::string { ratio_keys[i] } + "_num=" + std::to_string( ratio ) + "u,";

      i++;
    }

    result.back() = '\n';
    return result;
  }
};

// XXX(sadjad): this is not thread-safe; however, for now, I'm going to ignore thread-safety
inline Measurement& global_measurement()
{
  static Measurement the_global_measurement { "glinthawk" };
  return the_global_measurement;
}

} // namespace glinthawk::monitoring
