//
// Created by omer_siton on 28/04/2022.
//
#include <atomic>
#include <cstdio>
#include "MapReduceFramework.h"
#include "pthread.h"
#include <algorithm>
#include <iostream>
#include "Barrier.h"
#include "Barrier.cpp"


#define ERR_MSG_OUTVEC_MUTEX "system error: pthread_mutex_init 1 failed \n"
#define ERR_MSG_WAIT_MUTEX "system error: mutex_wait_job failed \n"

#define ERR_MSG_CREATE_THREAD "system error: pthread_create failed \n"
#define ERR_MSG_JOIN_THREAD "system error: pthread_join failed \n"
#define DEF_ONE_HUNDREND 100

/**
 * this struct binds the data of ThreadContext that run now
 */
typedef struct {
    int threadID;
    IntermediateVec *intermediate_vec;
    JobHandle job_handle;
    bool joined;

}ThreadContext;

/**
 * this struct binds the data of JobContext that run now
 */
typedef struct {
    std::vector<ThreadContext *> * thread_contexts;
    pthread_t * all_threads;
    JobState * job_state;
    Barrier * barrier;
    const MapReduceClient * client;
    const InputVec * input_vector;
    std::vector<std::vector<IntermediatePair>> * shuffled_vector;
    OutputVec * output_vector;

    std::atomic<uint32_t> * atomic_counter_map_next_pair;
    std::atomic<uint32_t> * atomic_counter_input_elem_mapped;
    std::atomic<uint32_t> * atomic_counter_pairs_mapped;
    std::atomic<uint32_t> * atomic_counter_pairs_shuffled;
    std::atomic<uint32_t> * atomic_counter_pairs_reduced;
    std::atomic<uint32_t> * atomic_counter_reduce_next_vec ;

    pthread_mutex_t * mutex_out_vec;
    pthread_mutex_t * mutex_wait_job;

}JobContext;

/**
 * In this phase each thread reads pairs of (k1,  v1) from the input vector and calls the map function
    on each of them
 * @param tc thread context
 * @param jc Job Context
 */
void map_stage (ThreadContext * tc, JobContext * jc)
{
    while (true){
        if (jc->job_state->stage == UNDEFINED_STAGE)
        {
            jc->job_state->stage = MAP_STAGE;
        }
        unsigned long input_vec_size = jc->input_vector->size();
        uint32_t old_value = (*jc->atomic_counter_map_next_pair)++; //return the old pair
        if (old_value < input_vec_size){
            InputPair input_pair = jc->input_vector->at(old_value);
            jc->client->map(input_pair.first, input_pair.second, tc);
            (*jc->atomic_counter_input_elem_mapped)++;
        }
        else
          break;
    }
}

/**
 * the function sort that each thread will sort its intermediate vector according to the keys within.
 * @param lhs first pair
 * @param rhs second pair
 * @return true if key of rhs is bigger, else false
 */
bool compare_pairs(const IntermediatePair &lhs, const IntermediatePair &rhs){
    return (*lhs.first) < (*rhs.first);
}

/**
 * sort the thread context intermediate vector
 * @param tc Thread Context
 */
void sort_stage(ThreadContext * tc){
    std::sort(tc->intermediate_vec->begin(), tc->intermediate_vec->end(), compare_pairs);
}

/**
 * create new sequences of (k2, v2) where in each sequence all
    keys are identical and all elements with a given key are in a single sequence.
 * @param tc Thread Context
 * @param jc job Context
 */
void shuffle_stage(ThreadContext * tc, JobContext * jc)
{
    if (tc->threadID == 0){
        unsigned long total_pairs = jc->atomic_counter_pairs_mapped->load();
        jc->job_state->stage = SHUFFLE_STAGE;
        jc->job_state->percentage = 0;

        auto * shuffledVector = new std::vector<std::vector<IntermediatePair>>;
        IntermediatePair max_pair, tmp_pair;
        while ((jc->atomic_counter_pairs_shuffled->load()) < total_pairs){
            std::vector<IntermediatePair> tmp;
            // First step: init max_pair with some valid value
            for (auto t: *(jc->thread_contexts)){
                if (!(t->intermediate_vec->empty()))
                {
                    max_pair = t->intermediate_vec->back();
                    break;
                }
            }
            // Second step: find the maxK2 key and assign its pair to max_pair
            for (auto t : *(jc->thread_contexts)){
                if (!(t->intermediate_vec->empty())){
                    tmp_pair = t->intermediate_vec->back();
                    if(*(max_pair.first) < *(tmp_pair.first)){ //check keys
                        max_pair = tmp_pair;
                    }
                }
            }
            // Third step: find all equal pairs and create a vector of all of them
            for (auto t : *(jc->thread_contexts)){
                if (!(t->intermediate_vec->empty())){
                    IntermediatePair cur_pair = t->intermediate_vec->back();
                    while ((!(t->intermediate_vec->empty())) && (!(*(cur_pair.first) < *(max_pair.first)))){ // equal =
                        tmp.push_back(t->intermediate_vec->back());
                        t->intermediate_vec->pop_back();
                        (*jc->atomic_counter_pairs_shuffled)++; // inc the pairs shuffled done
                        cur_pair = t->intermediate_vec->back();
                    }
                }
            }
            shuffledVector->push_back(tmp);
        }
        jc->shuffled_vector = shuffledVector;
    }
}

/**
 * pop a vector from the back of the queue and run reduce on it
 * @param jc Job Context
 */
void reduce_stage(JobContext * jc)
{
    while (true){
        if (jc->job_state->stage == SHUFFLE_STAGE){
            jc->job_state->stage = REDUCE_STAGE;
            jc->job_state->percentage = 0;
        }
        unsigned long shuffled_vector_size = jc->shuffled_vector->size();
        uint32_t old_vec = (*jc->atomic_counter_reduce_next_vec)++; //return the old vector index
        if (old_vec < shuffled_vector_size){ // until the threads take all the vectors
            IntermediateVec tmp_vector = jc->shuffled_vector->at(old_vec);
            jc->client->reduce(&tmp_vector, jc);
            (*jc->atomic_counter_pairs_reduced) += tmp_vector.size();
        }
        else
          break;
    }
}

/**
 * This is the function we send to all threads that it needs to perform when it is created
 * @param context convert to threadContext
 */
void *  ThreadMapReduce(void* context){
    ThreadContext * threadContext = static_cast<ThreadContext*>(context);
    JobContext * jobContext = static_cast<JobContext *>(threadContext->job_handle);
    //Map Phase
    map_stage(threadContext, jobContext);
    //Sort Phase
    sort_stage(threadContext);
    //Barrier
    jobContext->barrier->barrier();
    //Shuffle Phase
    shuffle_stage(threadContext, jobContext);
    //Barrier
    jobContext->barrier->barrier();
    //Reduce Phase
    reduce_stage(jobContext);
}

/**
 * The function receives as input intermediary element (K2, V2) and context which contains data structure of the thread
 * that created the intermediary element. The function saves the intermediary element in the context data structures.
 * In addition, the function updates the number of intermediary elements using atomic counter.
 * @param key key of pair
 * @param value value of pair
 * @param context convert to threadContext -  the thread run now
 */
void emit2 (K2* key, V2* value, void* context){
    ThreadContext * threadContext = static_cast<ThreadContext*>(context);
    JobContext * jobContext = static_cast<JobContext *>(threadContext->job_handle);
    threadContext->intermediate_vec->push_back(IntermediatePair(key, value));
    (*jobContext->atomic_counter_pairs_mapped)++; //inc the pairs done
}

/**
 * The function receives as input output element (K3, V3) and context which contains data structure of the thread that
 * created the output element. The function saves the output element in the context data structures (output vector).
 * In addition, the function updates the number of output elements using atomic counter.
 * @param key key of pair
 * @param value value of pair
 * @param context convert to jobContext - the job now
 */
void emit3 (K3* key, V3* value, void* context){
    JobContext * jobContext = static_cast<JobContext *>(context);
    pthread_mutex_lock(jobContext->mutex_out_vec);
    jobContext->output_vector->push_back(OutputPair(key, value));
    pthread_mutex_unlock(jobContext->mutex_out_vec);
}

/**
 * This function starts running the MapReduce algorithm (with several threads) and returns a JobHandle.
 * @param client The implementation of MapReduceClient or in other words the task that the framework should run.
 * @param inputVec a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec a vector of type std::vector<std::pair<K3*, V3*>>, to which the output elements will be added before
 * returning
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm.
 * @return The function returns JobHandle that will be used for monitoring the job
 */
JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){

    pthread_t * threads = new pthread_t[multiThreadLevel];
    std::vector<ThreadContext *> * threadContextsVector = new std::vector<ThreadContext *>;
    auto * barrier = new Barrier(multiThreadLevel);
    JobState * jobState = new JobState {UNDEFINED_STAGE, 0};

    /// Atomics
    auto * atomic_counter_next_pair_map = new std::atomic<uint32_t>(0);
    auto * atomic_counter_elem_input_map = new std::atomic<uint32_t>(0);
    auto * atomic_counter_pairs_mapped = new std::atomic<uint32_t>(0);
    auto * atomic_counter_pairs_shuffled = new std::atomic<uint32_t>(0);
    auto * atomic_counter_pairs_reduced = new std::atomic<uint32_t>(0);
    auto * atomic_counter_next_vec_reduce = new std::atomic<uint32_t>(0);

    ///Mutex
    pthread_mutex_t * mutex_out_vector = new pthread_mutex_t;
    if (pthread_mutex_init(mutex_out_vector, NULL) != 0 )
    {
        std::cerr << ERR_MSG_OUTVEC_MUTEX << std::endl;
        exit(1);
    }
    pthread_mutex_t * mutex_wait_job = new pthread_mutex_t;
    if (pthread_mutex_init(mutex_wait_job, NULL) != 0 )
    {
        std::cerr << ERR_MSG_WAIT_MUTEX << std::endl;
        exit(1);
    }

    JobContext * jobContext = new JobContext {threadContextsVector, threads, jobState, barrier, &client, &inputVec,
                                              nullptr, &outputVec, atomic_counter_next_pair_map, atomic_counter_elem_input_map,
                                              atomic_counter_pairs_mapped, atomic_counter_pairs_shuffled, atomic_counter_pairs_reduced,
                                              atomic_counter_next_vec_reduce, mutex_out_vector, mutex_wait_job};

    for (int i = 0; i < multiThreadLevel; ++i){
        ThreadContext * threadContext = new ThreadContext {i, new std::vector<IntermediatePair>, jobContext, false};
        threadContextsVector->push_back(threadContext);
        if (pthread_create(threads + i, NULL, ThreadMapReduce, threadContext) != 0)
        {
            std::cerr << ERR_MSG_CREATE_THREAD  << std::endl;
            exit(1);
        }
    }
    return (JobHandle) jobContext;
}

/**
 * a function gets JobHandle returned by startMapReduceFramework and waits until it is finished.
 * @param job jobContext
 */
void waitForJob(JobHandle job){
    JobContext * jobContext = static_cast<JobContext *>(job);
    pthread_mutex_lock (jobContext->mutex_wait_job);
    for (unsigned long i=0; i < jobContext->thread_contexts->size(); i++)
    {
        if (!(jobContext->thread_contexts->at(i)->joined))
        {
            if (pthread_join(jobContext->all_threads[i], NULL) != 0) {
                std::cerr << ERR_MSG_JOIN_THREAD  << std::endl;
                exit(1);
            }
            jobContext->thread_contexts->at(i)->joined = true;
        }
    }
    pthread_mutex_unlock (jobContext->mutex_wait_job);
}

/**
 *  this function gets a JobHandle and updates the state of the job into the given JobState struct.
 * @param job jobContext
 * @param state the state that we want
 */
void getJobState(JobHandle job, JobState* state){
    JobContext * jobContext = static_cast<JobContext *>(job);
    unsigned long total = jobContext->input_vector->size();
    state->stage = jobContext->job_state->stage;

    if (state->stage == MAP_STAGE)
    {
        uint32_t mapped_elem = jobContext->atomic_counter_input_elem_mapped->load();
        state->percentage= (float) mapped_elem * DEF_ONE_HUNDREND / (float)(total);
    }
    if (state->stage == SHUFFLE_STAGE){
        uint32_t total_pairs_mapped = jobContext->atomic_counter_pairs_mapped->load();
        uint32_t shuffled_pairs = jobContext->atomic_counter_pairs_shuffled->load();
        state->percentage = (float) shuffled_pairs * DEF_ONE_HUNDREND / (float) total_pairs_mapped;
    }
    if (state->stage == REDUCE_STAGE){
        uint32_t total_pairs_mapped = jobContext->atomic_counter_pairs_mapped->load();
        uint32_t reduced_pairs = jobContext->atomic_counter_pairs_reduced->load();
        state->percentage = (float) reduced_pairs * DEF_ONE_HUNDREND / (float) total_pairs_mapped;
    }
}

/**
 * eleasing all resources of a job. You should prevent releasing resources before the job finished. After this function
 * is called the job handle will be invalid.
 * @param job jobContext
 */
void closeJobHandle(JobHandle job){
    JobContext * jobContext = static_cast<JobContext *>(job);
    waitForJob(job); // wait until the job ends
    unsigned long num_all_threads = jobContext->thread_contexts->size();
    for (unsigned  long i=0; i < num_all_threads; i++)
    {
        delete jobContext->thread_contexts->at(i)->intermediate_vec;
        delete jobContext->thread_contexts->at(i);
    }
    delete jobContext->thread_contexts;
    delete jobContext->all_threads;
    delete jobContext->job_state;
    delete jobContext->barrier;
    delete jobContext->shuffled_vector;

    delete jobContext->atomic_counter_map_next_pair;
    delete jobContext->atomic_counter_input_elem_mapped;
    delete jobContext->atomic_counter_pairs_mapped;
    delete jobContext->atomic_counter_pairs_shuffled;
    delete jobContext->atomic_counter_pairs_reduced;
    delete jobContext->atomic_counter_reduce_next_vec;

    pthread_mutex_destroy(jobContext->mutex_out_vec);
    pthread_mutex_destroy(jobContext->mutex_wait_job);
}


